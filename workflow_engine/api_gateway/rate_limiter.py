import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import redis
from fastapi import Request, HTTPException
from sqlalchemy.orm import Session

from ..database.connection import db_manager
from ..database.models import APIRateLimit
from ..config.settings import settings


class RateLimitType(Enum):
    """Rate limit identifier types."""
    IP_ADDRESS = "ip"
    USER_ID = "user"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"


@dataclass
class RateLimitRule:
    """Rate limit rule definition."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int = None  # Maximum requests in a short burst
    block_duration_minutes: int = 15  # How long to block when limit exceeded


class RateLimiter:
    """Advanced rate limiting with multiple strategies."""
    
    def __init__(self):
        self.redis_client = db_manager.get_redis()
        self.in_memory_store = {}  # Fallback for when Redis is not available
        
        # Default rate limit rules by endpoint pattern
        self.default_rules = {
            "/auth/login": RateLimitRule(10, 50, 200, burst_limit=3),
            "/auth/register": RateLimitRule(5, 10, 20, burst_limit=2),
            "/auth/forgot-password": RateLimitRule(3, 5, 10, burst_limit=1),
            "/api/workflows": RateLimitRule(100, 1000, 10000, burst_limit=20),
            "/api/applications": RateLimitRule(200, 2000, 20000, burst_limit=50),
            "default": RateLimitRule(60, 1000, 10000, burst_limit=10)
        }
    
    def _get_redis_key(self, identifier: str, identifier_type: RateLimitType, 
                      endpoint: str, window: str) -> str:
        """Generate Redis key for rate limit tracking."""
        return f"rate_limit:{identifier_type.value}:{identifier}:{endpoint}:{window}"
    
    def _get_rate_limit_rule(self, endpoint: str) -> RateLimitRule:
        """Get rate limit rule for endpoint."""
        # Check for exact match first
        if endpoint in self.default_rules:
            return self.default_rules[endpoint]
        
        # Check for pattern matches
        for pattern, rule in self.default_rules.items():
            if pattern != "default" and endpoint.startswith(pattern):
                return rule
        
        # Return default rule
        return self.default_rules["default"]
    
    def _get_current_window(self, window_type: str) -> str:
        """Get current time window for rate limiting."""
        now = datetime.now(timezone.utc)
        
        if window_type == "minute":
            return now.strftime("%Y-%m-%d-%H-%M")
        elif window_type == "hour":
            return now.strftime("%Y-%m-%d-%H")
        elif window_type == "day":
            return now.strftime("%Y-%m-%d")
        elif window_type == "burst":
            # 10-second window for burst detection
            window_second = now.second // 10 * 10
            return now.strftime(f"%Y-%m-%d-%H-%M-{window_second:02d}")
        else:
            raise ValueError(f"Invalid window type: {window_type}")
    
    def _increment_counter(self, key: str, window_duration_seconds: int) -> int:
        """Increment rate limit counter in Redis or in-memory fallback."""
        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, window_duration_seconds)
                results = pipe.execute()
                return results[0]  # Return the incremented value
            except Exception:
                pass  # Fall back to in-memory
        
        # In-memory fallback
        import time
        now = time.time()
        if key not in self.in_memory_store:
            self.in_memory_store[key] = {'count': 0, 'expires': now + window_duration_seconds}
        
        if now > self.in_memory_store[key]['expires']:
            self.in_memory_store[key] = {'count': 1, 'expires': now + window_duration_seconds}
        else:
            self.in_memory_store[key]['count'] += 1
        
        return self.in_memory_store[key]['count']
    
    def _check_burst_limit(self, identifier: str, identifier_type: RateLimitType,
                          endpoint: str, rule: RateLimitRule) -> bool:
        """Check if request exceeds burst limit."""
        if not rule.burst_limit:
            return True
        
        burst_window = self._get_current_window("burst")
        burst_key = self._get_redis_key(identifier, identifier_type, endpoint, f"burst_{burst_window}")
        
        current_count = self._increment_counter(burst_key, 10)  # 10-second window
        
        return current_count <= rule.burst_limit
    
    def check_rate_limit(self, request: Request, user_id: Optional[str] = None,
                        api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if request should be rate limited.
        
        Returns:
            dict: Rate limit status with remaining requests and reset time
        """
        endpoint = request.url.path
        ip_address = self._get_client_ip(request)
        
        rule = self._get_rate_limit_rule(endpoint)
        
        # Determine primary identifier for rate limiting
        if user_id:
            primary_identifier = user_id
            identifier_type = RateLimitType.USER_ID
        elif api_key:
            primary_identifier = api_key
            identifier_type = RateLimitType.API_KEY
        else:
            primary_identifier = ip_address
            identifier_type = RateLimitType.IP_ADDRESS
        
        # Check burst limit first
        if not self._check_burst_limit(primary_identifier, identifier_type, endpoint, rule):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded: Too many requests in a short time",
                headers={"Retry-After": "10"}
            )
        
        # Check per-minute limit
        minute_window = self._get_current_window("minute")
        minute_key = self._get_redis_key(primary_identifier, identifier_type, endpoint, f"minute_{minute_window}")
        minute_count = self._increment_counter(minute_key, 60)
        
        if minute_count > rule.requests_per_minute:
            self._handle_rate_limit_exceeded(primary_identifier, identifier_type, endpoint, "minute")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded: Too many requests per minute",
                headers={"Retry-After": "60"}
            )
        
        # Check per-hour limit
        hour_window = self._get_current_window("hour")
        hour_key = self._get_redis_key(primary_identifier, identifier_type, endpoint, f"hour_{hour_window}")
        hour_count = self._increment_counter(hour_key, 3600)
        
        if hour_count > rule.requests_per_hour:
            self._handle_rate_limit_exceeded(primary_identifier, identifier_type, endpoint, "hour")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded: Too many requests per hour",
                headers={"Retry-After": "3600"}
            )
        
        # Check per-day limit
        day_window = self._get_current_window("day")
        day_key = self._get_redis_key(primary_identifier, identifier_type, endpoint, f"day_{day_window}")
        day_count = self._increment_counter(day_key, 86400)
        
        if day_count > rule.requests_per_day:
            self._handle_rate_limit_exceeded(primary_identifier, identifier_type, endpoint, "day")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded: Too many requests per day",
                headers={"Retry-After": "86400"}
            )
        
        # Return rate limit status
        return {
            "allowed": True,
            "limits": {
                "requests_per_minute": rule.requests_per_minute,
                "requests_per_hour": rule.requests_per_hour,
                "requests_per_day": rule.requests_per_day
            },
            "current": {
                "minute": minute_count,
                "hour": hour_count,
                "day": day_count
            },
            "remaining": {
                "minute": max(0, rule.requests_per_minute - minute_count),
                "hour": max(0, rule.requests_per_hour - hour_count),
                "day": max(0, rule.requests_per_day - day_count)
            }
        }
    
    def _handle_rate_limit_exceeded(self, identifier: str, identifier_type: RateLimitType,
                                   endpoint: str, window_type: str):
        """Handle rate limit exceeded event."""
        # Log to database for monitoring and analysis
        try:
            with db_manager.get_session() as session:
                rate_limit_entry = APIRateLimit(
                    identifier=identifier,
                    identifier_type=identifier_type.value,
                    endpoint=endpoint,
                    request_count=0,  # Will be updated
                    window_start=datetime.now(timezone.utc),
                    window_end=datetime.now(timezone.utc) + timedelta(minutes=1),
                    is_blocked=True,
                    blocked_until=datetime.now(timezone.utc) + timedelta(minutes=15)
                )
                session.add(rate_limit_entry)
                session.commit()
        except Exception:
            # Don't fail the rate limiting if logging fails
            pass
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (load balancer/proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def is_blocked(self, identifier: str, identifier_type: RateLimitType) -> bool:
        """Check if identifier is currently blocked."""
        if self.redis_client:
            try:
                block_key = f"rate_limit_block:{identifier_type.value}:{identifier}"
                return bool(self.redis_client.get(block_key))
            except Exception:
                pass
        
        # In-memory fallback
        block_key = f"rate_limit_block:{identifier_type.value}:{identifier}"
        if block_key in self.in_memory_store:
            import time
            now = time.time()
            if now > self.in_memory_store[block_key].get('expires', 0):
                del self.in_memory_store[block_key]
                return False
            return True
        return False
    
    def block_identifier(self, identifier: str, identifier_type: RateLimitType,
                        duration_minutes: int = 15):
        """Block an identifier for specified duration."""
        if self.redis_client:
            try:
                block_key = f"rate_limit_block:{identifier_type.value}:{identifier}"
                self.redis_client.setex(block_key, duration_minutes * 60, "blocked")
                return
            except Exception:
                pass
        
        # In-memory fallback
        import time
        block_key = f"rate_limit_block:{identifier_type.value}:{identifier}"
        self.in_memory_store[block_key] = {
            'blocked': True,
            'expires': time.time() + (duration_minutes * 60)
        }
    
    def unblock_identifier(self, identifier: str, identifier_type: RateLimitType):
        """Remove block for an identifier."""
        if self.redis_client:
            try:
                block_key = f"rate_limit_block:{identifier_type.value}:{identifier}"
                self.redis_client.delete(block_key)
                return
            except Exception:
                pass
        
        # In-memory fallback
        block_key = f"rate_limit_block:{identifier_type.value}:{identifier}"
        if block_key in self.in_memory_store:
            del self.in_memory_store[block_key]
    
    def get_rate_limit_stats(self, identifier: str, identifier_type: RateLimitType,
                           endpoint: str) -> Dict[str, Any]:
        """Get current rate limit statistics for an identifier."""
        stats = {}
        
        for window_type, duration in [("minute", 60), ("hour", 3600), ("day", 86400)]:
            window = self._get_current_window(window_type)
            key = self._get_redis_key(identifier, identifier_type, endpoint, f"{window_type}_{window}")
            
            if self.redis_client:
                try:
                    count = self.redis_client.get(key)
                    stats[window_type] = int(count) if count else 0
                    continue
                except Exception:
                    pass
            
            # In-memory fallback
            if key in self.in_memory_store:
                import time
                now = time.time()
                if now <= self.in_memory_store[key].get('expires', 0):
                    stats[window_type] = self.in_memory_store[key].get('count', 0)
                else:
                    stats[window_type] = 0
            else:
                stats[window_type] = 0
        
        return stats


# Global rate limiter instance
rate_limiter = RateLimiter()