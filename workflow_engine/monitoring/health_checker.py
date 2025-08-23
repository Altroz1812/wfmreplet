import asyncio
import time
import psutil
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

import httpx
from sqlalchemy.orm import Session

from ..database.connection import db_manager
from ..database.models import SystemHealth
from ..config.settings import settings


@dataclass
class HealthStatus:
    """Health check status data class."""
    service_name: str
    status: str  # HEALTHY, DEGRADED, UNHEALTHY
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthChecker:
    """System health monitoring and checking."""
    
    def __init__(self):
        self.services = {
            "postgresql": self._check_postgresql,
            "redis": self._check_redis,
            "system_resources": self._check_system_resources,
            "api_endpoints": self._check_api_endpoints,
        }
    
    async def check_all_services(self) -> Dict[str, HealthStatus]:
        """Check health of all registered services."""
        results = {}
        
        for service_name, check_func in self.services.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    health_status = await check_func()
                else:
                    health_status = check_func()
                results[service_name] = health_status
            except Exception as e:
                results[service_name] = HealthStatus(
                    service_name=service_name,
                    status="UNHEALTHY",
                    error=str(e)
                )
        
        return results
    
    def _check_postgresql(self) -> HealthStatus:
        """Check PostgreSQL database health."""
        start_time = time.time()
        
        try:
            with db_manager.get_session() as session:
                # Test basic connectivity
                session.execute("SELECT 1")
                
                # Check active connections
                result = session.execute("""
                    SELECT count(*) as active_connections
                    FROM pg_stat_activity
                    WHERE state = 'active'
                """)
                active_connections = result.fetchone()[0]
                
                # Check for slow queries
                slow_queries = session.execute("""
                    SELECT count(*) as slow_queries
                    FROM pg_stat_activity
                    WHERE state = 'active'
                    AND query_start < NOW() - INTERVAL '30 seconds'
                    AND query != '<IDLE>'
                """).fetchone()[0]
                
                response_time = (time.time() - start_time) * 1000
                
                # Determine status based on metrics
                if response_time > 5000:  # 5 seconds
                    status = "UNHEALTHY"
                elif response_time > 1000 or slow_queries > 10:  # 1 second or 10 slow queries
                    status = "DEGRADED"
                else:
                    status = "HEALTHY"
                
                return HealthStatus(
                    service_name="postgresql",
                    status=status,
                    response_time_ms=round(response_time, 2),
                    details={
                        "active_connections": active_connections,
                        "slow_queries": slow_queries
                    }
                )
                
        except Exception as e:
            return HealthStatus(
                service_name="postgresql",
                status="UNHEALTHY",
                error=str(e)
            )
    
    def _check_redis(self) -> HealthStatus:
        """Check Redis health."""
        start_time = time.time()
        
        try:
            redis_client = db_manager.get_redis()
            
            # Test basic connectivity
            redis_client.ping()
            
            # Get Redis info
            info = redis_client.info()
            
            response_time = (time.time() - start_time) * 1000
            
            # Check memory usage
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            memory_usage_percent = (used_memory / max_memory * 100) if max_memory > 0 else 0
            
            # Determine status
            if response_time > 1000:  # 1 second
                status = "UNHEALTHY"
            elif response_time > 500 or memory_usage_percent > 80:  # 500ms or 80% memory
                status = "DEGRADED"
            else:
                status = "HEALTHY"
            
            return HealthStatus(
                service_name="redis",
                status=status,
                response_time_ms=round(response_time, 2),
                details={
                    "connected_clients": info.get('connected_clients', 0),
                    "memory_usage_percent": round(memory_usage_percent, 2),
                    "ops_per_sec": info.get('instantaneous_ops_per_sec', 0)
                }
            )
            
        except Exception as e:
            return HealthStatus(
                service_name="redis",
                status="UNHEALTHY",
                error=str(e)
            )
    
    def _check_system_resources(self) -> HealthStatus:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                load_1min = load_avg[0]
                cpu_count = psutil.cpu_count()
                load_percent = (load_1min / cpu_count) * 100 if cpu_count > 0 else 0
            except (AttributeError, OSError):
                load_percent = 0
            
            # Determine overall status
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 95:
                status = "UNHEALTHY"
            elif cpu_percent > 70 or memory_percent > 75 or disk_percent > 85 or load_percent > 80:
                status = "DEGRADED"
            else:
                status = "HEALTHY"
            
            return HealthStatus(
                service_name="system_resources",
                status=status,
                details={
                    "cpu_usage_percent": round(cpu_percent, 1),
                    "memory_usage_percent": round(memory_percent, 1),
                    "disk_usage_percent": round(disk_percent, 1),
                    "load_average_percent": round(load_percent, 1),
                    "available_memory_mb": round(memory.available / 1024 / 1024),
                    "free_disk_gb": round(disk.free / 1024 / 1024 / 1024, 1)
                }
            )
            
        except Exception as e:
            return HealthStatus(
                service_name="system_resources",
                status="UNHEALTHY",
                error=str(e)
            )
    
    async def _check_api_endpoints(self) -> HealthStatus:
        """Check critical API endpoints."""
        endpoints_to_check = [
            "/health",
            "/api/v1/workflows",
            "/auth/status"
        ]
        
        results = []
        total_response_time = 0
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint in endpoints_to_check:
                try:
                    start_time = time.time()
                    response = await client.get(f"http://localhost:{settings.api.port}{endpoint}")
                    response_time = (time.time() - start_time) * 1000
                    
                    total_response_time += response_time
                    
                    results.append({
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "response_time_ms": round(response_time, 2),
                        "healthy": response.status_code < 500
                    })
                    
                except Exception as e:
                    results.append({
                        "endpoint": endpoint,
                        "error": str(e),
                        "healthy": False
                    })
        
        # Calculate overall status
        healthy_endpoints = sum(1 for r in results if r.get("healthy", False))
        total_endpoints = len(results)
        health_ratio = healthy_endpoints / total_endpoints if total_endpoints > 0 else 0
        
        avg_response_time = total_response_time / total_endpoints if total_endpoints > 0 else 0
        
        if health_ratio < 0.5:  # Less than 50% healthy
            status = "UNHEALTHY"
        elif health_ratio < 0.8 or avg_response_time > 2000:  # Less than 80% healthy or slow
            status = "DEGRADED"
        else:
            status = "HEALTHY"
        
        return HealthStatus(
            service_name="api_endpoints",
            status=status,
            response_time_ms=round(avg_response_time, 2),
            details={
                "endpoints_checked": total_endpoints,
                "healthy_endpoints": healthy_endpoints,
                "health_ratio": round(health_ratio, 2),
                "endpoint_results": results
            }
        )
    
    async def store_health_metrics(self, health_results: Dict[str, HealthStatus]):
        """Store health check results in database."""
        try:
            with db_manager.get_session() as session:
                for service_name, health_status in health_results.items():
                    # Extract metrics for database storage
                    details = health_status.details or {}
                    
                    health_record = SystemHealth(
                        service_name=service_name,
                        status=health_status.status,
                        response_time_ms=int(health_status.response_time_ms) if health_status.response_time_ms else None,
                        cpu_usage_percent=details.get('cpu_usage_percent'),
                        memory_usage_percent=details.get('memory_usage_percent'),
                        disk_usage_percent=details.get('disk_usage_percent'),
                        active_connections=details.get('active_connections'),
                        slow_queries_count=details.get('slow_queries'),
                        health_metrics=details,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    session.add(health_record)
                session.commit()
                
        except Exception as e:
            # Don't fail health checks if storage fails
            print(f"Failed to store health metrics: {e}")
    
    def get_overall_health_status(self, health_results: Dict[str, HealthStatus]) -> str:
        """Determine overall system health status."""
        if not health_results:
            return "UNKNOWN"
        
        statuses = [result.status for result in health_results.values()]
        
        if "UNHEALTHY" in statuses:
            return "UNHEALTHY"
        elif "DEGRADED" in statuses:
            return "DEGRADED"
        elif all(status == "HEALTHY" for status in statuses):
            return "HEALTHY"
        else:
            return "UNKNOWN"


# Global health checker instance
health_checker = HealthChecker()