import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse

from ..auth.security import security_manager, TokenData
from ..database.connection import db_manager
from ..database.models import AuditLog
from ..monitoring.logger import logger
from .rate_limiter import rate_limiter, RateLimitType


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for authentication, authorization, and audit logging."""
    
    def __init__(self, app):
        super().__init__(app)
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/health", "/metrics", "/docs", "/redoc", "/openapi.json",
            "/auth/login", "/auth/register", "/auth/forgot-password"
        }
        
        # Endpoints that require specific permissions
        self.protected_endpoints = {
            "/api/v1/workflows": ["workflow.read"],
            "/api/v1/workflows/create": ["workflow.create"],
            "/api/v1/applications": ["application.read"],
            "/api/v1/users": ["user.read"],
            "/api/v1/admin": ["system.configure"]
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request ID to response headers
        start_time = time.time()
        
        try:
            # Skip authentication for public endpoints
            if self._is_public_endpoint(request.url.path):
                response = await call_next(request)
                self._add_security_headers(response)
                return response
            
            # Authenticate request
            token_data = await self._authenticate_request(request)
            if not token_data:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing authentication token"},
                    headers={"X-Request-ID": request_id}
                )
            
            # Store user info in request state
            request.state.user_id = token_data.user_id
            request.state.username = token_data.username
            request.state.user_roles = token_data.roles or []
            request.state.user_permissions = token_data.permissions or []
            
            # Authorize request
            if not self._authorize_request(request, token_data):
                await self._log_security_event(request, "AUTHORIZATION_FAILED", token_data.user_id)
                return JSONResponse(
                    status_code=403,
                    content={"error": "Insufficient permissions"},
                    headers={"X-Request-ID": request_id}
                )
            
            # Process request
            response = await call_next(request)
            
            # Log successful request
            await self._log_audit_event(request, response, token_data.user_id)
            
            # Add security headers
            self._add_security_headers(response)
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except HTTPException as e:
            # Log HTTP exceptions
            await self._log_security_event(request, "HTTP_EXCEPTION", getattr(request.state, 'user_id', None), str(e))
            response = JSONResponse(
                status_code=e.status_code,
                content={"error": e.detail},
                headers={"X-Request-ID": request_id}
            )
            self._add_security_headers(response)
            return response
            
        except Exception as e:
            # Log unexpected errors
            await self._log_security_event(request, "INTERNAL_ERROR", getattr(request.state, 'user_id', None), str(e))
            logger.error("Unexpected error in security middleware", error=str(e), request_id=request_id)
            
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
                headers={"X-Request-ID": request_id}
            )
            self._add_security_headers(response)
            return response
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public and doesn't require authentication."""
        return any(path.startswith(endpoint) for endpoint in self.public_endpoints)
    
    async def _authenticate_request(self, request: Request) -> Optional[TokenData]:
        """Authenticate request using JWT token."""
        # Get token from Authorization header
        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            return None
        
        token = authorization[7:]  # Remove "Bearer " prefix
        return security_manager.verify_token(token, "access")
    
    def _authorize_request(self, request: Request, token_data: TokenData) -> bool:
        """Authorize request based on user permissions."""
        path = request.url.path
        
        # Check if endpoint requires specific permissions
        for endpoint_pattern, required_permissions in self.protected_endpoints.items():
            if path.startswith(endpoint_pattern):
                # Check if user has any of the required permissions
                user_permissions = token_data.permissions or []
                return any(perm in user_permissions for perm in required_permissions)
        
        # If no specific permissions required, allow authenticated users
        return True
    
    async def _log_audit_event(self, request: Request, response: Response, user_id: str):
        """Log audit event for successful requests."""
        try:
            # Only log state-changing operations
            if request.method in ["POST", "PUT", "PATCH", "DELETE"]:
                with db_manager.get_session() as session:
                    audit_log = AuditLog(
                        user_id=user_id,
                        action=f"{request.method}_{request.url.path}",
                        resource_type="api_endpoint",
                        resource_id=request.url.path,
                        endpoint=str(request.url),
                        http_method=request.method,
                        ip_address=self._get_client_ip(request),
                        user_agent=request.headers.get("User-Agent"),
                        status="SUCCESS",
                        audit_metadata={
                            "response_status": response.status_code,
                            "request_id": getattr(request.state, 'request_id', None)
                        }
                    )
                    session.add(audit_log)
                    session.commit()
        except Exception as e:
            logger.error("Failed to log audit event", error=str(e))
    
    async def _log_security_event(self, request: Request, event_type: str, 
                                 user_id: Optional[str] = None, details: Optional[str] = None):
        """Log security-related events."""
        try:
            with db_manager.get_session() as session:
                audit_log = AuditLog(
                    user_id=user_id,
                    action=event_type,
                    resource_type="security",
                    endpoint=str(request.url),
                    http_method=request.method,
                    ip_address=self._get_client_ip(request),
                    user_agent=request.headers.get("User-Agent"),
                    status="FAILURE",
                    error_message=details,
                    audit_metadata={
                        "request_id": getattr(request.state, 'request_id', None)
                    }
                )
                session.add(audit_log)
                session.commit()
        except Exception as e:
            logger.error("Failed to log security event", error=str(e))
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        })
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        try:
            # Check rate limits
            user_id = getattr(request.state, 'user_id', None)
            api_key = request.headers.get("X-API-Key")
            
            rate_limit_result = rate_limiter.check_rate_limit(
                request=request,
                user_id=user_id,
                api_key=api_key
            )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers.update({
                "X-RateLimit-Limit-Minute": str(rate_limit_result["limits"]["requests_per_minute"]),
                "X-RateLimit-Remaining-Minute": str(rate_limit_result["remaining"]["minute"]),
                "X-RateLimit-Limit-Hour": str(rate_limit_result["limits"]["requests_per_hour"]),
                "X-RateLimit-Remaining-Hour": str(rate_limit_result["remaining"]["hour"])
            })
            
            return response
            
        except HTTPException:
            # Rate limit exceeded - re-raise the exception
            raise
        except Exception as e:
            # Log error but don't block request
            logger.error("Error in rate limiting middleware", error=str(e))
            return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        
        # Log request
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        logger.info(
            "Incoming request",
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params),
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("User-Agent"),
            request_id=request_id
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log response
        logger.info(
            "Request processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            processing_time_ms=round(processing_time * 1000, 2),
            request_id=request_id
        )
        
        # Add processing time header
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"