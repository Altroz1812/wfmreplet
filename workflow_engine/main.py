import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from .config.settings import settings
from .database.connection import db_manager
from .api_gateway.middleware import SecurityMiddleware, RateLimitMiddleware, RequestLoggingMiddleware
from .monitoring.health_checker import health_checker
from .monitoring.metrics import metrics_collector, performance_tracker
from .monitoring.logger import logger, security_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Workflow Management System")
    
    try:
        # Initialize database connections
        db_manager.initialize()
        logger.info("Database connections initialized")
        
        # Run initial health checks
        health_results = await health_checker.check_all_services()
        overall_health = health_checker.get_overall_health_status(health_results)
        logger.info("Initial health check completed", overall_status=overall_health)
        
        if overall_health == "UNHEALTHY":
            logger.critical("System failed initial health checks - some services may be unavailable")
        
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Workflow Management System")
    try:
        db_manager.close_connections()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


# Create FastAPI application
app = FastAPI(
    title=settings.api.api_title,
    description=settings.api.api_description,
    version="1.0.0",
    lifespan=lifespan,
    debug=settings.api.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=settings.api.cors_methods,
    allow_headers=settings.api.cors_headers,
)

# Add custom middleware (order matters!)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging and metrics."""
    # Log the exception
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )
    
    # Record metrics
    metrics_collector.record_system_error("http_handler", f"http_{exc.status_code}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": str(performance_tracker.response_times[-1][0] if performance_tracker.response_times else ""),
            "request_id": getattr(request.state, 'request_id', None)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(
        "Request validation error",
        errors=exc.errors(),
        path=request.url.path,
        method=request.method
    )
    
    metrics_collector.record_system_error("validation", "request_validation")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "details": exc.errors(),
            "request_id": getattr(request.state, 'request_id', None)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(
        "Unexpected error occurred",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        method=request.method
    )
    
    metrics_collector.record_system_error("application", "unexpected_error")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_results = await health_checker.check_all_services()
        overall_status = health_checker.get_overall_health_status(health_results)
        
        # Store health metrics
        await health_checker.store_health_metrics(health_results)
        
        response_data = {
            "status": overall_status.lower(),
            "timestamp": str(performance_tracker.response_times[-1][0] if performance_tracker.response_times else ""),
            "services": {
                name: {
                    "status": result.status.lower(),
                    "response_time_ms": result.response_time_ms,
                    "details": result.details
                } for name, result in health_results.items()
            },
            "performance": performance_tracker.get_performance_summary()
        }
        
        # Return appropriate HTTP status
        if overall_status == "HEALTHY":
            return response_data
        elif overall_status == "DEGRADED":
            return Response(content=str(response_data), status_code=200)  # Still operational
        else:
            return Response(content=str(response_data), status_code=503)  # Service unavailable
            
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return Response(
            content=str({"status": "error", "message": "Health check failed"}),
            status_code=503
        )


# Metrics endpoint for Prometheus
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint."""
    try:
        # Update system metrics before returning
        metrics_collector.set_active_sessions(50)  # This would come from actual session tracking
        metrics_collector.set_db_connections(10)   # This would come from connection pool
        
        return metrics_collector.get_metrics_as_text()
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        return PlainTextResponse("# Metrics generation failed\n", status_code=500)


# Basic status endpoint
@app.get("/status")
async def status():
    """Basic status endpoint for load balancer health checks."""
    return {
        "status": "operational",
        "service": "workflow-management-system",
        "version": "1.0.0",
        "environment": settings.environment
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": "Workflow Management System",
        "version": "1.0.0",
        "status": "operational",
        "environment": settings.environment,
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "workflow_engine.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level=settings.monitoring.log_level.lower(),
        access_log=True,
        server_header=False,  # Don't expose server information
        date_header=False     # Don't include date header for security
    )