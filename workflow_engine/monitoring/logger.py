import sys
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pathlib import Path

import structlog
from pythonjsonlogger import jsonlogger

from ..config.settings import settings


class StructuredLogger:
    """Structured logging with JSON output and contextual information."""
    
    def __init__(self):
        self._configure_logging()
        self.logger = structlog.get_logger()
    
    def _configure_logging(self):
        """Configure structured logging with appropriate processors and formatters."""
        
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, settings.monitoring.log_level.upper()),
        )
        
        # Shared processors for all loggers
        shared_processors = [
            # Add log level
            structlog.stdlib.add_log_level,
            
            # Add timestamp
            self._add_timestamp,
            
            # Add service information
            self._add_service_info,
            
            # Filter out internal keys
            structlog.stdlib.filter_by_level,
        ]
        
        if settings.monitoring.log_format == "json":
            # JSON formatting for production
            shared_processors.extend([
                structlog.processors.JSONRenderer()
            ])
        else:
            # Human-readable formatting for development
            shared_processors.extend([
                structlog.dev.ConsoleRenderer(colors=True)
            ])
        
        # Configure structlog
        structlog.configure(
            processors=shared_processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )
        
        # Configure file logging if specified
        if settings.monitoring.log_file_path:
            self._configure_file_logging()
    
    def _add_timestamp(self, logger, method_name, event_dict):
        """Add ISO timestamp to log entries."""
        event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        return event_dict
    
    def _add_service_info(self, logger, method_name, event_dict):
        """Add service information to log entries."""
        event_dict.update({
            "service": "workflow-management-system",
            "version": "1.0.0",
            "environment": settings.environment
        })
        return event_dict
    
    def _configure_file_logging(self):
        """Configure file-based logging."""
        log_file = Path(settings.monitoring.log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler with JSON formatting
        file_handler = logging.FileHandler(log_file)
        
        if settings.monitoring.log_format == "json":
            json_formatter = jsonlogger.JsonFormatter(
                fmt="%(timestamp)s %(level)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S"
            )
            file_handler.setFormatter(json_formatter)
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self.logger.critical(message, **kwargs)
    
    def with_context(self, **kwargs) -> structlog.BoundLogger:
        """Create a logger with additional context."""
        return self.logger.bind(**kwargs)
    
    def log_request(self, method: str, path: str, status_code: int, 
                   processing_time: float, user_id: Optional[str] = None, **kwargs):
        """Log HTTP request with standard fields."""
        self.info(
            "HTTP request processed",
            http_method=method,
            http_path=path,
            http_status_code=status_code,
            processing_time_ms=round(processing_time * 1000, 2),
            user_id=user_id,
            **kwargs
        )
    
    def log_database_operation(self, operation: str, table: str, 
                              execution_time: float, affected_rows: Optional[int] = None, **kwargs):
        """Log database operation with timing."""
        self.info(
            "Database operation",
            db_operation=operation,
            db_table=table,
            execution_time_ms=round(execution_time * 1000, 2),
            affected_rows=affected_rows,
            **kwargs
        )
    
    def log_security_event(self, event_type: str, user_id: Optional[str] = None, 
                          ip_address: Optional[str] = None, success: bool = True, **kwargs):
        """Log security-related events."""
        level = self.info if success else self.warning
        level(
            "Security event",
            security_event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            success=success,
            **kwargs
        )
    
    def log_business_event(self, event_type: str, entity_type: str, entity_id: str,
                          user_id: Optional[str] = None, **kwargs):
        """Log business logic events."""
        self.info(
            "Business event",
            business_event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            **kwargs
        )
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "ms", **kwargs):
        """Log performance metrics."""
        self.info(
            "Performance metric",
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            **kwargs
        )
    
    def log_error_with_traceback(self, message: str, error: Exception, **kwargs):
        """Log error with full traceback information."""
        import traceback
        
        self.error(
            message,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            **kwargs
        )


class SecurityLogger:
    """Specialized logger for security events and audit trails."""
    
    def __init__(self, base_logger: StructuredLogger):
        self.logger = base_logger.with_context(component="security")
    
    def log_login_attempt(self, username: str, ip_address: str, success: bool, 
                         reason: Optional[str] = None):
        """Log user login attempt."""
        self.logger.info(
            "User login attempt",
            username=username,
            ip_address=ip_address,
            success=success,
            failure_reason=reason if not success else None,
            security_event="login_attempt"
        )
    
    def log_logout(self, user_id: str, session_duration_minutes: float):
        """Log user logout."""
        self.logger.info(
            "User logout",
            user_id=user_id,
            session_duration_minutes=round(session_duration_minutes, 2),
            security_event="logout"
        )
    
    def log_permission_denied(self, user_id: str, resource: str, required_permission: str,
                             ip_address: Optional[str] = None):
        """Log permission denied events."""
        self.logger.warning(
            "Permission denied",
            user_id=user_id,
            resource=resource,
            required_permission=required_permission,
            ip_address=ip_address,
            security_event="permission_denied"
        )
    
    def log_suspicious_activity(self, activity_type: str, user_id: Optional[str] = None,
                               ip_address: Optional[str] = None, details: Optional[Dict] = None):
        """Log suspicious activity."""
        self.logger.warning(
            "Suspicious activity detected",
            activity_type=activity_type,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            security_event="suspicious_activity"
        )
    
    def log_rate_limit_exceeded(self, identifier: str, identifier_type: str,
                               endpoint: str, current_count: int, limit: int):
        """Log rate limit exceeded events."""
        self.logger.warning(
            "Rate limit exceeded",
            identifier=identifier,
            identifier_type=identifier_type,
            endpoint=endpoint,
            current_count=current_count,
            limit=limit,
            security_event="rate_limit_exceeded"
        )


class ApplicationLogger:
    """Application-specific logger for business logic events."""
    
    def __init__(self, base_logger: StructuredLogger):
        self.logger = base_logger.with_context(component="application")
    
    def log_workflow_event(self, workflow_id: str, event_type: str, user_id: str,
                          stage: Optional[str] = None, details: Optional[Dict] = None):
        """Log workflow-related events."""
        self.logger.info(
            "Workflow event",
            workflow_id=workflow_id,
            event_type=event_type,
            user_id=user_id,
            current_stage=stage,
            details=details,
            business_event="workflow"
        )
    
    def log_application_status_change(self, application_id: str, old_status: str,
                                    new_status: str, user_id: str, reason: Optional[str] = None):
        """Log application status changes."""
        self.logger.info(
            "Application status changed",
            application_id=application_id,
            old_status=old_status,
            new_status=new_status,
            user_id=user_id,
            change_reason=reason,
            business_event="status_change"
        )
    
    def log_document_event(self, document_id: str, event_type: str, user_id: str,
                          application_id: Optional[str] = None):
        """Log document-related events."""
        self.logger.info(
            "Document event",
            document_id=document_id,
            event_type=event_type,
            user_id=user_id,
            application_id=application_id,
            business_event="document"
        )
    
    def log_approval_event(self, application_id: str, approver_id: str, 
                          decision: str, comments: Optional[str] = None):
        """Log approval decisions."""
        self.logger.info(
            "Approval decision",
            application_id=application_id,
            approver_id=approver_id,
            decision=decision,
            comments=comments,
            business_event="approval"
        )


# Global logger instances
logger = StructuredLogger()
security_logger = SecurityLogger(logger)
app_logger = ApplicationLogger(logger)