"""
Real-time Workflow Execution Monitoring and Progress Tracking.

This module provides comprehensive monitoring capabilities for workflow execution,
including progress tracking, performance metrics, error detection, and alerts.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database.connection import db_manager
from ..database.models import WorkflowInstance, WorkflowStepExecution
from ..config.settings import settings


class MonitoringEventType(str, Enum):
    """Types of monitoring events."""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    PERFORMANCE_ALERT = "performance_alert"
    ERROR_DETECTED = "error_detected"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_COMPLETED = "approval_completed"


@dataclass
class MonitoringEvent:
    """Workflow monitoring event."""
    event_type: MonitoringEventType
    workflow_id: str
    instance_id: str
    step_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical
    message: str = ""


@dataclass
class WorkflowMetrics:
    """Workflow execution metrics."""
    instance_id: str
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    progress_percentage: float = 0.0
    execution_start_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    average_step_duration: float = 0.0
    current_step_duration: float = 0.0
    performance_score: float = 100.0
    bottlenecks: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class WorkflowMonitor:
    """Real-time workflow execution monitor."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_monitors: Dict[str, WorkflowMetrics] = {}
        self.event_handlers: Dict[MonitoringEventType, List[Callable]] = {
            event_type: [] for event_type in MonitoringEventType
        }
        self.alert_thresholds = {
            "step_timeout_minutes": 30,
            "workflow_timeout_hours": 24,
            "error_rate_threshold": 0.1,
            "performance_degradation_threshold": 0.3
        }
        self.monitoring_active = True
        
        # Background monitoring will be started via FastAPI startup event
        self._monitoring_task = None
    
    async def start_monitoring(self, instance_id: str, workflow_definition: Dict[str, Any]):
        """Start monitoring a workflow instance."""
        try:
            total_steps = len(workflow_definition.get("steps", []))
            
            metrics = WorkflowMetrics(
                instance_id=instance_id,
                total_steps=total_steps,
                execution_start_time=datetime.now(timezone.utc)
            )
            
            self.active_monitors[instance_id] = metrics
            
            # Emit monitoring start event
            event = MonitoringEvent(
                event_type=MonitoringEventType.WORKFLOW_STARTED,
                workflow_id=workflow_definition.get("id", "unknown"),
                instance_id=instance_id,
                message=f"Started monitoring workflow instance {instance_id}",
                data={
                    "total_steps": total_steps,
                    "workflow_name": workflow_definition.get("name")
                }
            )
            
            await self._emit_event(event)
            
            self.logger.info(f"Started monitoring workflow instance {instance_id}")
            
        except Exception as e:
            self.logger.error(f"Error starting workflow monitoring: {str(e)}")
    
    async def update_step_progress(
        self, 
        instance_id: str, 
        step_id: str, 
        status: str,
        duration: Optional[float] = None,
        error_message: Optional[str] = None
    ):
        """Update step execution progress."""
        metrics = self.active_monitors.get(instance_id)
        if not metrics:
            self.logger.warning(f"No monitoring active for instance {instance_id}")
            return
        
        try:
            # Update step counts
            if status == "completed":
                metrics.completed_steps += 1
            elif status == "failed":
                metrics.failed_steps += 1
                if error_message:
                    metrics.errors.append(f"Step {step_id}: {error_message}")
            elif status == "skipped":
                metrics.skipped_steps += 1
            
            # Update progress percentage
            total_processed = metrics.completed_steps + metrics.failed_steps + metrics.skipped_steps
            if metrics.total_steps > 0:
                metrics.progress_percentage = (total_processed / metrics.total_steps) * 100
            
            # Update timing metrics
            if duration:
                metrics.current_step_duration = duration
                
                # Calculate average step duration
                if metrics.completed_steps > 0:
                    total_duration = sum([step.get("duration", 0) for step in self._get_completed_steps(instance_id)])
                    metrics.average_step_duration = total_duration / metrics.completed_steps
                
                # Detect performance bottlenecks
                if duration > metrics.average_step_duration * 2:
                    bottleneck_msg = f"Step {step_id} took {duration:.2f}s (avg: {metrics.average_step_duration:.2f}s)"
                    if bottleneck_msg not in metrics.bottlenecks:
                        metrics.bottlenecks.append(bottleneck_msg)
            
            # Update performance score
            metrics.performance_score = self._calculate_performance_score(metrics)
            
            # Estimate completion time
            if metrics.average_step_duration > 0 and metrics.total_steps > 0:
                remaining_steps = metrics.total_steps - total_processed
                estimated_remaining_time = remaining_steps * metrics.average_step_duration
                metrics.estimated_completion_time = datetime.now(timezone.utc) + timedelta(seconds=estimated_remaining_time)
            
            # Emit progress event
            event = MonitoringEvent(
                event_type=MonitoringEventType.STEP_COMPLETED if status == "completed" else MonitoringEventType.STEP_FAILED,
                workflow_id="unknown",  # Will be populated from database
                instance_id=instance_id,
                step_id=step_id,
                message=f"Step {step_id} {status}",
                severity="error" if status == "failed" else "info",
                data={
                    "status": status,
                    "duration": duration,
                    "progress": metrics.progress_percentage,
                    "performance_score": metrics.performance_score
                }
            )
            
            await self._emit_event(event)
            
        except Exception as e:
            self.logger.error(f"Error updating step progress: {str(e)}")
    
    def get_workflow_metrics(self, instance_id: str) -> Optional[WorkflowMetrics]:
        """Get current workflow metrics."""
        return self.active_monitors.get(instance_id)
    
    def get_all_active_workflows(self) -> Dict[str, WorkflowMetrics]:
        """Get metrics for all active workflows."""
        return self.active_monitors.copy()
    
    async def stop_monitoring(self, instance_id: str, final_status: str = "completed"):
        """Stop monitoring a workflow instance."""
        metrics = self.active_monitors.get(instance_id)
        if not metrics:
            return
        
        try:
            # Calculate final metrics
            if metrics.execution_start_time:
                total_duration = (datetime.now(timezone.utc) - metrics.execution_start_time).total_seconds()
            else:
                total_duration = 0
            
            # Emit completion event
            event = MonitoringEvent(
                event_type=MonitoringEventType.WORKFLOW_COMPLETED if final_status == "completed" else MonitoringEventType.WORKFLOW_FAILED,
                workflow_id="unknown",
                instance_id=instance_id,
                message=f"Workflow {final_status} - Duration: {total_duration:.2f}s",
                severity="error" if final_status == "failed" else "info",
                data={
                    "final_status": final_status,
                    "total_duration": total_duration,
                    "completed_steps": metrics.completed_steps,
                    "failed_steps": metrics.failed_steps,
                    "performance_score": metrics.performance_score,
                    "bottlenecks": metrics.bottlenecks,
                    "errors": metrics.errors
                }
            )
            
            await self._emit_event(event)
            
            # Store final metrics to database
            await self._persist_workflow_metrics(instance_id, metrics, final_status)
            
            # Remove from active monitoring
            del self.active_monitors[instance_id]
            
            self.logger.info(f"Stopped monitoring workflow instance {instance_id}")
            
        except Exception as e:
            self.logger.error(f"Error stopping workflow monitoring: {str(e)}")
    
    def add_event_handler(self, event_type: MonitoringEventType, handler: Callable):
        """Add an event handler for specific monitoring events."""
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: MonitoringEventType, handler: Callable):
        """Remove an event handler."""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    async def _emit_event(self, event: MonitoringEvent):
        """Emit a monitoring event to all registered handlers."""
        try:
            # Call event handlers
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {str(e)}")
            
            # Log event
            self.logger.info(
                f"Monitoring event: {event.event_type.value}",
                extra={
                    "instance_id": event.instance_id,
                    "step_id": event.step_id,
                    "severity": event.severity,
                    "message": event.message,
                    "data": event.data
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error emitting monitoring event: {str(e)}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop for alerts and cleanup."""
        while self.monitoring_active:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now(timezone.utc)
                
                for instance_id, metrics in self.active_monitors.items():
                    # Check for workflow timeout
                    if metrics.execution_start_time:
                        duration = (current_time - metrics.execution_start_time).total_seconds()
                        if duration > self.alert_thresholds["workflow_timeout_hours"] * 3600:
                            await self._emit_alert(
                                instance_id,
                                "workflow_timeout",
                                f"Workflow has been running for {duration/3600:.1f} hours",
                                "warning"
                            )
                    
                    # Check performance degradation
                    if metrics.performance_score < (100 * (1 - self.alert_thresholds["performance_degradation_threshold"])):
                        await self._emit_alert(
                            instance_id,
                            "performance_degradation",
                            f"Performance score: {metrics.performance_score:.1f}%",
                            "warning"
                        )
                    
                    # Check error rate
                    total_processed = metrics.completed_steps + metrics.failed_steps
                    if total_processed > 0:
                        error_rate = metrics.failed_steps / total_processed
                        if error_rate > self.alert_thresholds["error_rate_threshold"]:
                            await self._emit_alert(
                                instance_id,
                                "high_error_rate",
                                f"Error rate: {error_rate*100:.1f}%",
                                "error"
                            )
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
    
    async def _emit_alert(self, instance_id: str, alert_type: str, message: str, severity: str):
        """Emit a monitoring alert."""
        event = MonitoringEvent(
            event_type=MonitoringEventType.PERFORMANCE_ALERT,
            workflow_id="unknown",
            instance_id=instance_id,
            message=f"{alert_type}: {message}",
            severity=severity,
            data={"alert_type": alert_type}
        )
        
        await self._emit_event(event)
    
    def _calculate_performance_score(self, metrics: WorkflowMetrics) -> float:
        """Calculate workflow performance score (0-100)."""
        try:
            score = 100.0
            
            # Deduct for errors
            total_processed = metrics.completed_steps + metrics.failed_steps + metrics.skipped_steps
            if total_processed > 0:
                error_rate = metrics.failed_steps / total_processed
                score -= error_rate * 50  # Max 50 point deduction for errors
            
            # Deduct for bottlenecks
            bottleneck_penalty = min(len(metrics.bottlenecks) * 5, 30)  # Max 30 point deduction
            score -= bottleneck_penalty
            
            # Deduct for long execution times
            if metrics.execution_start_time:
                duration_hours = (datetime.now(timezone.utc) - metrics.execution_start_time).total_seconds() / 3600
                if duration_hours > 2:  # If workflow takes more than 2 hours
                    time_penalty = min((duration_hours - 2) * 5, 20)  # Max 20 point deduction
                    score -= time_penalty
            
            return max(0.0, min(100.0, score))
            
        except Exception:
            return 50.0  # Default score if calculation fails
    
    def _get_completed_steps(self, instance_id: str) -> List[Dict[str, Any]]:
        """Get completed steps for an instance from database."""
        try:
            with db_manager.get_session() as session:
                steps = session.query(WorkflowStepExecution).filter(
                    WorkflowStepExecution.workflow_instance_id == instance_id,
                    WorkflowStepExecution.status == "completed"
                ).all()
                
                return [
                    {
                        "step_id": step.step_id,
                        "duration": step.duration_seconds or 0
                    }
                    for step in steps
                ]
        except Exception:
            return []
    
    async def _persist_workflow_metrics(
        self, 
        instance_id: str, 
        metrics: WorkflowMetrics, 
        final_status: str
    ):
        """Persist final workflow metrics to database."""
        try:
            # This would typically store metrics in a dedicated metrics table
            # For now, we'll log the metrics
            self.logger.info(
                f"Final workflow metrics for {instance_id}",
                extra={
                    "instance_id": instance_id,
                    "final_status": final_status,
                    "metrics": {
                        "total_steps": metrics.total_steps,
                        "completed_steps": metrics.completed_steps,
                        "failed_steps": metrics.failed_steps,
                        "progress_percentage": metrics.progress_percentage,
                        "performance_score": metrics.performance_score,
                        "bottlenecks": metrics.bottlenecks,
                        "errors": metrics.errors
                    }
                }
            )
        except Exception as e:
            self.logger.error(f"Error persisting workflow metrics: {str(e)}")
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        try:
            active_workflows = len(self.active_monitors)
            total_steps_running = sum(m.total_steps for m in self.active_monitors.values())
            total_completed_steps = sum(m.completed_steps for m in self.active_monitors.values())
            total_failed_steps = sum(m.failed_steps for m in self.active_monitors.values())
            
            avg_performance = sum(m.performance_score for m in self.active_monitors.values()) / active_workflows if active_workflows > 0 else 100
            
            return {
                "summary": {
                    "active_workflows": active_workflows,
                    "total_steps_running": total_steps_running,
                    "total_completed_steps": total_completed_steps,
                    "total_failed_steps": total_failed_steps,
                    "average_performance_score": round(avg_performance, 1)
                },
                "active_workflows": {
                    instance_id: {
                        "progress_percentage": metrics.progress_percentage,
                        "performance_score": metrics.performance_score,
                        "completed_steps": metrics.completed_steps,
                        "failed_steps": metrics.failed_steps,
                        "bottlenecks": len(metrics.bottlenecks),
                        "errors": len(metrics.errors),
                        "estimated_completion": metrics.estimated_completion_time.isoformat() if metrics.estimated_completion_time else None
                    }
                    for instance_id, metrics in self.active_monitors.items()
                },
                "system_health": {
                    "monitoring_active": self.monitoring_active,
                    "alert_thresholds": self.alert_thresholds,
                    "total_event_handlers": sum(len(handlers) for handlers in self.event_handlers.values())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {str(e)}")
            return {"error": "Failed to generate dashboard data"}
    
    def shutdown(self):
        """Shutdown the monitoring system."""
        self.monitoring_active = False
        self.logger.info("Workflow monitoring system shut down")


# Global workflow monitor instance
workflow_monitor = WorkflowMonitor()


# Built-in event handlers
async def log_workflow_events(event: MonitoringEvent):
    """Built-in event handler for logging workflow events."""
    logger = logging.getLogger("workflow_events")
    
    log_level = {
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
        "critical": logger.critical
    }.get(event.severity, logger.info)
    
    log_level(
        f"Workflow Event: {event.event_type.value}",
        extra={
            "instance_id": event.instance_id,
            "step_id": event.step_id,
            "message": event.message,
            "data": event.data,
            "timestamp": event.timestamp.isoformat()
        }
    )


async def performance_alert_handler(event: MonitoringEvent):
    """Built-in handler for performance alerts."""
    if event.event_type == MonitoringEventType.PERFORMANCE_ALERT:
        logger = logging.getLogger("performance_alerts")
        logger.warning(
            f"Performance Alert: {event.message}",
            extra={
                "instance_id": event.instance_id,
                "alert_data": event.data
            }
        )


# Register built-in event handlers
workflow_monitor.add_event_handler(MonitoringEventType.WORKFLOW_STARTED, log_workflow_events)
workflow_monitor.add_event_handler(MonitoringEventType.WORKFLOW_COMPLETED, log_workflow_events)
workflow_monitor.add_event_handler(MonitoringEventType.WORKFLOW_FAILED, log_workflow_events)
workflow_monitor.add_event_handler(MonitoringEventType.STEP_COMPLETED, log_workflow_events)
workflow_monitor.add_event_handler(MonitoringEventType.STEP_FAILED, log_workflow_events)
workflow_monitor.add_event_handler(MonitoringEventType.PERFORMANCE_ALERT, performance_alert_handler)