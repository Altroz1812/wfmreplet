import time
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque
from threading import Lock
from datetime import datetime, timedelta, timezone

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.exposition import MetricsHandler

from ..config.settings import settings


class MetricsCollector:
    """Comprehensive metrics collection for monitoring and observability."""
    
    def __init__(self):
        # Create separate registry for workflow management metrics
        self.registry = CollectorRegistry()
        
        # Request metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Authentication metrics
        self.auth_attempts = Counter(
            'auth_attempts_total',
            'Total authentication attempts',
            ['type', 'status'],
            registry=self.registry
        )
        
        self.active_sessions = Gauge(
            'active_user_sessions',
            'Number of active user sessions',
            registry=self.registry
        )
        
        # Database metrics
        self.db_queries = Counter(
            'database_queries_total',
            'Total database queries',
            ['operation', 'table'],
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration in seconds',
            ['operation', 'table'],
            registry=self.registry
        )
        
        self.db_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        # Application-specific metrics
        self.workflow_operations = Counter(
            'workflow_operations_total',
            'Total workflow operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.application_status_changes = Counter(
            'application_status_changes_total',
            'Total application status changes',
            ['from_status', 'to_status'],
            registry=self.registry
        )
        
        self.document_operations = Counter(
            'document_operations_total',
            'Total document operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        # System metrics
        self.system_errors = Counter(
            'system_errors_total',
            'Total system errors',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        self.rate_limit_violations = Counter(
            'rate_limit_violations_total',
            'Total rate limit violations',
            ['identifier_type', 'endpoint'],
            registry=self.registry
        )
        
        # Performance metrics
        self.cache_operations = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        # Custom business metrics
        self.business_metrics = Gauge(
            'business_metric_value',
            'Custom business metrics',
            ['metric_name', 'metric_type'],
            registry=self.registry
        )
        
        # Internal tracking for custom metrics
        self._custom_counters = defaultdict(int)
        self._custom_timers = {}
        self._lock = Lock()
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.request_count.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_auth_attempt(self, auth_type: str, success: bool):
        """Record authentication attempt."""
        status = "success" if success else "failure"
        self.auth_attempts.labels(type=auth_type, status=status).inc()
    
    def set_active_sessions(self, count: int):
        """Set number of active user sessions."""
        self.active_sessions.set(count)
    
    def record_db_query(self, operation: str, table: str, duration: float):
        """Record database query metrics."""
        self.db_queries.labels(operation=operation, table=table).inc()
        self.db_query_duration.labels(operation=operation, table=table).observe(duration)
    
    def set_db_connections(self, count: int):
        """Set number of active database connections."""
        self.db_connections.set(count)
    
    def record_workflow_operation(self, operation: str, success: bool):
        """Record workflow operation metrics."""
        status = "success" if success else "failure"
        self.workflow_operations.labels(operation=operation, status=status).inc()
    
    def record_application_status_change(self, from_status: str, to_status: str):
        """Record application status change."""
        self.application_status_changes.labels(from_status=from_status, to_status=to_status).inc()
    
    def record_document_operation(self, operation: str, success: bool):
        """Record document operation metrics."""
        status = "success" if success else "failure"
        self.document_operations.labels(operation=operation, status=status).inc()
    
    def record_system_error(self, component: str, error_type: str):
        """Record system error metrics."""
        self.system_errors.labels(component=component, error_type=error_type).inc()
    
    def record_rate_limit_violation(self, identifier_type: str, endpoint: str):
        """Record rate limit violation."""
        self.rate_limit_violations.labels(identifier_type=identifier_type, endpoint=endpoint).inc()
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation metrics."""
        result = "hit" if hit else "miss"
        self.cache_operations.labels(operation=operation, result=result).inc()
    
    def set_business_metric(self, metric_name: str, value: float, metric_type: str = "gauge"):
        """Set custom business metric value."""
        self.business_metrics.labels(metric_name=metric_name, metric_type=metric_type).set(value)
    
    def increment_custom_counter(self, counter_name: str, value: int = 1):
        """Increment a custom counter."""
        with self._lock:
            self._custom_counters[counter_name] += value
    
    def get_custom_counter(self, counter_name: str) -> int:
        """Get custom counter value."""
        with self._lock:
            return self._custom_counters[counter_name]
    
    def start_timer(self, timer_name: str) -> str:
        """Start a custom timer and return timer ID."""
        timer_id = f"{timer_name}_{int(time.time() * 1000000)}"
        self._custom_timers[timer_id] = time.time()
        return timer_id
    
    def stop_timer(self, timer_id: str) -> float:
        """Stop a custom timer and return elapsed time in seconds."""
        if timer_id in self._custom_timers:
            start_time = self._custom_timers.pop(timer_id)
            return time.time() - start_time
        return 0.0
    
    def get_metrics_as_text(self) -> str:
        """Get all metrics in Prometheus text format."""
        return generate_latest(self.registry).decode('utf-8')


class PerformanceTracker:
    """Track and analyze application performance patterns."""
    
    def __init__(self, window_size_minutes: int = 60):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.response_times = deque()
        self.error_counts = defaultdict(int)
        self.throughput_data = deque()
        self.lock = Lock()
    
    def record_response_time(self, endpoint: str, response_time: float):
        """Record response time for an endpoint."""
        with self.lock:
            now = datetime.now(timezone.utc)
            self.response_times.append((now, endpoint, response_time))
            self._cleanup_old_data()
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        with self.lock:
            self.error_counts[error_type] += 1
    
    def record_throughput(self, requests_count: int):
        """Record throughput data point."""
        with self.lock:
            now = datetime.now(timezone.utc)
            self.throughput_data.append((now, requests_count))
            self._cleanup_old_data()
    
    def _cleanup_old_data(self):
        """Remove data points older than the window size."""
        cutoff_time = datetime.now(timezone.utc) - self.window_size
        
        # Clean response times
        while self.response_times and self.response_times[0][0] < cutoff_time:
            self.response_times.popleft()
        
        # Clean throughput data
        while self.throughput_data and self.throughput_data[0][0] < cutoff_time:
            self.throughput_data.popleft()
    
    def get_average_response_time(self, endpoint: Optional[str] = None) -> float:
        """Get average response time for an endpoint or all endpoints."""
        with self.lock:
            if not self.response_times:
                return 0.0
            
            if endpoint:
                filtered_times = [rt for _, ep, rt in self.response_times if ep == endpoint]
                return sum(filtered_times) / len(filtered_times) if filtered_times else 0.0
            else:
                return sum(rt for _, _, rt in self.response_times) / len(self.response_times)
    
    def get_percentile_response_time(self, percentile: float, endpoint: Optional[str] = None) -> float:
        """Get percentile response time."""
        with self.lock:
            if not self.response_times:
                return 0.0
            
            if endpoint:
                times = [rt for _, ep, rt in self.response_times if ep == endpoint]
            else:
                times = [rt for _, _, rt in self.response_times]
            
            if not times:
                return 0.0
            
            times.sort()
            index = int(len(times) * percentile / 100)
            return times[index] if index < len(times) else times[-1]
    
    def get_throughput(self) -> float:
        """Get current throughput (requests per minute)."""
        with self.lock:
            if not self.throughput_data:
                return 0.0
            
            total_requests = sum(count for _, count in self.throughput_data)
            if len(self.throughput_data) == 0:
                return 0.0
            
            # Calculate requests per minute
            time_span_minutes = (self.throughput_data[-1][0] - self.throughput_data[0][0]).total_seconds() / 60
            return total_requests / max(time_span_minutes, 1)
    
    def get_error_rate(self) -> float:
        """Get current error rate percentage."""
        with self.lock:
            total_errors = sum(self.error_counts.values())
            total_requests = len(self.response_times)
            
            if total_requests == 0:
                return 0.0
            
            return (total_errors / total_requests) * 100
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "average_response_time_ms": round(self.get_average_response_time() * 1000, 2),
            "p95_response_time_ms": round(self.get_percentile_response_time(95) * 1000, 2),
            "p99_response_time_ms": round(self.get_percentile_response_time(99) * 1000, 2),
            "throughput_rpm": round(self.get_throughput(), 2),
            "error_rate_percent": round(self.get_error_rate(), 2),
            "total_requests": len(self.response_times),
            "window_size_minutes": self.window_size.total_seconds() / 60
        }


# Global metrics instances
metrics_collector = MetricsCollector()
performance_tracker = PerformanceTracker()