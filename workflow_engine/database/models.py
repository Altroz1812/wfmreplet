from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

Base = declarative_base()


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Security fields
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    failed_login_attempts = Column(Integer, default=0)
    last_login = Column(DateTime(timezone=True))
    password_changed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # MFA fields
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(255))  # Encrypted TOTP secret
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Relationships
    roles = relationship("UserRole", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    __table_args__ = (
        Index("idx_user_email_active", "email", "is_active"),
        Index("idx_user_username_active", "username", "is_active"),
    )


class Role(Base):
    """Role model for RBAC."""
    
    __tablename__ = "roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    permissions = Column(JSONB, default=list)  # List of permission strings
    is_active = Column(Boolean, default=True)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user_roles = relationship("UserRole", back_populates="role")


class UserRole(Base):
    """Many-to-many relationship between users and roles."""
    
    __tablename__ = "user_roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role_id = Column(UUID(as_uuid=True), ForeignKey("roles.id"), nullable=False)
    assigned_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    assigned_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Relationships
    user = relationship("User", back_populates="roles")
    role = relationship("Role", back_populates="user_roles")
    
    __table_args__ = (
        Index("idx_user_role", "user_id", "role_id", unique=True),
    )


class UserSession(Base):
    """User session tracking for security monitoring."""
    
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    refresh_token = Column(String(255), unique=True, nullable=False, index=True)
    
    # Session details
    ip_address = Column(String(45))  # Support IPv6
    user_agent = Column(Text)
    device_fingerprint = Column(String(255))
    
    # Session lifecycle
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_activity = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    __table_args__ = (
        Index("idx_session_user_active", "user_id", "is_active"),
        Index("idx_session_expires", "expires_at"),
    )


class AuditLog(Base):
    """Comprehensive audit logging for security and compliance."""
    
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Who performed the action
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    session_id = Column(UUID(as_uuid=True), ForeignKey("user_sessions.id"))
    
    # What action was performed
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False, index=True)
    resource_id = Column(String(255), index=True)
    
    # Request details
    endpoint = Column(String(255))
    http_method = Column(String(10))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # Action details
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    audit_metadata = Column(JSONB)
    
    # Status
    status = Column(String(20), nullable=False, index=True)  # SUCCESS, FAILURE, ERROR
    error_message = Column(Text)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    __table_args__ = (
        Index("idx_audit_user_timestamp", "user_id", "timestamp"),
        Index("idx_audit_action_timestamp", "action", "timestamp"),
        Index("idx_audit_resource_timestamp", "resource_type", "resource_id", "timestamp"),
    )


class SystemHealth(Base):
    """System health monitoring and metrics."""
    
    __tablename__ = "system_health"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Health check details
    service_name = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)  # HEALTHY, DEGRADED, UNHEALTHY
    response_time_ms = Column(Integer)
    
    # Resource metrics
    cpu_usage_percent = Column(Integer)
    memory_usage_percent = Column(Integer)
    disk_usage_percent = Column(Integer)
    
    # Database metrics
    active_connections = Column(Integer)
    slow_queries_count = Column(Integer)
    
    # API metrics
    requests_per_minute = Column(Integer)
    error_rate_percent = Column(Integer)
    
    # Additional metrics
    health_metrics = Column(JSONB)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    
    __table_args__ = (
        Index("idx_health_service_timestamp", "service_name", "timestamp"),
        Index("idx_health_status_timestamp", "status", "timestamp"),
    )


class Configuration(Base):
    """System configuration management."""
    
    __tablename__ = "configurations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Configuration details
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)  # JSON string or plain text
    data_type = Column(String(20), nullable=False)  # string, integer, boolean, json
    category = Column(String(50), nullable=False, index=True)
    description = Column(Text)
    
    # Security
    is_sensitive = Column(Boolean, default=False)  # If true, value should be encrypted
    is_readonly = Column(Boolean, default=False)
    
    # Versioning
    version = Column(Integer, default=1)
    previous_value = Column(Text)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    updated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    __table_args__ = (
        Index("idx_config_category_key", "category", "key"),
    )


class APIRateLimit(Base):
    """API rate limiting tracking."""
    
    __tablename__ = "api_rate_limits"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Rate limit details
    identifier = Column(String(255), nullable=False, index=True)  # user_id, ip_address, api_key
    identifier_type = Column(String(20), nullable=False, index=True)  # user, ip, api_key
    endpoint = Column(String(255), nullable=False, index=True)
    
    # Counters
    request_count = Column(Integer, default=0)
    window_start = Column(DateTime(timezone=True), nullable=False)
    window_end = Column(DateTime(timezone=True), nullable=False)
    
    # Status
    is_blocked = Column(Boolean, default=False)
    blocked_until = Column(DateTime(timezone=True))
    
    __table_args__ = (
        Index("idx_rate_limit_identifier_endpoint", "identifier", "endpoint"),
        Index("idx_rate_limit_window", "window_end"),
    )