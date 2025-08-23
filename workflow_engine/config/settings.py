import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    # JWT Settings
    secret_key: str = Field(default="workflow_secret_key_change_this_in_production", env="JWT_SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_minutes: int = 60 * 24 * 7  # 7 days
    
    # Password Settings
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    
    # Session Settings
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 5
    
    # Rate Limiting
    rate_limit_per_minute: int = 100
    auth_rate_limit_per_minute: int = 10


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # PostgreSQL Settings (use Replit's database)
    database_url: str = Field(default="", env="DATABASE_URL")
    postgres_host: str = Field("localhost", env="PGHOST")
    postgres_port: int = Field(5432, env="PGPORT")
    postgres_db: str = Field("main", env="PGDATABASE")
    postgres_user: str = Field("main", env="PGUSER")
    postgres_password: str = Field("password", env="PGPASSWORD")
    
    # Redis Settings
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_db: int = Field(0, env="REDIS_DB")
    
    # Connection Pool Settings
    db_pool_size: int = 20
    db_max_overflow: int = 30
    db_pool_timeout: int = 30
    
    @property
    def database_url_computed(self) -> str:
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    # Logging Settings
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = "json"  # json or text
    log_file_path: Optional[str] = Field(None, env="LOG_FILE_PATH")
    
    # Metrics Settings
    metrics_enabled: bool = True
    metrics_port: int = 8000
    
    # Health Check Settings
    health_check_interval: int = 30  # seconds
    database_health_timeout: int = 5
    redis_health_timeout: int = 3
    
    # Alert Settings
    alert_email_enabled: bool = False
    alert_email_smtp_host: Optional[str] = Field(None, env="SMTP_HOST")
    alert_email_smtp_port: int = Field(587, env="SMTP_PORT")
    alert_email_username: Optional[str] = Field(None, env="SMTP_USERNAME")
    alert_email_password: Optional[str] = Field(None, env="SMTP_PASSWORD")


class APISettings(BaseSettings):
    """API configuration settings."""
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = Field(False, env="DEBUG")
    
    # CORS Settings
    cors_origins: list[str] = ["*"]
    cors_methods: list[str] = ["GET", "POST", "PUT", "DELETE", "PATCH"]
    cors_headers: list[str] = ["*"]
    
    # API Versioning
    api_version: str = "v1"
    api_title: str = "Workflow Management System"
    api_description: str = "Enterprise Workflow Management System API"
    
    # Request Settings
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 60


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field("development", env="ENVIRONMENT")
    
    # Sub-settings
    security: SecuritySettings = SecuritySettings()
    database: DatabaseSettings = DatabaseSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    api: APISettings = APISettings()
    
    # Application Settings
    app_name: str = "Workflow Management System"
    app_version: str = "1.0.0"
    
    model_config = {"env_file": ".env", "case_sensitive": False}


# Global settings instance
settings = Settings()