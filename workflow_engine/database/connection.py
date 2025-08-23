import redis
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from contextlib import contextmanager
from typing import Generator

from ..config.settings import settings


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self):
        self._engine: Engine = None
        self._session_factory: sessionmaker = None
        self._redis_client: redis.Redis = None
    
    def initialize(self):
        """Initialize database connections."""
        self._create_engine()
        self._create_session_factory()
        self._create_redis_client()
    
    def _create_engine(self):
        """Create SQLAlchemy engine with connection pooling."""
        self._engine = create_engine(
            settings.database.database_url_computed,
            poolclass=pool.QueuePool,
            pool_size=settings.database.db_pool_size,
            max_overflow=settings.database.db_max_overflow,
            pool_timeout=settings.database.db_pool_timeout,
            pool_recycle=3600,  # Recycle connections every hour
            pool_pre_ping=True,  # Validate connections before use
            echo=settings.api.debug,  # SQL logging in debug mode
        )
    
    def _create_session_factory(self):
        """Create session factory."""
        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )
    
    def _create_redis_client(self):
        """Create Redis client for caching and session storage."""
        try:
            self._redis_client = redis.Redis.from_url(
                settings.database.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            # Test the connection
            self._redis_client.ping()
        except Exception as e:
            print(f"Redis not available, using in-memory fallback: {e}")
            self._redis_client = None
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_engine(self) -> Engine:
        """Get SQLAlchemy engine."""
        if self._engine is None:
            raise RuntimeError("Database not initialized")
        return self._engine
    
    def get_redis(self) -> redis.Redis:
        """Get Redis client."""
        return self._redis_client
    
    def health_check(self) -> dict:
        """Perform database health check."""
        health_status = {
            "postgresql": {"status": "unknown", "response_time_ms": None},
            "redis": {"status": "unknown", "response_time_ms": None}
        }
        
        # PostgreSQL health check
        try:
            import time
            start_time = time.time()
            
            with self.get_session() as session:
                session.execute("SELECT 1")
            
            response_time = (time.time() - start_time) * 1000
            health_status["postgresql"] = {
                "status": "healthy",
                "response_time_ms": round(response_time, 2)
            }
        except Exception as e:
            health_status["postgresql"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Redis health check
        try:
            import time
            start_time = time.time()
            
            self._redis_client.ping()
            
            response_time = (time.time() - start_time) * 1000
            health_status["redis"] = {
                "status": "healthy",
                "response_time_ms": round(response_time, 2)
            }
        except Exception as e:
            health_status["redis"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        return health_status
    
    def close_connections(self):
        """Close all database connections."""
        if self._engine:
            self._engine.dispose()
        if self._redis_client:
            self._redis_client.close()


# Global database manager instance
db_manager = DatabaseManager()