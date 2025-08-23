import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Union, Optional

from jose import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
from cryptography.fernet import Fernet
from pydantic import BaseModel

from ..config.settings import settings


class TokenData(BaseModel):
    """Token data model."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    roles: Optional[list[str]] = None
    permissions: Optional[list[str]] = None


class SecurityManager:
    """Central security management class."""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.encryption_key = self._get_or_generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def _get_or_generate_encryption_key(self) -> bytes:
        """Get existing encryption key or generate new one."""
        key_file = "encryption_key.key"
        
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            return key
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> tuple[bool, list[str]]:
        """Validate password meets security requirements."""
        errors = []
        
        if len(password) < settings.security.password_min_length:
            errors.append(f"Password must be at least {settings.security.password_min_length} characters long")
        
        if settings.security.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if settings.security.password_require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if settings.security.password_require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if settings.security.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=settings.security.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.security.secret_key, 
            algorithm=settings.security.algorithm
        )
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.security.refresh_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.security.secret_key,
            algorithm=settings.security.algorithm
        )
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[TokenData]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                settings.security.secret_key,
                algorithms=[settings.security.algorithm]
            )
            
            # Verify token type
            if payload.get("type") != token_type:
                return None
            
            # Extract token data
            user_id: str = payload.get("sub")
            username: str = payload.get("username")
            roles: list[str] = payload.get("roles", [])
            permissions: list[str] = payload.get("permissions", [])
            
            if user_id is None:
                return None
            
            return TokenData(
                user_id=user_id,
                username=username,
                roles=roles,
                permissions=permissions
            )
        except jwt.JWTError:
            return None
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def generate_otp(self, length: int = 6) -> str:
        """Generate numeric OTP."""
        return ''.join([str(secrets.randbelow(10)) for _ in range(length)])


# Global security manager instance
security_manager = SecurityManager()


class RBACManager:
    """Role-Based Access Control Manager."""
    
    def __init__(self):
        self.role_permissions = {
            "admin": [
                "workflow.create", "workflow.read", "workflow.update", "workflow.delete",
                "user.create", "user.read", "user.update", "user.delete",
                "system.monitor", "system.configure"
            ],
            "workflow_manager": [
                "workflow.create", "workflow.read", "workflow.update",
                "application.read", "application.update", "application.assign"
            ],
            "underwriter": [
                "application.read", "application.update", "application.approve",
                "application.reject", "document.read", "document.upload"
            ],
            "operations": [
                "application.read", "application.update", "application.process",
                "document.read", "document.upload", "document.verify"
            ],
            "credit_team": [
                "application.read", "application.assess", "credit.analyze",
                "document.read", "report.generate"
            ],
            "legal_team": [
                "application.read", "legal.review", "document.read",
                "legal.approve", "legal.reject"
            ],
            "technical_team": [
                "application.read", "technical.review", "document.read",
                "technical.approve", "technical.reject"
            ],
            "viewer": [
                "application.read", "document.read", "report.read"
            ]
        }
    
    def get_role_permissions(self, role: str) -> list[str]:
        """Get permissions for a specific role."""
        return self.role_permissions.get(role, [])
    
    def get_user_permissions(self, roles: list[str]) -> list[str]:
        """Get all permissions for a user based on their roles."""
        permissions = set()
        for role in roles:
            permissions.update(self.get_role_permissions(role))
        return list(permissions)
    
    def has_permission(self, user_roles: list[str], required_permission: str) -> bool:
        """Check if user has required permission."""
        user_permissions = self.get_user_permissions(user_roles)
        return required_permission in user_permissions
    
    def has_any_permission(self, user_roles: list[str], required_permissions: list[str]) -> bool:
        """Check if user has any of the required permissions."""
        user_permissions = self.get_user_permissions(user_roles)
        return any(perm in user_permissions for perm in required_permissions)
    
    def has_all_permissions(self, user_roles: list[str], required_permissions: list[str]) -> bool:
        """Check if user has all required permissions."""
        user_permissions = self.get_user_permissions(user_roles)
        return all(perm in user_permissions for perm in required_permissions)


# Global RBAC manager instance
rbac_manager = RBACManager()