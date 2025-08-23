-- Initial database schema for Workflow Management System
-- This migration creates the core tables for Phase 0

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    
    -- Security fields
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    failed_login_attempts INTEGER DEFAULT 0,
    last_login TIMESTAMP WITH TIME ZONE,
    password_changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- MFA fields
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id)
);

-- Create indexes for users table
CREATE INDEX idx_user_email_active ON users(email, is_active);
CREATE INDEX idx_user_username_active ON users(username, is_active);

-- Roles table
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB DEFAULT '[]'::JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User roles junction table
CREATE TABLE user_roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    assigned_by UUID REFERENCES users(id),
    
    UNIQUE(user_id, role_id)
);

CREATE INDEX idx_user_role ON user_roles(user_id, role_id);

-- User sessions table
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE NOT NULL,
    
    -- Session details
    ip_address VARCHAR(45),
    user_agent TEXT,
    device_fingerprint VARCHAR(255),
    
    -- Session lifecycle
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_session_user_active ON user_sessions(user_id, is_active);
CREATE INDEX idx_session_expires ON user_sessions(expires_at);
CREATE INDEX idx_session_token ON user_sessions(session_token);
CREATE INDEX idx_refresh_token ON user_sessions(refresh_token);

-- Audit logs table
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Who performed the action
    user_id UUID REFERENCES users(id),
    session_id UUID REFERENCES user_sessions(id),
    
    -- What action was performed
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255),
    
    -- Request details
    endpoint VARCHAR(255),
    http_method VARCHAR(10),
    ip_address VARCHAR(45),
    user_agent TEXT,
    
    -- Action details
    old_values JSONB,
    new_values JSONB,
    metadata JSONB,
    
    -- Status
    status VARCHAR(20) NOT NULL, -- SUCCESS, FAILURE, ERROR
    error_message TEXT,
    
    -- Timestamp
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_user_timestamp ON audit_logs(user_id, timestamp);
CREATE INDEX idx_audit_action_timestamp ON audit_logs(action, timestamp);
CREATE INDEX idx_audit_resource_timestamp ON audit_logs(resource_type, resource_id, timestamp);
CREATE INDEX idx_audit_status ON audit_logs(status);

-- System health table
CREATE TABLE system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Health check details
    service_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL, -- HEALTHY, DEGRADED, UNHEALTHY
    response_time_ms INTEGER,
    
    -- Resource metrics
    cpu_usage_percent INTEGER,
    memory_usage_percent INTEGER,
    disk_usage_percent INTEGER,
    
    -- Database metrics
    active_connections INTEGER,
    slow_queries_count INTEGER,
    
    -- API metrics
    requests_per_minute INTEGER,
    error_rate_percent INTEGER,
    
    -- Additional metrics
    metrics JSONB,
    
    -- Timestamp
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_health_service_timestamp ON system_health(service_name, timestamp);
CREATE INDEX idx_health_status_timestamp ON system_health(status, timestamp);

-- Configuration table
CREATE TABLE configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Configuration details
    key VARCHAR(100) UNIQUE NOT NULL,
    value TEXT NOT NULL,
    data_type VARCHAR(20) NOT NULL, -- string, integer, boolean, json
    category VARCHAR(50) NOT NULL,
    description TEXT,
    
    -- Security
    is_sensitive BOOLEAN DEFAULT FALSE,
    is_readonly BOOLEAN DEFAULT FALSE,
    
    -- Versioning
    version INTEGER DEFAULT 1,
    previous_value TEXT,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by UUID REFERENCES users(id)
);

CREATE INDEX idx_config_category_key ON configurations(category, key);

-- API rate limits table
CREATE TABLE api_rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Rate limit details
    identifier VARCHAR(255) NOT NULL, -- user_id, ip_address, api_key
    identifier_type VARCHAR(20) NOT NULL, -- user, ip, api_key
    endpoint VARCHAR(255) NOT NULL,
    
    -- Counters
    request_count INTEGER DEFAULT 0,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Status
    is_blocked BOOLEAN DEFAULT FALSE,
    blocked_until TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_rate_limit_identifier_endpoint ON api_rate_limits(identifier, endpoint);
CREATE INDEX idx_rate_limit_window ON api_rate_limits(window_end);

-- Insert default roles
INSERT INTO roles (name, description, permissions) VALUES
('admin', 'System Administrator', 
 '["workflow.create", "workflow.read", "workflow.update", "workflow.delete", 
   "user.create", "user.read", "user.update", "user.delete", 
   "system.monitor", "system.configure"]'::JSONB),
   
('workflow_manager', 'Workflow Manager', 
 '["workflow.create", "workflow.read", "workflow.update", 
   "application.read", "application.update", "application.assign"]'::JSONB),
   
('underwriter', 'Underwriter', 
 '["application.read", "application.update", "application.approve", 
   "application.reject", "document.read", "document.upload"]'::JSONB),
   
('operations', 'Operations Team', 
 '["application.read", "application.update", "application.process", 
   "document.read", "document.upload", "document.verify"]'::JSONB),
   
('credit_team', 'Credit Team', 
 '["application.read", "application.assess", "credit.analyze", 
   "document.read", "report.generate"]'::JSONB),
   
('legal_team', 'Legal Team', 
 '["application.read", "legal.review", "document.read", 
   "legal.approve", "legal.reject"]'::JSONB),
   
('technical_team', 'Technical Team', 
 '["application.read", "technical.review", "document.read", 
   "technical.approve", "technical.reject"]'::JSONB),
   
('viewer', 'Read Only Viewer', 
 '["application.read", "document.read", "report.read"]'::JSONB);

-- Insert default system configurations
INSERT INTO configurations (key, value, data_type, category, description, is_readonly) VALUES
('system.name', 'Workflow Management System', 'string', 'system', 'System name', true),
('system.version', '1.0.0', 'string', 'system', 'System version', true),
('auth.session_timeout_minutes', '60', 'integer', 'security', 'Session timeout in minutes', false),
('auth.max_failed_attempts', '5', 'integer', 'security', 'Maximum failed login attempts', false),
('rate_limit.default_per_minute', '100', 'integer', 'api', 'Default rate limit per minute', false),
('monitoring.health_check_interval', '30', 'integer', 'monitoring', 'Health check interval in seconds', false);

-- Create trigger function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_roles_updated_at BEFORE UPDATE ON roles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_configurations_updated_at BEFORE UPDATE ON configurations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create a default admin user (password: admin123!)
-- In production, this should be changed immediately
INSERT INTO users (username, email, full_name, hashed_password, is_active, is_verified) VALUES
('admin', 'admin@example.com', 'System Administrator', 
 '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeGwuKhGtlOEefULu', -- admin123!
 true, true);

-- Assign admin role to the default admin user
INSERT INTO user_roles (user_id, role_id) 
SELECT u.id, r.id 
FROM users u, roles r 
WHERE u.username = 'admin' AND r.name = 'admin';

-- Create basic performance indexes
CREATE INDEX CONCURRENTLY idx_users_created_at ON users(created_at);
CREATE INDEX CONCURRENTLY idx_audit_logs_timestamp_desc ON audit_logs(timestamp DESC);
CREATE INDEX CONCURRENTLY idx_system_health_timestamp_desc ON system_health(timestamp DESC);