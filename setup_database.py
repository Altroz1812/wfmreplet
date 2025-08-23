#!/usr/bin/env python3
"""
Database setup script for Workflow Management System.
This script initializes the database with the required schema and default data.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from workflow_engine.config.settings import settings
from workflow_engine.database.models import Base


def create_database():
    """Create database tables and initial data."""
    
    print("Setting up Workflow Management System database...")
    
    # Create engine
    engine = create_engine(settings.database.database_url_computed, echo=True)
    
    # Create all tables
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Load and execute initial schema
    schema_file = Path("workflow_engine/database/migrations/001_initial_schema.sql")
    if schema_file.exists():
        print("Executing initial schema migration...")
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Split into individual statements and execute
        with engine.connect() as conn:
            # Execute each statement separately to handle potential conflicts
            statements = schema_sql.split(';')
            for statement in statements:
                if statement.strip() and not statement.strip().startswith('--'):
                    try:
                        conn.execute(text(statement))
                        conn.commit()
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            print(f"Warning: {e}")
                        continue
    
    print("Database setup completed successfully!")
    print(f"Database URL: {settings.database.database_url_computed}")


if __name__ == "__main__":
    try:
        create_database()
    except Exception as e:
        print(f"Error setting up database: {e}")
        sys.exit(1)