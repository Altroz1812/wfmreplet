"""
Workflow Management REST API Endpoints.

This module provides REST API endpoints for workflow definition management,
workflow instance execution, monitoring, and control operations.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...database.connection import db_manager
from ...database.models import WorkflowDefinition, WorkflowInstance, WorkflowStepExecution, WorkflowApproval, User
from ...auth.security import security_manager, TokenData
from ...core.workflow_engine import workflow_engine, WorkflowStatus, StepStatus
from ...core.ai_processor import ai_processor
import logging

router = APIRouter(prefix="/workflows", tags=["workflows"])
logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class WorkflowDefinitionCreate(BaseModel):
    """Request model for creating workflow definitions."""
    name: str = Field(..., min_length=1, max_length=255)
    version: str = Field(default="1.0.0", max_length=50)
    description: Optional[str] = None
    category: str = Field(default="general", max_length=100)
    definition: Dict[str, Any] = Field(..., description="Complete workflow definition JSON")
    variables: Dict[str, Any] = Field(default_factory=dict)
    permissions: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class WorkflowDefinitionResponse(BaseModel):
    """Response model for workflow definitions."""
    id: str
    name: str
    version: str
    description: Optional[str]
    category: str
    definition: Dict[str, Any]
    variables: Dict[str, Any]
    permissions: List[str]
    tags: List[str]
    is_active: bool
    is_published: bool
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]


class WorkflowInstanceCreate(BaseModel):
    """Request model for creating workflow instances."""
    workflow_definition_id: str
    name: Optional[str] = None
    initial_variables: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowInstanceResponse(BaseModel):
    """Response model for workflow instances."""
    id: str
    workflow_definition_id: str
    name: Optional[str]
    status: str
    current_step: Optional[str]
    variables: Dict[str, Any]
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    progress_percentage: int
    steps_completed: int
    steps_total: int
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    last_activity_at: datetime
    created_at: datetime
    created_by: Optional[str]


class WorkflowExecutionRequest(BaseModel):
    """Request model for workflow execution."""
    auto_execute: bool = Field(default=True, description="Automatically execute workflow steps")
    execution_options: Dict[str, Any] = Field(default_factory=dict)


class WorkflowApprovalRequest(BaseModel):
    """Request model for workflow approval."""
    action: str = Field(..., description="approve, reject, or request_changes")
    comment: Optional[str] = None
    additional_data: Dict[str, Any] = Field(default_factory=dict)


# Workflow Definition Management
@router.post("/definitions", response_model=WorkflowDefinitionResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow_definition(
    definition_data: WorkflowDefinitionCreate,
    current_user: TokenData = Depends(security_manager.get_current_user),
    db: Session = Depends(db_manager.get_session)
):
    """Create a new workflow definition."""
    try:
        # Validate permissions
        if not security_manager.check_permission(current_user, "workflow:create"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to create workflows"
            )
        
        # Check if workflow with same name and version exists
        existing = db.query(WorkflowDefinition).filter(
            WorkflowDefinition.name == definition_data.name,
            WorkflowDefinition.version == definition_data.version
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Workflow {definition_data.name} version {definition_data.version} already exists"
            )
        
        # Create new workflow definition
        workflow_def = WorkflowDefinition(
            id=uuid.uuid4(),
            name=definition_data.name,
            version=definition_data.version,
            description=definition_data.description,
            category=definition_data.category,
            definition=definition_data.definition,
            variables=definition_data.variables,
            permissions=definition_data.permissions,
            tags=definition_data.tags,
            created_by=uuid.UUID(current_user.user_id)
        )
        
        db.add(workflow_def)
        db.commit()
        db.refresh(workflow_def)
        
        logger.info(f"Created workflow definition {workflow_def.name} v{workflow_def.version}")
        
        return WorkflowDefinitionResponse(
            id=str(workflow_def.id),
            name=workflow_def.name,
            version=workflow_def.version,
            description=workflow_def.description,
            category=workflow_def.category,
            definition=workflow_def.definition,
            variables=workflow_def.variables or {},
            permissions=workflow_def.permissions or [],
            tags=workflow_def.tags or [],
            is_active=workflow_def.is_active,
            is_published=workflow_def.is_published,
            created_at=workflow_def.created_at,
            updated_at=workflow_def.updated_at,
            created_by=str(workflow_def.created_by) if workflow_def.created_by else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workflow definition: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create workflow definition"
        )


@router.get("/definitions", response_model=List[WorkflowDefinitionResponse])
async def list_workflow_definitions(
    category: Optional[str] = Query(None, description="Filter by category"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    active_only: bool = Query(True, description="Show only active workflows"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    current_user: TokenData = Depends(security_manager.get_current_user),
    db: Session = Depends(db_manager.get_session)
):
    """List workflow definitions with filtering options."""
    try:
        # Build query
        query = db.query(WorkflowDefinition)
        
        if active_only:
            query = query.filter(WorkflowDefinition.is_active == True)
        
        if category:
            query = query.filter(WorkflowDefinition.category == category)
        
        if tag:
            query = query.filter(WorkflowDefinition.tags.op('?')(tag))
        
        # Apply pagination
        workflows = query.offset(offset).limit(limit).all()
        
        return [
            WorkflowDefinitionResponse(
                id=str(wf.id),
                name=wf.name,
                version=wf.version,
                description=wf.description,
                category=wf.category,
                definition=wf.definition,
                variables=wf.variables or {},
                permissions=wf.permissions or [],
                tags=wf.tags or [],
                is_active=wf.is_active,
                is_published=wf.is_published,
                created_at=wf.created_at,
                updated_at=wf.updated_at,
                created_by=str(wf.created_by) if wf.created_by else None
            )
            for wf in workflows
        ]
        
    except Exception as e:
        logger.error(f"Error listing workflow definitions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workflow definitions"
        )


@router.get("/definitions/{definition_id}", response_model=WorkflowDefinitionResponse)
async def get_workflow_definition(
    definition_id: str,
    current_user: TokenData = Depends(security_manager.get_current_user),
    db: Session = Depends(db_manager.get_session)
):
    """Get a specific workflow definition by ID."""
    try:
        workflow_def = db.query(WorkflowDefinition).filter(
            WorkflowDefinition.id == uuid.UUID(definition_id)
        ).first()
        
        if not workflow_def:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow definition not found"
            )
        
        return WorkflowDefinitionResponse(
            id=str(workflow_def.id),
            name=workflow_def.name,
            version=workflow_def.version,
            description=workflow_def.description,
            category=workflow_def.category,
            definition=workflow_def.definition,
            variables=workflow_def.variables or {},
            permissions=workflow_def.permissions or [],
            tags=workflow_def.tags or [],
            is_active=workflow_def.is_active,
            is_published=workflow_def.is_published,
            created_at=workflow_def.created_at,
            updated_at=workflow_def.updated_at,
            created_by=str(workflow_def.created_by) if workflow_def.created_by else None
        )
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid workflow definition ID format"
        )
    except Exception as e:
        logger.error(f"Error getting workflow definition {definition_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workflow definition"
        )


# Workflow Instance Management
@router.post("/instances", response_model=WorkflowInstanceResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow_instance(
    instance_data: WorkflowInstanceCreate,
    current_user: TokenData = Depends(security_manager.get_current_user),
    db: Session = Depends(db_manager.get_session)
):
    """Create a new workflow instance."""
    try:
        # Validate permissions
        if not security_manager.check_permission(current_user, "workflow:execute"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to create workflow instances"
            )
        
        # Get workflow definition
        workflow_def = db.query(WorkflowDefinition).filter(
            WorkflowDefinition.id == uuid.UUID(instance_data.workflow_definition_id)
        ).first()
        
        if not workflow_def or not workflow_def.is_active:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow definition not found or inactive"
            )
        
        # Create workflow instance in database
        workflow_instance = WorkflowInstance(
            id=uuid.uuid4(),
            workflow_definition_id=workflow_def.id,
            name=instance_data.name,
            variables=instance_data.initial_variables,
            metadata=instance_data.metadata,
            created_by=uuid.UUID(current_user.user_id)
        )
        
        db.add(workflow_instance)
        db.commit()
        db.refresh(workflow_instance)
        
        logger.info(f"Created workflow instance {workflow_instance.id}")
        
        return WorkflowInstanceResponse(
            id=str(workflow_instance.id),
            workflow_definition_id=str(workflow_instance.workflow_definition_id),
            name=workflow_instance.name,
            status=workflow_instance.status,
            current_step=workflow_instance.current_step,
            variables=workflow_instance.variables or {},
            context=workflow_instance.context or {},
            metadata=workflow_instance.metadata or {},
            progress_percentage=workflow_instance.progress_percentage,
            steps_completed=workflow_instance.steps_completed,
            steps_total=workflow_instance.steps_total,
            error_message=workflow_instance.error_message,
            started_at=workflow_instance.started_at,
            completed_at=workflow_instance.completed_at,
            last_activity_at=workflow_instance.last_activity_at,
            created_at=workflow_instance.created_at,
            created_by=str(workflow_instance.created_by) if workflow_instance.created_by else None
        )
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid workflow definition ID format"
        )
    except Exception as e:
        logger.error(f"Error creating workflow instance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create workflow instance"
        )


@router.post("/instances/{instance_id}/execute")
async def execute_workflow_instance(
    instance_id: str,
    execution_request: WorkflowExecutionRequest = Body(...),
    current_user: TokenData = Depends(security_manager.get_current_user),
    db: Session = Depends(db_manager.get_session)
):
    """Execute a workflow instance."""
    try:
        # Validate permissions
        if not security_manager.check_permission(current_user, "workflow:execute"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to execute workflows"
            )
        
        # Get workflow instance and definition
        workflow_instance = db.query(WorkflowInstance).filter(
            WorkflowInstance.id == uuid.UUID(instance_id)
        ).first()
        
        if not workflow_instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow instance not found"
            )
        
        workflow_def = db.query(WorkflowDefinition).filter(
            WorkflowDefinition.id == workflow_instance.workflow_definition_id
        ).first()
        
        if not workflow_def:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow definition not found"
            )
        
        # Convert database model to engine format
        from ...core.workflow_engine import WorkflowDefinition as EngineWorkflowDef
        
        engine_workflow_def = EngineWorkflowDef(
            id=str(workflow_def.id),
            name=workflow_def.name,
            version=workflow_def.version,
            description=workflow_def.description,
            category=workflow_def.category,
            steps=workflow_def.definition.get("steps", []),
            start_step=workflow_def.definition.get("start_step"),
            variables=workflow_def.variables or {},
            permissions=workflow_def.permissions or [],
            tags=workflow_def.tags or [],
            created_by=str(workflow_def.created_by) if workflow_def.created_by else None
        )
        
        # Create or get workflow context
        context = workflow_engine.active_workflows.get(instance_id)
        if not context:
            context = await workflow_engine.create_workflow_instance(
                engine_workflow_def,
                workflow_instance.variables or {},
                current_user.user_id
            )
        
        # Execute workflow
        if execution_request.auto_execute:
            context = await workflow_engine.execute_workflow(instance_id, engine_workflow_def)
        
        # Update database
        workflow_instance.status = context.metadata.get("status", WorkflowStatus.RUNNING.value)
        workflow_instance.current_step = context.current_step
        workflow_instance.variables = context.variables
        workflow_instance.last_activity_at = datetime.now(timezone.utc)
        
        if not workflow_instance.started_at:
            workflow_instance.started_at = datetime.now(timezone.utc)
        
        if context.current_step is None:
            workflow_instance.completed_at = datetime.now(timezone.utc)
            workflow_instance.status = WorkflowStatus.COMPLETED.value
        
        db.commit()
        
        return {
            "instance_id": instance_id,
            "status": workflow_instance.status,
            "current_step": workflow_instance.current_step,
            "message": "Workflow execution started successfully" if execution_request.auto_execute else "Workflow instance prepared for execution"
        }
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid instance ID format"
        )
    except Exception as e:
        logger.error(f"Error executing workflow instance {instance_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute workflow instance"
        )


@router.get("/instances/{instance_id}/status")
async def get_workflow_status(
    instance_id: str,
    current_user: TokenData = Depends(security_manager.get_current_user),
    db: Session = Depends(db_manager.get_session)
):
    """Get workflow instance status and progress."""
    try:
        # Get from workflow engine first
        engine_status = workflow_engine.get_workflow_status(instance_id)
        
        if engine_status:
            return engine_status
        
        # Fallback to database
        workflow_instance = db.query(WorkflowInstance).filter(
            WorkflowInstance.id == uuid.UUID(instance_id)
        ).first()
        
        if not workflow_instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow instance not found"
            )
        
        return {
            "instance_id": instance_id,
            "workflow_id": str(workflow_instance.workflow_definition_id),
            "current_step": workflow_instance.current_step,
            "status": workflow_instance.status,
            "progress": workflow_instance.progress_percentage,
            "created_at": workflow_instance.created_at.isoformat(),
            "updated_at": workflow_instance.updated_at.isoformat(),
            "variables": workflow_instance.variables or {},
            "error_message": workflow_instance.error_message
        }
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid instance ID format"
        )
    except Exception as e:
        logger.error(f"Error getting workflow status {instance_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workflow status"
        )


@router.post("/instances/{instance_id}/approve/{step_id}")
async def approve_workflow_step(
    instance_id: str,
    step_id: str,
    approval_request: WorkflowApprovalRequest,
    current_user: TokenData = Depends(security_manager.get_current_user),
    db: Session = Depends(db_manager.get_session)
):
    """Approve or reject a workflow step requiring human approval."""
    try:
        # Get workflow approval record
        approval = db.query(WorkflowApproval).filter(
            WorkflowApproval.workflow_instance_id == uuid.UUID(instance_id),
            WorkflowApproval.status == "pending"
        ).first()
        
        if not approval:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No pending approval found for this workflow instance"
            )
        
        # Validate user has required role for this approval
        required_roles = approval.required_roles or []
        user_roles = [role.lower() for role in current_user.roles]
        
        if required_roles and not any(role.lower() in user_roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User role required: {', '.join(required_roles)}"
            )
        
        # Update approval
        approval.status = approval_request.action
        approval.approved_by = uuid.UUID(current_user.user_id)
        approval.approval_comment = approval_request.comment
        approval.responded_at = datetime.now(timezone.utc)
        
        db.commit()
        
        # Resume workflow if approved
        if approval_request.action == "approve":
            # Update workflow instance to resume execution
            workflow_instance = db.query(WorkflowInstance).filter(
                WorkflowInstance.id == uuid.UUID(instance_id)
            ).first()
            
            if workflow_instance:
                workflow_instance.status = WorkflowStatus.RUNNING.value
                workflow_instance.last_activity_at = datetime.now(timezone.utc)
                db.commit()
        
        logger.info(f"Workflow step {step_id} {approval_request.action} by {current_user.username}")
        
        return {
            "instance_id": instance_id,
            "step_id": step_id,
            "action": approval_request.action,
            "message": f"Workflow step {approval_request.action} successfully",
            "approved_by": current_user.username,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ID format"
        )
    except Exception as e:
        logger.error(f"Error approving workflow step: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process approval"
        )


@router.get("/categories")
async def get_workflow_categories(
    current_user: TokenData = Depends(security_manager.get_current_user),
    db: Session = Depends(db_manager.get_session)
):
    """Get list of workflow categories."""
    try:
        categories = db.query(WorkflowDefinition.category).distinct().all()
        return {
            "categories": [cat[0] for cat in categories if cat[0]]
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow categories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workflow categories"
        )