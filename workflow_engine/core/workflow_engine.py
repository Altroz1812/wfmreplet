"""
Core Workflow Engine for executing complex business processes.

This module provides the main workflow execution engine with state management,
step execution, and AI-powered decision making capabilities.
"""

import uuid
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import json
import logging

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database.connection import db_manager
from ..auth.security import security_manager
from ..config.settings import settings


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepType(str, Enum):
    """Types of workflow steps."""
    TASK = "task"
    DECISION = "decision"
    PARALLEL = "parallel"
    LOOP = "loop"
    AI_ANALYSIS = "ai_analysis"
    HUMAN_APPROVAL = "human_approval"
    API_CALL = "api_call"
    DATA_TRANSFORM = "data_transform"


class StepStatus(str, Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class WorkflowContext:
    """Workflow execution context containing all runtime data."""
    workflow_id: str
    instance_id: str
    current_step: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    step_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    workflow_metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowStep(BaseModel):
    """Definition of a workflow step."""
    id: str
    name: str
    type: StepType
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    error_handling: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    retries: int = 0
    ai_prompt: Optional[str] = None
    requires_approval: bool = False


class WorkflowDefinition(BaseModel):
    """Complete workflow definition schema."""
    id: str
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    category: str = "general"
    steps: List[WorkflowStep]
    start_step: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    permissions: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkflowEngine:
    """Main workflow execution engine."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_workflows: Dict[str, WorkflowContext] = {}
        self.step_handlers: Dict[StepType, Callable] = {
            StepType.TASK: self._execute_task_step,
            StepType.DECISION: self._execute_decision_step,
            StepType.PARALLEL: self._execute_parallel_step,
            StepType.AI_ANALYSIS: self._execute_ai_step,
            StepType.HUMAN_APPROVAL: self._execute_approval_step,
            StepType.API_CALL: self._execute_api_step,
            StepType.DATA_TRANSFORM: self._execute_transform_step,
            StepType.LOOP: self._execute_loop_step,
        }
    
    async def create_workflow_instance(
        self, 
        workflow_definition: WorkflowDefinition,
        initial_data: Dict[str, Any] = None,
        user_id: Optional[str] = None
    ) -> WorkflowContext:
        """Create a new workflow instance."""
        instance_id = str(uuid.uuid4())
        
        context = WorkflowContext(
            workflow_id=workflow_definition.id,
            instance_id=instance_id,
            current_step=workflow_definition.start_step,
            variables=initial_data or {},
            workflow_metadata={
                "workflow_name": workflow_definition.name,
                "workflow_version": workflow_definition.version,
                "created_by": user_id,
                "category": workflow_definition.category
            }
        )
        
        # Merge workflow default variables
        if workflow_definition.variables:
            context.variables.update(workflow_definition.variables)
        
        self.active_workflows[instance_id] = context
        
        self.logger.info(
            f"Created workflow instance {instance_id} for workflow {workflow_definition.id}",
            extra={
                "workflow_id": workflow_definition.id,
                "instance_id": instance_id,
                "user_id": user_id
            }
        )
        
        # Store in database
        await self._persist_workflow_instance(context, workflow_definition)
        
        return context
    
    async def execute_workflow(
        self, 
        instance_id: str,
        workflow_definition: WorkflowDefinition
    ) -> WorkflowContext:
        """Execute a workflow instance."""
        context = self.active_workflows.get(instance_id)
        if not context:
            raise ValueError(f"Workflow instance {instance_id} not found")
        
        self.logger.info(f"Starting execution of workflow instance {instance_id}")
        
        try:
            while context.current_step:
                step = self._get_step_by_id(workflow_definition, context.current_step)
                if not step:
                    raise ValueError(f"Step {context.current_step} not found in workflow definition")
                
                self.logger.info(f"Executing step {step.id}: {step.name}")
                
                # Record step start
                step_execution = {
                    "step_id": step.id,
                    "step_name": step.name,
                    "step_type": step.type.value,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "status": StepStatus.RUNNING.value
                }
                
                try:
                    # Execute the step
                    handler = self.step_handlers.get(step.type)
                    if not handler:
                        raise ValueError(f"No handler found for step type {step.type}")
                    
                    result = await handler(step, context, workflow_definition)
                    
                    # Update step execution record
                    step_execution.update({
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "status": StepStatus.COMPLETED.value,
                        "result": result
                    })
                    
                    # Determine next step
                    next_step = self._determine_next_step(step, result, context)
                    context.current_step = next_step
                    
                    self.logger.info(f"Step {step.id} completed successfully. Next step: {next_step}")
                    
                except Exception as e:
                    # Handle step failure
                    step_execution.update({
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "status": StepStatus.FAILED.value,
                        "error": str(e)
                    })
                    
                    self.logger.error(f"Step {step.id} failed: {str(e)}")
                    
                    # Check for error handling
                    if step.error_handling:
                        context.current_step = step.error_handling.get("next_step")
                    else:
                        context.current_step = None  # Stop workflow
                        break
                
                finally:
                    context.step_history.append(step_execution)
                    context.updated_at = datetime.now(timezone.utc)
                    await self._persist_workflow_instance(context, workflow_definition)
            
            # Workflow completed
            self.logger.info(f"Workflow instance {instance_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Workflow instance {instance_id} failed: {str(e)}")
            context.current_step = None
            await self._persist_workflow_instance(context, workflow_definition)
            raise
        
        return context
    
    async def pause_workflow(self, instance_id: str) -> bool:
        """Pause a running workflow."""
        context = self.active_workflows.get(instance_id)
        if not context:
            return False
        
        context.workflow_metadata["status"] = WorkflowStatus.PAUSED.value
        context.updated_at = datetime.now(timezone.utc)
        
        self.logger.info(f"Paused workflow instance {instance_id}")
        return True
    
    async def resume_workflow(
        self, 
        instance_id: str, 
        workflow_definition: WorkflowDefinition
    ) -> WorkflowContext:
        """Resume a paused workflow."""
        context = self.active_workflows.get(instance_id)
        if not context:
            raise ValueError(f"Workflow instance {instance_id} not found")
        
        context.workflow_metadata["status"] = WorkflowStatus.RUNNING.value
        context.updated_at = datetime.now(timezone.utc)
        
        self.logger.info(f"Resuming workflow instance {instance_id}")
        
        return await self.execute_workflow(instance_id, workflow_definition)
    
    def _get_step_by_id(self, workflow_definition: WorkflowDefinition, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by its ID from the workflow definition."""
        return next((step for step in workflow_definition.steps if step.id == step_id), None)
    
    def _determine_next_step(
        self, 
        current_step: WorkflowStep, 
        result: Dict[str, Any], 
        context: WorkflowContext
    ) -> Optional[str]:
        """Determine the next step based on current step result and conditions."""
        if not current_step.next_steps:
            return None
        
        # Simple case: single next step
        if len(current_step.next_steps) == 1:
            return current_step.next_steps[0]
        
        # Complex case: conditional branching
        for condition in current_step.conditions:
            if self._evaluate_condition(condition, result, context):
                return condition.get("next_step")
        
        # Default: first next step
        return current_step.next_steps[0] if current_step.next_steps else None
    
    def _evaluate_condition(
        self, 
        condition: Dict[str, Any], 
        result: Dict[str, Any], 
        context: WorkflowContext
    ) -> bool:
        """Evaluate a condition against step result and context."""
        # Simple condition evaluation logic
        field = condition.get("field")
        operator = condition.get("operator", "equals")
        value = condition.get("value")
        
        if not field:
            return True
        
        actual_value = result.get(field) or context.variables.get(field)
        
        if operator == "equals":
            return actual_value == value
        elif operator == "not_equals":
            return actual_value != value
        elif operator == "greater_than":
            try:
                return float(actual_value or 0) > float(value or 0)
            except (ValueError, TypeError):
                return False
        elif operator == "less_than":
            try:
                return float(actual_value or 0) < float(value or 0)
            except (ValueError, TypeError):
                return False
        elif operator == "contains":
            return str(value).lower() in str(actual_value or "").lower()
        
        return True
    
    async def _execute_task_step(
        self, 
        step: WorkflowStep, 
        context: WorkflowContext,
        workflow_definition: WorkflowDefinition
    ) -> Dict[str, Any]:
        """Execute a basic task step."""
        # Basic task execution logic
        task_config = step.config
        task_type = task_config.get("task_type", "generic")
        
        result = {
            "step_id": step.id,
            "task_type": task_type,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Update context variables if specified
        if "output_variables" in task_config:
            for var_name, var_value in task_config["output_variables"].items():
                context.variables[var_name] = var_value
        
        return result
    
    async def _execute_decision_step(
        self, 
        step: WorkflowStep, 
        context: WorkflowContext,
        workflow_definition: WorkflowDefinition
    ) -> Dict[str, Any]:
        """Execute a decision step with conditional logic."""
        decision_config = step.config
        decision_field = decision_config.get("decision_field")
        decision_value = context.variables.get(decision_field)
        
        result = {
            "step_id": step.id,
            "decision_field": decision_field,
            "decision_value": decision_value,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return result
    
    async def _execute_parallel_step(
        self, 
        step: WorkflowStep, 
        context: WorkflowContext,
        workflow_definition: WorkflowDefinition
    ) -> Dict[str, Any]:
        """Execute parallel steps concurrently."""
        parallel_steps = step.config.get("parallel_steps", [])
        
        # Execute all parallel steps concurrently
        tasks = []
        for parallel_step_id in parallel_steps:
            parallel_step = self._get_step_by_id(workflow_definition, parallel_step_id)
            if parallel_step:
                handler = self.step_handlers.get(parallel_step.type)
                if handler:
                    tasks.append(handler(parallel_step, context, workflow_definition))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
        
        return {
            "step_id": step.id,
            "parallel_results": results,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _execute_ai_step(
        self, 
        step: WorkflowStep, 
        context: WorkflowContext,
        workflow_definition: WorkflowDefinition
    ) -> Dict[str, Any]:
        """Execute an AI analysis step."""
        # This will be implemented with OpenAI integration
        ai_config = step.config
        prompt = step.ai_prompt or ai_config.get("prompt", "")
        
        # Placeholder for AI integration - will be enhanced with actual OpenAI calls
        result = {
            "step_id": step.id,
            "ai_type": ai_config.get("ai_type", "analysis"),
            "prompt": prompt,
            "ai_response": "AI analysis placeholder - to be implemented",
            "confidence": 0.85,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return result
    
    async def _execute_approval_step(
        self, 
        step: WorkflowStep, 
        context: WorkflowContext,
        workflow_definition: WorkflowDefinition
    ) -> Dict[str, Any]:
        """Execute a human approval step."""
        approval_config = step.config
        
        # Mark step as requiring human approval
        result = {
            "step_id": step.id,
            "approval_type": approval_config.get("approval_type", "standard"),
            "required_roles": approval_config.get("required_roles", []),
            "status": "pending_approval",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Set workflow to paused state for human approval
        context.workflow_metadata["status"] = WorkflowStatus.PAUSED.value
        context.workflow_metadata["pending_approval"] = {
            "step_id": step.id,
            "approval_type": approval_config.get("approval_type"),
            "message": approval_config.get("message", f"Approval required for step: {step.name}")
        }
        
        return result
    
    async def _execute_api_step(
        self, 
        step: WorkflowStep, 
        context: WorkflowContext,
        workflow_definition: WorkflowDefinition
    ) -> Dict[str, Any]:
        """Execute an API call step."""
        api_config = step.config
        
        # Placeholder for API call execution
        result = {
            "step_id": step.id,
            "api_endpoint": api_config.get("endpoint"),
            "api_method": api_config.get("method", "GET"),
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response": "API call placeholder - to be implemented"
        }
        
        return result
    
    async def _execute_transform_step(
        self, 
        step: WorkflowStep, 
        context: WorkflowContext,
        workflow_definition: WorkflowDefinition
    ) -> Dict[str, Any]:
        """Execute a data transformation step."""
        transform_config = step.config
        
        # Basic data transformation logic
        result = {
            "step_id": step.id,
            "transform_type": transform_config.get("transform_type", "generic"),
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Apply transformations to context variables
        transformations = transform_config.get("transformations", {})
        for var_name, transformation in transformations.items():
            if var_name in context.variables and isinstance(transformation, dict):
                # Simple transformation logic - can be extended
                transform_type = transformation.get("type")
                if transform_type == "uppercase":
                    context.variables[var_name] = str(context.variables[var_name]).upper()
                elif transform_type == "lowercase":
                    context.variables[var_name] = str(context.variables[var_name]).lower()
                elif transform_type == "multiply":
                    factor = transformation.get("factor", 1)
                    try:
                        context.variables[var_name] = float(context.variables[var_name]) * float(factor)
                    except (ValueError, TypeError):
                        pass
        
        return result
    
    async def _execute_loop_step(
        self, 
        step: WorkflowStep, 
        context: WorkflowContext,
        workflow_definition: WorkflowDefinition
    ) -> Dict[str, Any]:
        """Execute a loop step with iteration control."""
        loop_config = step.config
        loop_type = loop_config.get("loop_type", "while")
        max_iterations = loop_config.get("max_iterations", 100)
        
        result = {
            "step_id": step.id,
            "loop_type": loop_type,
            "iterations_completed": 0,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Simple implementation - can be extended for complex loop logic
        if loop_type == "while":
            condition = loop_config.get("condition", {})
            iterations = 0
            
            while iterations < max_iterations and self._evaluate_condition(condition, result, context):
                # Execute loop body steps
                loop_steps = loop_config.get("loop_steps", [])
                for loop_step_id in loop_steps:
                    loop_step = self._get_step_by_id(workflow_definition, loop_step_id)
                    if loop_step:
                        handler = self.step_handlers.get(loop_step.type)
                        if handler:
                            await handler(loop_step, context, workflow_definition)
                
                iterations += 1
                result["iterations_completed"] = iterations
        
        elif loop_type == "for":
            items = context.variables.get(loop_config.get("items_variable", "items"), [])
            for i, item in enumerate(items[:max_iterations]):
                context.variables["loop_item"] = item
                context.variables["loop_index"] = i
                
                # Execute loop body steps
                loop_steps = loop_config.get("loop_steps", [])
                for loop_step_id in loop_steps:
                    loop_step = self._get_step_by_id(workflow_definition, loop_step_id)
                    if loop_step:
                        handler = self.step_handlers.get(loop_step.type)
                        if handler:
                            await handler(loop_step, context, workflow_definition)
                
                result["iterations_completed"] = i + 1
        
        return result
    
    async def _persist_workflow_instance(
        self, 
        context: WorkflowContext, 
        workflow_definition: WorkflowDefinition
    ):
        """Persist workflow instance to database."""
        try:
            with db_manager.get_session() as session:
                # This will be implemented when database models are ready
                pass
        except Exception as e:
            self.logger.error(f"Failed to persist workflow instance: {str(e)}")
    
    def get_workflow_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status."""
        context = self.active_workflows.get(instance_id)
        if not context:
            return None
        
        return {
            "instance_id": instance_id,
            "workflow_id": context.workflow_id,
            "current_step": context.current_step,
            "status": context.workflow_metadata.get("status", WorkflowStatus.RUNNING.value),
            "progress": len(context.step_history),
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat(),
            "variables": context.variables,
            "step_history": context.step_history
        }


# Global workflow engine instance
workflow_engine = WorkflowEngine()