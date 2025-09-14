"""
Loan Application Workflow Definition.

This module defines a comprehensive loan application processing workflow
that includes document verification, AI risk assessment, credit analysis,
and approval processing with human oversight.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
import uuid

from ..core.workflow_engine import WorkflowDefinition, WorkflowStep, StepType
from ..core.ai_processor import AITaskType


def create_loan_application_workflow() -> WorkflowDefinition:
    """Create a comprehensive loan application processing workflow."""
    
    # Define workflow steps
    steps = [
        # Initial Application Processing
        WorkflowStep(
            id="receive_application",
            name="Receive Loan Application",
            type=StepType.TASK,
            description="Receive and validate initial loan application data",
            config={
                "task_type": "data_validation",
                "required_fields": [
                    "applicant_name", "loan_amount", "loan_purpose", 
                    "employment_status", "annual_income", "credit_score"
                ],
                "validation_rules": {
                    "loan_amount": {"min": 1000, "max": 500000},
                    "annual_income": {"min": 0},
                    "credit_score": {"min": 300, "max": 850}
                },
                "output_variables": {
                    "application_received": True,
                    "validation_status": "passed"
                }
            },
            next_steps=["document_collection"]
        ),
        
        # Document Collection and Verification
        WorkflowStep(
            id="document_collection",
            name="Collect Required Documents",
            type=StepType.TASK,
            description="Collect and categorize required documents",
            config={
                "task_type": "document_collection",
                "required_documents": [
                    "identity_verification", "income_verification",
                    "employment_verification", "bank_statements",
                    "credit_report"
                ],
                "document_types": {
                    "identity_verification": ["passport", "driver_license", "national_id"],
                    "income_verification": ["tax_returns", "pay_stubs", "bank_statements"],
                    "employment_verification": ["employment_letter", "contract"]
                }
            },
            next_steps=["ai_document_analysis"]
        ),
        
        # AI Document Analysis
        WorkflowStep(
            id="ai_document_analysis",
            name="AI Document Analysis",
            type=StepType.AI_ANALYSIS,
            description="Use AI to analyze and extract information from documents",
            ai_prompt="Analyze the provided loan application documents for completeness, authenticity, and consistency. Extract key financial information and identify any red flags.",
            config={
                "ai_type": "document_analysis",
                "analysis_scope": [
                    "document_authenticity",
                    "data_extraction",
                    "consistency_check",
                    "completeness_assessment"
                ],
                "confidence_threshold": 0.8,
                "extraction_schema": {
                    "monthly_income": "number",
                    "employment_tenure": "string",
                    "debt_obligations": "number",
                    "assets_value": "number"
                }
            },
            conditions=[
                {
                    "field": "ai_confidence",
                    "operator": "greater_than",
                    "value": 0.8,
                    "next_step": "credit_assessment"
                },
                {
                    "field": "ai_confidence",
                    "operator": "less_than",
                    "value": 0.8,
                    "next_step": "manual_document_review"
                }
            ],
            next_steps=["credit_assessment", "manual_document_review"],
            timeout_seconds=300
        ),
        
        # Manual Document Review (if AI confidence is low)
        WorkflowStep(
            id="manual_document_review",
            name="Manual Document Review",
            type=StepType.HUMAN_APPROVAL,
            description="Human review of documents when AI confidence is low",
            requires_approval=True,
            config={
                "approval_type": "document_review",
                "required_roles": ["loan_officer", "document_specialist"],
                "message": "AI document analysis confidence is below threshold. Manual review required.",
                "approval_criteria": [
                    "Document authenticity verified",
                    "Financial information extracted accurately",
                    "No inconsistencies found"
                ]
            },
            next_steps=["credit_assessment"]
        ),
        
        # Credit Assessment and Risk Analysis
        WorkflowStep(
            id="credit_assessment",
            name="Credit Assessment & Risk Analysis",
            type=StepType.AI_ANALYSIS,
            description="Comprehensive credit and risk assessment using AI",
            ai_prompt="Perform comprehensive credit risk assessment based on applicant's financial profile, credit history, and loan details. Consider debt-to-income ratio, credit utilization, payment history, and employment stability.",
            config={
                "ai_type": "risk_assessment",
                "assessment_factors": [
                    "credit_score_analysis",
                    "debt_to_income_ratio",
                    "employment_stability",
                    "payment_history",
                    "loan_to_value_ratio"
                ],
                "risk_categories": ["Low", "Medium", "High"],
                "scoring_model": "comprehensive_risk_model_v2"
            },
            conditions=[
                {
                    "field": "risk_level",
                    "operator": "equals",
                    "value": "Low",
                    "next_step": "loan_pricing"
                },
                {
                    "field": "risk_level",
                    "operator": "equals",
                    "value": "Medium",
                    "next_step": "additional_verification"
                },
                {
                    "field": "risk_level",
                    "operator": "equals",
                    "value": "High",
                    "next_step": "decline_decision"
                }
            ],
            next_steps=["loan_pricing", "additional_verification", "decline_decision"],
            timeout_seconds=180
        ),
        
        # Additional Verification for Medium Risk
        WorkflowStep(
            id="additional_verification",
            name="Additional Verification",
            type=StepType.TASK,
            description="Request additional documentation or verification for medium-risk applications",
            config={
                "task_type": "verification_request",
                "verification_types": [
                    "employment_verification_call",
                    "reference_check",
                    "additional_income_documentation",
                    "collateral_evaluation"
                ],
                "verification_timeline": "3-5 business days"
            },
            next_steps=["enhanced_risk_assessment"]
        ),
        
        # Enhanced Risk Assessment
        WorkflowStep(
            id="enhanced_risk_assessment",
            name="Enhanced Risk Assessment",
            type=StepType.AI_ANALYSIS,
            description="Re-assess risk with additional verification data",
            ai_prompt="Re-evaluate credit risk based on additional verification data. Update risk assessment considering new information.",
            config={
                "ai_type": "enhanced_risk_assessment",
                "updated_factors": True,
                "verification_weight": 0.3
            },
            conditions=[
                {
                    "field": "updated_risk_level",
                    "operator": "equals",
                    "value": "Low",
                    "next_step": "loan_pricing"
                },
                {
                    "field": "updated_risk_level",
                    "operator": "not_equals",
                    "value": "Low",
                    "next_step": "underwriter_review"
                }
            ],
            next_steps=["loan_pricing", "underwriter_review"]
        ),
        
        # Loan Pricing and Terms
        WorkflowStep(
            id="loan_pricing",
            name="Loan Pricing & Terms Calculation",
            type=StepType.AI_ANALYSIS,
            description="Calculate optimal loan terms, interest rates, and conditions",
            ai_prompt="Calculate competitive and appropriate loan terms based on risk assessment, market conditions, and regulatory requirements.",
            config={
                "ai_type": "pricing_optimization",
                "pricing_factors": [
                    "risk_level",
                    "market_rates",
                    "competitive_analysis",
                    "regulatory_constraints",
                    "profit_margins"
                ],
                "term_options": {
                    "loan_terms": [12, 24, 36, 48, 60],
                    "interest_rate_range": {"min": 3.5, "max": 29.9},
                    "fee_structure": "standard"
                }
            },
            next_steps=["loan_approval"]
        ),
        
        # Loan Approval Decision
        WorkflowStep(
            id="loan_approval",
            name="Loan Approval Decision",
            type=StepType.HUMAN_APPROVAL,
            description="Final approval decision by authorized personnel",
            requires_approval=True,
            config={
                "approval_type": "loan_approval",
                "required_roles": ["senior_underwriter", "loan_manager"],
                "message": "Loan application ready for final approval decision",
                "approval_data": {
                    "recommended_amount": "variable",
                    "recommended_terms": "variable",
                    "risk_assessment": "variable",
                    "ai_confidence": "variable"
                },
                "decision_options": ["approve", "conditional_approve", "decline"]
            },
            conditions=[
                {
                    "field": "approval_decision",
                    "operator": "equals",
                    "value": "approve",
                    "next_step": "loan_documentation"
                },
                {
                    "field": "approval_decision",
                    "operator": "equals",
                    "value": "conditional_approve",
                    "next_step": "conditional_approval_processing"
                },
                {
                    "field": "approval_decision",
                    "operator": "equals",
                    "value": "decline",
                    "next_step": "decline_processing"
                }
            ],
            next_steps=["loan_documentation", "conditional_approval_processing", "decline_processing"]
        ),
        
        # Conditional Approval Processing
        WorkflowStep(
            id="conditional_approval_processing",
            name="Process Conditional Approval",
            type=StepType.TASK,
            description="Handle conditional approval with specific conditions",
            config={
                "task_type": "conditional_processing",
                "condition_types": [
                    "additional_collateral",
                    "co_signer_required",
                    "increased_down_payment",
                    "reduced_loan_amount",
                    "modified_terms"
                ]
            },
            next_steps=["customer_notification"]
        ),
        
        # Loan Documentation
        WorkflowStep(
            id="loan_documentation",
            name="Generate Loan Documentation",
            type=StepType.TASK,
            description="Generate comprehensive loan documentation and agreements",
            config={
                "task_type": "document_generation",
                "documents": [
                    "loan_agreement",
                    "promissory_note",
                    "disclosure_statements",
                    "payment_schedule",
                    "terms_and_conditions"
                ],
                "template_version": "2024_v1",
                "digital_signature_required": True
            },
            next_steps=["customer_notification"]
        ),
        
        # Decline Processing
        WorkflowStep(
            id="decline_processing",
            name="Process Loan Decline",
            type=StepType.TASK,
            description="Handle loan decline with proper notifications and compliance",
            config={
                "task_type": "decline_processing",
                "decline_reasons": "from_assessment",
                "compliance_requirements": [
                    "adverse_action_notice",
                    "credit_score_disclosure",
                    "appeal_process_information"
                ]
            },
            next_steps=["customer_notification"]
        ),
        
        # Underwriter Review (for edge cases)
        WorkflowStep(
            id="underwriter_review",
            name="Senior Underwriter Review",
            type=StepType.HUMAN_APPROVAL,
            description="Senior underwriter review for complex cases",
            requires_approval=True,
            config={
                "approval_type": "underwriter_review",
                "required_roles": ["senior_underwriter", "chief_underwriter"],
                "message": "Complex application requires senior underwriter review",
                "escalation_level": "high"
            },
            next_steps=["loan_pricing", "decline_processing"]
        ),
        
        # Customer Notification
        WorkflowStep(
            id="customer_notification",
            name="Customer Notification",
            type=StepType.TASK,
            description="Send appropriate notification to customer based on decision",
            config={
                "task_type": "customer_communication",
                "notification_types": {
                    "approval": {
                        "template": "loan_approval_notification",
                        "includes": ["loan_terms", "next_steps", "documentation_requirements"]
                    },
                    "conditional": {
                        "template": "conditional_approval_notification",
                        "includes": ["conditions", "timeline", "next_steps"]
                    },
                    "decline": {
                        "template": "loan_decline_notification",
                        "includes": ["decline_reasons", "appeal_process", "credit_report_info"]
                    }
                },
                "delivery_methods": ["email", "sms", "postal_mail"]
            },
            next_steps=["workflow_completion"]
        ),
        
        # Workflow Completion
        WorkflowStep(
            id="workflow_completion",
            name="Complete Loan Application Process",
            type=StepType.TASK,
            description="Finalize loan application workflow and update systems",
            config={
                "task_type": "workflow_finalization",
                "completion_tasks": [
                    "update_customer_record",
                    "update_loan_portfolio",
                    "generate_completion_report",
                    "archive_documents",
                    "update_compliance_records"
                ],
                "retention_policy": "7_years",
                "audit_trail_complete": True
            },
            next_steps=[]  # End of workflow
        ),
        
        # Decline Decision (from initial risk assessment)
        WorkflowStep(
            id="decline_decision",
            name="Immediate Decline Decision",
            type=StepType.TASK,
            description="Process immediate decline for high-risk applications",
            config={
                "task_type": "immediate_decline",
                "decline_reason": "high_risk_profile",
                "auto_decline": True
            },
            next_steps=["decline_processing"]
        )
    ]
    
    # Create workflow definition
    workflow_definition = WorkflowDefinition(
        id="loan_application_v1",
        name="Comprehensive Loan Application Processing",
        version="1.0.0",
        description="AI-powered loan application processing workflow with document analysis, risk assessment, and approval management",
        category="financial_services",
        steps=steps,
        start_step="receive_application",
        variables={
            "max_loan_amount": 500000,
            "min_credit_score": 500,
            "max_debt_to_income_ratio": 0.45,
            "ai_confidence_threshold": 0.8,
            "approval_timeout_hours": 72,
            "document_retention_years": 7
        },
        permissions=[
            "loan:process",
            "document:review",
            "credit:assess",
            "loan:approve"
        ],
        tags=[
            "loan_processing",
            "ai_powered",
            "financial_services",
            "credit_assessment",
            "document_analysis",
            "risk_management"
        ],
        created_by="system"
    )
    
    return workflow_definition


def get_sample_loan_application_data() -> Dict[str, Any]:
    """Get sample loan application data for testing."""
    return {
        "application_id": str(uuid.uuid4()),
        "applicant_name": "John Smith",
        "applicant_email": "john.smith@email.com",
        "applicant_phone": "+1-555-0123",
        "loan_amount": 50000,
        "loan_purpose": "home_improvement",
        "loan_term_months": 60,
        
        # Personal Information
        "date_of_birth": "1985-06-15",
        "social_security_number": "***-**-1234",  # Masked for security
        "address": {
            "street": "123 Main Street",
            "city": "Anytown",
            "state": "CA",
            "zip_code": "12345"
        },
        
        # Employment Information
        "employment_status": "employed",
        "employer_name": "Tech Solutions Inc",
        "job_title": "Software Engineer",
        "employment_tenure_months": 36,
        "annual_income": 85000,
        "monthly_income": 7083,
        
        # Financial Information
        "credit_score": 720,
        "monthly_debt_obligations": 1500,
        "monthly_rent_mortgage": 1200,
        "savings_balance": 15000,
        "checking_balance": 5000,
        
        # Loan Details
        "requested_interest_rate": 6.5,
        "down_payment": 5000,
        "collateral_type": "none",
        
        # Documents Status
        "documents_provided": [
            "identity_verification",
            "income_verification",
            "employment_verification",
            "bank_statements"
        ],
        "documents_pending": [
            "credit_report"  # Will be pulled automatically
        ],
        
        # Application Metadata
        "application_date": datetime.now(timezone.utc).isoformat(),
        "application_source": "online_portal",
        "referral_source": "direct_application"
    }


# Workflow execution helper functions
async def process_loan_application(
    application_data: Dict[str, Any],
    workflow_engine,
    ai_processor
) -> Dict[str, Any]:
    """Process a loan application through the workflow."""
    
    # Create workflow definition
    workflow_def = create_loan_application_workflow()
    
    # Create workflow instance
    context = await workflow_engine.create_workflow_instance(
        workflow_def,
        application_data,
        "system"
    )
    
    # Execute workflow
    result = await workflow_engine.execute_workflow(
        context.instance_id,
        workflow_def
    )
    
    return {
        "instance_id": context.instance_id,
        "workflow_status": result.workflow_metadata.get("status"),
        "current_step": result.current_step,
        "application_status": "processed",
        "processing_summary": {
            "steps_completed": len(result.step_history),
            "total_processing_time": (result.updated_at - result.created_at).total_seconds(),
            "ai_analyses_performed": sum(1 for step in result.step_history if step.get("step_type") == "ai_analysis"),
            "approvals_required": sum(1 for step in result.step_history if step.get("step_type") == "human_approval")
        }
    }