"""
AI Integration Module for Intelligent Workflow Processing.

This module provides AI-powered capabilities including document analysis,
decision making, risk assessment, and workflow optimization using OpenAI.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from enum import Enum
import logging

from openai import OpenAI
from pydantic import BaseModel, Field

from ..config.settings import settings


class AITaskType(str, Enum):
    """Types of AI tasks supported."""
    DOCUMENT_ANALYSIS = "document_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    DECISION_SUPPORT = "decision_support"
    DATA_EXTRACTION = "data_extraction"
    CONTENT_GENERATION = "content_generation"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    VALIDATION = "validation"


class AnalysisResult(BaseModel):
    """AI analysis result structure."""
    task_type: AITaskType
    result: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    recommendations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AIProcessor:
    """Main AI processing class using OpenAI for intelligent workflow operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.logger.warning("OpenAI API key not found. AI functionality will be limited.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        
        # AI model configuration
        self.default_model = "gpt-4o"  # Latest model as specified
        self.fallback_model = "gpt-4"
        
        # Task-specific prompts
        self.task_prompts = {
            AITaskType.DOCUMENT_ANALYSIS: self._get_document_analysis_prompt(),
            AITaskType.RISK_ASSESSMENT: self._get_risk_assessment_prompt(),
            AITaskType.DECISION_SUPPORT: self._get_decision_support_prompt(),
            AITaskType.DATA_EXTRACTION: self._get_data_extraction_prompt(),
            AITaskType.CLASSIFICATION: self._get_classification_prompt(),
            AITaskType.SUMMARIZATION: self._get_summarization_prompt(),
            AITaskType.VALIDATION: self._get_validation_prompt()
        }
    
    async def process_loan_application(
        self, 
        application_data: Dict[str, Any],
        documents: List[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """Process a complete loan application with AI analysis."""
        if not self.client:
            return self._create_fallback_result(AITaskType.RISK_ASSESSMENT, 
                                              "AI service unavailable")
        
        try:
            # Prepare comprehensive analysis prompt
            prompt = f"""
            You are an expert loan underwriter with extensive experience in risk assessment and loan processing.
            
            Please analyze the following loan application and provide a comprehensive assessment:
            
            APPLICATION DATA:
            {json.dumps(application_data, indent=2, default=str)}
            
            DOCUMENTS PROVIDED:
            {json.dumps(documents or [], indent=2, default=str) if documents else "No documents provided"}
            
            Please provide:
            1. Risk Assessment (Low/Medium/High) with specific reasons
            2. Loan Recommendation (Approve/Conditional Approve/Decline) with conditions if applicable
            3. Required Additional Documentation (if any)
            4. Suggested Interest Rate Range
            5. Key Risk Factors Identified
            6. Strengths of the Application
            7. Overall Confidence Score (0-100)
            
            Format your response as JSON with the following structure:
            {{
                "risk_level": "Low/Medium/High",
                "recommendation": "Approve/Conditional Approve/Decline",
                "confidence_score": 85,
                "interest_rate_range": "4.5% - 5.2%",
                "risk_factors": ["factor1", "factor2"],
                "strengths": ["strength1", "strength2"],
                "required_documents": ["doc1", "doc2"],
                "conditions": ["condition1", "condition2"],
                "detailed_reasoning": "Comprehensive explanation of the decision"
            }}
            """
            
            response = await self._call_openai_api(prompt, temperature=0.3)
            
            # Parse AI response
            try:
                ai_result = json.loads(response)
            except json.JSONDecodeError:
                # Fallback parsing
                ai_result = {"raw_response": response, "parsing_error": True}
            
            # Calculate confidence based on AI response
            confidence = ai_result.get("confidence_score", 75) / 100.0
            
            recommendations = []
            if ai_result.get("recommendation") == "Conditional Approve":
                recommendations.extend(ai_result.get("conditions", []))
            recommendations.extend(ai_result.get("required_documents", []))
            
            return AnalysisResult(
                task_type=AITaskType.RISK_ASSESSMENT,
                result=ai_result,
                confidence=confidence,
                reasoning=ai_result.get("detailed_reasoning", "AI analysis completed"),
                recommendations=recommendations,
                metadata={
                    "model_used": self.default_model,
                    "application_id": application_data.get("application_id"),
                    "processing_time": datetime.now(timezone.utc).isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing loan application: {str(e)}")
            return self._create_fallback_result(
                AITaskType.RISK_ASSESSMENT,
                f"Error in AI processing: {str(e)}"
            )
    
    async def analyze_document(
        self, 
        document_content: str,
        document_type: str,
        context: Dict[str, Any] = None
    ) -> AnalysisResult:
        """Analyze a document using AI."""
        if not self.client:
            return self._create_fallback_result(AITaskType.DOCUMENT_ANALYSIS, 
                                              "AI service unavailable")
        
        try:
            prompt = f"""
            Analyze the following {document_type} document and extract key information:
            
            DOCUMENT CONTENT:
            {document_content}
            
            CONTEXT:
            {json.dumps(context or {}, indent=2, default=str)}
            
            Please provide:
            1. Document Type Verification
            2. Key Information Extracted
            3. Data Quality Assessment
            4. Completeness Score (0-100)
            5. Any Red Flags or Inconsistencies
            6. Suggested Actions
            
            Format as JSON:
            {{
                "document_type_verified": true/false,
                "extracted_data": {{}},
                "quality_score": 85,
                "completeness_score": 90,
                "red_flags": [],
                "inconsistencies": [],
                "suggested_actions": [],
                "summary": "Brief summary of document contents"
            }}
            """
            
            response = await self._call_openai_api(prompt, temperature=0.2)
            
            try:
                ai_result = json.loads(response)
                confidence = (ai_result.get("quality_score", 70) + 
                            ai_result.get("completeness_score", 70)) / 200.0
            except json.JSONDecodeError:
                ai_result = {"raw_response": response, "parsing_error": True}
                confidence = 0.5
            
            return AnalysisResult(
                task_type=AITaskType.DOCUMENT_ANALYSIS,
                result=ai_result,
                confidence=confidence,
                reasoning=ai_result.get("summary", "Document analyzed"),
                recommendations=ai_result.get("suggested_actions", []),
                metadata={
                    "document_type": document_type,
                    "model_used": self.default_model
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing document: {str(e)}")
            return self._create_fallback_result(
                AITaskType.DOCUMENT_ANALYSIS,
                f"Error in document analysis: {str(e)}"
            )
    
    async def make_decision(
        self,
        decision_context: Dict[str, Any],
        options: List[str],
        criteria: List[str] = None
    ) -> AnalysisResult:
        """Make an AI-powered decision based on context and criteria."""
        if not self.client:
            return self._create_fallback_result(AITaskType.DECISION_SUPPORT, 
                                              "AI service unavailable")
        
        try:
            criteria_text = "\n".join(criteria) if criteria else "Use best business practices"
            
            prompt = f"""
            You are an expert decision-making assistant. Based on the provided context,
            make a recommendation from the available options.
            
            CONTEXT:
            {json.dumps(decision_context, indent=2, default=str)}
            
            AVAILABLE OPTIONS:
            {json.dumps(options, indent=2)}
            
            DECISION CRITERIA:
            {criteria_text}
            
            Please provide:
            1. Recommended Option
            2. Confidence Level (0-100)
            3. Reasoning for the Decision
            4. Pros and Cons of Each Option
            5. Risk Assessment
            6. Alternative Considerations
            
            Format as JSON:
            {{
                "recommended_option": "option_name",
                "confidence_level": 85,
                "reasoning": "Detailed explanation",
                "option_analysis": {{}},
                "risk_assessment": "Low/Medium/High",
                "alternatives": [],
                "considerations": []
            }}
            """
            
            response = await self._call_openai_api(prompt, temperature=0.4)
            
            try:
                ai_result = json.loads(response)
                confidence = ai_result.get("confidence_level", 70) / 100.0
            except json.JSONDecodeError:
                ai_result = {"raw_response": response, "parsing_error": True}
                confidence = 0.5
            
            return AnalysisResult(
                task_type=AITaskType.DECISION_SUPPORT,
                result=ai_result,
                confidence=confidence,
                reasoning=ai_result.get("reasoning", "AI decision completed"),
                recommendations=ai_result.get("alternatives", []),
                metadata={
                    "options_count": len(options),
                    "model_used": self.default_model
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in decision making: {str(e)}")
            return self._create_fallback_result(
                AITaskType.DECISION_SUPPORT,
                f"Error in AI decision making: {str(e)}"
            )
    
    async def extract_data(
        self,
        source_text: str,
        extraction_schema: Dict[str, Any]
    ) -> AnalysisResult:
        """Extract structured data from unstructured text."""
        if not self.client:
            return self._create_fallback_result(AITaskType.DATA_EXTRACTION, 
                                              "AI service unavailable")
        
        try:
            prompt = f"""
            Extract structured data from the following text according to the specified schema:
            
            SOURCE TEXT:
            {source_text}
            
            EXTRACTION SCHEMA:
            {json.dumps(extraction_schema, indent=2)}
            
            Please extract all relevant information that matches the schema.
            If a field is not found, use null. If you're uncertain about a value,
            indicate the confidence level.
            
            Return the extracted data in JSON format matching the schema structure.
            """
            
            response = await self._call_openai_api(prompt, temperature=0.1)
            
            try:
                ai_result = json.loads(response)
                # Calculate confidence based on how many fields were successfully extracted
                total_fields = len(extraction_schema)
                extracted_fields = sum(1 for v in ai_result.values() if v is not None)
                confidence = extracted_fields / total_fields if total_fields > 0 else 0.5
            except json.JSONDecodeError:
                ai_result = {"raw_response": response, "parsing_error": True}
                confidence = 0.3
            
            return AnalysisResult(
                task_type=AITaskType.DATA_EXTRACTION,
                result=ai_result,
                confidence=confidence,
                reasoning="Data extraction completed from source text",
                recommendations=[],
                metadata={
                    "schema_fields": len(extraction_schema),
                    "model_used": self.default_model
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in data extraction: {str(e)}")
            return self._create_fallback_result(
                AITaskType.DATA_EXTRACTION,
                f"Error in data extraction: {str(e)}"
            )
    
    async def _call_openai_api(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> str:
        """Make an API call to OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "You are an expert AI assistant specialized in business process analysis and decision making."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}")
            # Try fallback model
            try:
                response = self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=[
                        {"role": "system", "content": "You are an expert AI assistant specialized in business process analysis and decision making."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as fallback_error:
                self.logger.error(f"Fallback model also failed: {str(fallback_error)}")
                raise e
    
    def _create_fallback_result(
        self, 
        task_type: AITaskType, 
        error_message: str
    ) -> AnalysisResult:
        """Create a fallback result when AI processing fails."""
        return AnalysisResult(
            task_type=task_type,
            result={"error": error_message, "fallback": True},
            confidence=0.0,
            reasoning=f"AI processing unavailable: {error_message}",
            recommendations=["Manual review required"],
            metadata={"fallback_mode": True}
        )
    
    def _get_document_analysis_prompt(self) -> str:
        """Get document analysis prompt template."""
        return """
        Analyze the provided document for completeness, accuracy, and relevant information extraction.
        Focus on identifying key data points, potential issues, and compliance with expected formats.
        """
    
    def _get_risk_assessment_prompt(self) -> str:
        """Get risk assessment prompt template."""
        return """
        Conduct a comprehensive risk assessment based on the provided data.
        Consider financial, operational, and compliance risks. Provide clear recommendations.
        """
    
    def _get_decision_support_prompt(self) -> str:
        """Get decision support prompt template."""
        return """
        Provide intelligent decision support based on available options and criteria.
        Consider all relevant factors and provide clear reasoning for recommendations.
        """
    
    def _get_data_extraction_prompt(self) -> str:
        """Get data extraction prompt template."""
        return """
        Extract structured data from unstructured text according to the specified schema.
        Ensure accuracy and completeness while maintaining data integrity.
        """
    
    def _get_classification_prompt(self) -> str:
        """Get classification prompt template."""
        return """
        Classify the provided content into appropriate categories based on the given criteria.
        Provide confidence scores and reasoning for classifications.
        """
    
    def _get_summarization_prompt(self) -> str:
        """Get summarization prompt template."""
        return """
        Create concise, accurate summaries that capture the key points and essential information.
        Focus on actionable insights and important details.
        """
    
    def _get_validation_prompt(self) -> str:
        """Get validation prompt template."""
        return """
        Validate the provided data for accuracy, completeness, and consistency.
        Identify any anomalies, missing information, or potential issues.
        """
    
    def get_ai_capabilities(self) -> Dict[str, Any]:
        """Get current AI processing capabilities."""
        return {
            "available": self.client is not None,
            "models": [self.default_model, self.fallback_model],
            "supported_tasks": [task.value for task in AITaskType],
            "status": "operational" if self.client else "unavailable"
        }


# Global AI processor instance
ai_processor = AIProcessor()