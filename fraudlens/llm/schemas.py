"""
Pydantic schemas for structured LLM outputs.

These schemas ensure LLMs return consistent, validated responses
that can be directly used in the fraud detection pipeline.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FraudType(str, Enum):
    """Types of fraud that can be detected."""
    
    PHISHING = "phishing"
    IDENTITY_THEFT = "identity_theft"
    DOCUMENT_FORGERY = "document_forgery"
    DEEPFAKE = "deepfake"
    SOCIAL_ENGINEERING = "social_engineering"
    FINANCIAL_FRAUD = "financial_fraud"
    SCAM = "scam"
    BRAND_IMPERSONATION = "brand_impersonation"
    ACCOUNT_TAKEOVER = "account_takeover"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Severity levels for fraud."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    """Recommended action types."""
    
    BLOCK = "block"
    BLOCK_USER = "block_user"
    QUARANTINE = "quarantine"
    FLAG_FOR_REVIEW = "flag_for_review"
    NOTIFY_USER = "notify_user"
    LOG_INCIDENT = "log_incident"
    ESCALATE = "escalate"
    ALLOW_WITH_WARNING = "allow_with_warning"
    MONITOR = "monitor"


class EvidenceType(str, Enum):
    """Types of evidence that can be collected."""
    
    URL_ANALYSIS = "url_analysis"
    DOMAIN_ANALYSIS = "domain_analysis"
    CONTENT_ANALYSIS = "content_analysis"
    LINGUISTIC_ANALYSIS = "linguistic_analysis"
    PATTERN_MATCH = "pattern_match"
    THREAT_INTELLIGENCE = "threat_intelligence"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    IMAGE_ANALYSIS = "image_analysis"
    METADATA_ANALYSIS = "metadata_analysis"
    REPUTATION_CHECK = "reputation_check"
    ANOMALY_DETECTION = "anomaly_detection"


class Evidence(BaseModel):
    """
    Evidence supporting fraud detection.
    
    Each piece of evidence contributes to the overall fraud score
    and provides explainability.
    """
    
    type: EvidenceType = Field(..., description="Type of evidence")
    description: str = Field(..., description="Human-readable description of the evidence")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in this evidence (0-1)")
    source: str = Field(..., description="Source of evidence (e.g., 'url_analyzer', 'deepfake_detector')")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional evidence metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "suspicious_url",
                "description": "URL uses typosquatting to mimic PayPal domain",
                "confidence": 0.95,
                "source": "url_analyzer",
                "metadata": {"actual_domain": "paypa1.com", "legitimate_domain": "paypal.com"}
            }
        }


class Action(BaseModel):
    """
    Recommended action based on fraud analysis.
    """
    
    type: ActionType = Field(..., description="Type of action to take")
    priority: int = Field(..., ge=1, le=5, description="Priority level (1=lowest, 5=highest)")
    description: str = Field(..., description="Description of the recommended action")
    rationale: str = Field(default="", description="Why this action is recommended")
    automated: bool = Field(default=False, description="Whether this action can be automated")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "block",
                "priority": 5,
                "description": "Block this email and quarantine for analysis",
                "rationale": "High confidence phishing attempt with malicious URL",
                "automated": True
            }
        }


class ReasoningStep(BaseModel):
    """
    A step in the reasoning process.
    
    Captures chain-of-thought reasoning for explainability.
    """
    
    step_number: int = Field(..., ge=1, description="Step number in reasoning sequence")
    observation: str = Field(..., description="What was observed in this step")
    thought: str = Field(..., description="Reasoning about the observation")
    action_taken: str | None = Field(None, description="Action taken based on thought")
    result: str | None = Field(None, description="Result of the action")
    confidence_change: float = Field(0.0, ge=-1, le=1, description="Change in confidence (-1 to 1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "step_number": 1,
                "observation": "Email contains urgent language about account suspension",
                "thought": "Urgency is a common phishing tactic to bypass critical thinking",
                "action_taken": "Analyzed urgency score using linguistic patterns",
                "result": "Urgency score: 0.87/1.0",
                "confidence_change": 0.2
            }
        }


class ToolCall(BaseModel):
    """
    Request to call a tool during analysis.
    
    Used in multi-step reasoning where the LLM needs to invoke
    external tools to gather more information.
    """
    
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: dict[str, Any] = Field(..., description="Arguments to pass to the tool")
    reasoning: str = Field(..., description="Why this tool is being called")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tool_name": "check_url_reputation",
                "arguments": {"url": "https://suspicious-site.com"},
                "reasoning": "Need to verify URL reputation in threat intelligence databases"
            }
        }


class FraudAnalysisOutput(BaseModel):
    """
    Structured output from LLM-based fraud analysis.
    
    This is the main output format for all fraud detection operations,
    ensuring consistency and enabling downstream processing.
    """
    
    fraud_detected: bool = Field(..., description="Whether fraud was detected")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence in the analysis (0-1)")
    fraud_types: list[FraudType] = Field(..., description="Types of fraud detected")
    severity: Severity = Field(..., description="Overall severity of the fraud")
    
    fraud_score: float = Field(..., ge=0, le=1, description="Normalized fraud score (0-1)")
    
    reasoning: str = Field(..., description="High-level explanation of the analysis")
    reasoning_steps: list[ReasoningStep] = Field(
        default_factory=list,
        description="Detailed chain-of-thought reasoning"
    )
    
    evidence: list[Evidence] = Field(default_factory=list, description="Evidence supporting the conclusion")
    
    recommended_actions: list[Action] = Field(default_factory=list, description="Recommended actions to take")
    
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool calls made during analysis"
    )
    
    requires_human_review: bool = Field(
        default=False,
        description="Whether human review is recommended"
    )
    
    tools_used: list[str] = Field(
        default_factory=list,
        description="Tools that were used during analysis"
    )
    
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the analysis"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the analysis was performed"
    )
    
    def to_simple_result(self) -> dict[str, Any]:
        """
        Convert to simple result format for backward compatibility.
        
        Returns:
            Dictionary with essential fraud detection information
        """
        return {
            "is_fraud": self.fraud_detected,
            "fraud_score": self.fraud_score,
            "confidence": self.confidence,
            "fraud_types": [ft.value for ft in self.fraud_types],
            "severity": self.severity.value,
            "explanation": self.reasoning,
            "recommended_actions": [
                {"type": a.type.value, "description": a.description}
                for a in self.recommended_actions
            ],
        }


class AgentMessage(BaseModel):
    """
    Message passed between agents.
    
    Used for inter-agent communication and coordination.
    """
    
    sender: str = Field(..., description="Agent sending the message")
    recipient: str = Field(..., description="Agent receiving the message")
    message_type: str = Field(..., description="Type of message")
    content: dict[str, Any] = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When message was sent")
    
    class Config:
        json_schema_extra = {
            "example": {
                "fraud_detected": True,
                "confidence": 0.92,
                "fraud_types": ["phishing", "brand_impersonation"],
                "severity": "high",
                "fraud_score": 0.89,
                "reasoning": "Email attempts to impersonate PayPal using typosquatting and urgency tactics",
                "reasoning_steps": [
                    {
                        "step_number": 1,
                        "observation": "Sender domain is paypa1.com instead of paypal.com",
                        "thought": "Character substitution (L->1) is classic typosquatting",
                        "confidence_change": 0.4
                    }
                ],
                "evidence": [
                    {
                        "type": "typosquatting",
                        "description": "Domain uses '1' instead of 'l' to mimic PayPal",
                        "confidence": 0.95,
                        "source": "domain_analyzer"
                    }
                ],
                "recommended_actions": [
                    {
                        "type": "block",
                        "priority": 5,
                        "description": "Block email and quarantine",
                        "rationale": "High confidence phishing attack",
                        "automated": True
                    }
                ],
                "requires_human_review": False,
                "tools_used": ["domain_analyzer", "url_checker"],
                "metadata": {"processing_time_ms": 145}
            }
        }


class AgentMessage(BaseModel):
    """
    Message passed between agents in multi-agent system.
    """
    
    from_agent: str = Field(..., description="ID of sending agent")
    to_agent: str | None = Field(None, description="ID of recipient agent (None for broadcast)")
    message_type: str = Field(..., description="Type of message (e.g., 'analysis_request', 'result')")
    content: dict[str, Any] = Field(..., description="Message content")
    priority: int = Field(default=1, ge=1, le=5, description="Message priority")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "from_agent": "coordinator",
                "to_agent": "phishing_agent",
                "message_type": "analysis_request",
                "content": {"input": "Suspicious email text...", "context": {}},
                "priority": 3
            }
        }
