"""
FraudLens SDK models for requests and responses.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, HttpUrl


class AnalysisType(str, Enum):
    """Type of analysis to perform."""
    TEXT = "text"
    EMAIL = "email"
    DOCUMENT = "document"
    IMAGE = "image"
    TRANSACTION = "transaction"


class AnalysisStatus(str, Enum):
    """Status of an analysis."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EvidenceType(str, Enum):
    """Type of fraud evidence."""
    SUSPICIOUS_LINK = "suspicious_link"
    FAKE_SENDER = "fake_sender"
    URGENCY_LANGUAGE = "urgency_language"
    DEEPFAKE_DETECTED = "deepfake_detected"
    DOCUMENT_FORGERY = "document_forgery"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    KNOWN_PATTERN = "known_pattern"


class Evidence(BaseModel):
    """Evidence of fraud."""
    type: EvidenceType
    description: str
    severity: Literal["low", "medium", "high", "critical"]
    confidence: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalyzeTextRequest(BaseModel):
    """Request to analyze text content."""
    content: str = Field(description="Text content to analyze")
    context: str | None = Field(None, description="Additional context")
    metadata: dict[str, Any] = Field(default_factory=dict)
    webhook_url: HttpUrl | None = None


class AnalyzeEmailRequest(BaseModel):
    """Request to analyze email."""
    sender: EmailStr
    recipient: EmailStr | None = None
    subject: str
    body: str
    headers: dict[str, str] = Field(default_factory=dict)
    attachments: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    webhook_url: HttpUrl | None = None


class AnalyzeDocumentRequest(BaseModel):
    """Request to analyze document."""
    document_url: HttpUrl | None = None
    document_content: str | None = None
    document_type: str | None = Field(None, description="MIME type or extension")
    metadata: dict[str, Any] = Field(default_factory=dict)
    webhook_url: HttpUrl | None = None


class AnalyzeImageRequest(BaseModel):
    """Request to analyze image."""
    image_url: HttpUrl | None = None
    image_data: str | None = Field(None, description="Base64 encoded image")
    metadata: dict[str, Any] = Field(default_factory=dict)
    webhook_url: HttpUrl | None = None


class AnalyzeTransactionRequest(BaseModel):
    """Request to analyze transaction."""
    user_id: str
    transaction_id: str
    amount: float
    currency: str = "USD"
    merchant: str | None = None
    location: dict[str, Any] = Field(default_factory=dict)
    device_info: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    webhook_url: HttpUrl | None = None


class AnalysisResult(BaseModel):
    """Result of fraud analysis."""
    id: str = Field(description="Unique analysis ID")
    status: AnalysisStatus
    fraud_detected: bool
    fraud_score: float = Field(ge=0, le=1, description="Fraud probability score")
    fraud_types: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, description="Confidence in the result")
    reasoning: str = Field(description="Explanation of the analysis")
    evidence: list[Evidence] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    processing_time_ms: int | None = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    def is_fraud(self, threshold: float = 0.7) -> bool:
        """Check if content is fraudulent above given threshold."""
        return self.fraud_detected and self.fraud_score >= threshold
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if analysis has high confidence."""
        return self.confidence >= threshold
    
    def has_critical_evidence(self) -> bool:
        """Check if any critical evidence was found."""
        return any(e.severity == "critical" for e in self.evidence)


class BatchAnalysisRequest(BaseModel):
    """Request for batch analysis."""
    items: list[dict[str, Any]]
    analysis_type: AnalysisType
    concurrency: int = Field(default=5, ge=1, le=20)
    on_error: Literal["fail_fast", "continue", "retry"] = "continue"
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchAnalysisResult(BaseModel):
    """Result of batch analysis."""
    id: str
    total_items: int
    completed: int
    failed: int
    results: list[AnalysisResult | None]
    errors: list[dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: int | None = None
    created_at: datetime


class StreamEvent(BaseModel):
    """Event from streaming analysis."""
    type: Literal[
        "started",
        "agent_started",
        "agent_completed",
        "evidence_found",
        "progress",
        "complete",
        "error"
    ]
    timestamp: datetime
    data: dict[str, Any] = Field(default_factory=dict)
    agent_id: str | None = None
    progress: float | None = Field(None, ge=0, le=1)
    result: AnalysisResult | None = None
    error: str | None = None
