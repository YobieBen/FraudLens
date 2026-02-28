"""
FraudLens Client SDK.

Provides both local and remote fraud detection capabilities.

Author: Yobie Benjamin
Date: 2026-02-28

Example Usage:
    ```python
    from fraudlens import FraudLens
    
    # Local mode (self-hosted)
    client = FraudLens.local(model_path="./models")
    
    # Remote mode (SaaS)
    client = FraudLens.remote(api_key="fl_live_...")
    
    # Analyze content
    result = await client.analyze.email(
        sender="suspect@example.com",
        subject="You won!",
        body="Click here..."
    )
    
    if result.is_fraud(threshold=0.8):
        print(f"Fraud detected: {result.fraud_types}")
    ```
"""

from fraudlens.client.exceptions import (
    AnalysisError,
    APIError,
    AuthenticationError,
    ConfigurationError,
    FraudLensError,
    RateLimitError,
    ResourceNotFoundError,
    TimeoutError,
    TransportError,
    ValidationError,
)
from fraudlens.client.models import (
    AnalysisResult,
    AnalysisStatus,
    AnalysisType,
    AnalyzeDocumentRequest,
    AnalyzeEmailRequest,
    AnalyzeImageRequest,
    AnalyzeTextRequest,
    AnalyzeTransactionRequest,
    BatchAnalysisRequest,
    BatchAnalysisResult,
    Evidence,
    EvidenceType,
    StreamEvent,
)
from fraudlens.client.retry import RetryConfig

__all__ = [
    # Exceptions
    "FraudLensError",
    "ConfigurationError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "TransportError",
    "TimeoutError",
    "APIError",
    "AnalysisError",
    "ResourceNotFoundError",
    # Models
    "AnalysisResult",
    "AnalysisStatus",
    "AnalysisType",
    "Evidence",
    "EvidenceType",
    "AnalyzeTextRequest",
    "AnalyzeEmailRequest",
    "AnalyzeDocumentRequest",
    "AnalyzeImageRequest",
    "AnalyzeTransactionRequest",
    "BatchAnalysisRequest",
    "BatchAnalysisResult",
    "StreamEvent",
    # Config
    "RetryConfig",
]

__version__ = "0.1.0"
