"""
FraudLens SDK exceptions.

Author: Yobie Benjamin
Date: 2026-02-28
"""


class FraudLensError(Exception):
    """Base exception for all FraudLens SDK errors."""
    
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(FraudLensError):
    """Raised when SDK is misconfigured."""
    pass


class AuthenticationError(FraudLensError):
    """Raised when API key is invalid or missing."""
    pass


class RateLimitError(FraudLensError):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: int | None = None,
        details: dict | None = None
    ):
        super().__init__(message, details)
        self.retry_after = retry_after


class ValidationError(FraudLensError):
    """Raised when request data is invalid."""
    pass


class TransportError(FraudLensError):
    """Raised when transport/communication fails."""
    pass


class TimeoutError(FraudLensError):
    """Raised when request times out."""
    pass


class APIError(FraudLensError):
    """Raised when API returns an error."""
    
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: dict | None = None
    ):
        super().__init__(message, details)
        self.status_code = status_code


class AnalysisError(FraudLensError):
    """Raised when fraud analysis fails."""
    pass


class ResourceNotFoundError(FraudLensError):
    """Raised when requested resource is not found."""
    pass
