"""
Analyzer protocol for specialized fraud analysis.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from typing import Any, Protocol

from pydantic import BaseModel


class AnalysisResult(BaseModel):
    """Base model for analysis results."""
    
    detected: bool
    confidence: float
    indicators: list[str]
    metadata: dict[str, Any] = {}


class AnalyzerProtocol(Protocol):
    """
    Protocol for specialized fraud analyzers.
    
    Analyzers focus on specific fraud types (phishing, document forgery, etc.)
    and provide detailed analysis within their domain.
    """
    
    analyzer_id: str
    """Unique identifier for this analyzer."""
    
    fraud_type: str
    """Type of fraud this analyzer specializes in."""
    
    async def analyze(
        self,
        input_data: Any,
        context: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        """
        Analyze input for specific fraud type.
        
        Args:
            input_data: Data to analyze
            context: Optional context information
        
        Returns:
            Analysis result with fraud indicators
        """
        ...
    
    async def batch_analyze(
        self,
        inputs: list[Any],
        context: dict[str, Any] | None = None,
    ) -> list[AnalysisResult]:
        """
        Analyze multiple inputs in batch.
        
        Args:
            inputs: List of inputs to analyze
            context: Optional context information
        
        Returns:
            List of analysis results
        """
        ...
    
    def get_confidence_threshold(self) -> float:
        """
        Get the confidence threshold for this analyzer.
        
        Returns:
            Threshold value (0.0 to 1.0)
        """
        ...
    
    def supports_streaming(self) -> bool:
        """
        Check if analyzer supports streaming analysis.
        
        Returns:
            True if streaming is supported
        """
        ...
