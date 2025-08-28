"""
Type definitions for text fraud detection.

Author: Yobie Benjamin
Date: 2025-08-27 18:35:00 PDT
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FraudResult:
    """Fraud detection result."""
    
    fraud_score: float
    confidence: float
    fraud_types: List[Any]  # Will be FraudType enums
    features: Dict[str, Any]
    explanation: str
    processing_time_ms: float


@dataclass
class PhishingResult:
    """Phishing detection result."""
    
    is_phishing: bool
    confidence: float
    indicators: List[str]
    suspicious_urls: List[str]


@dataclass
class SocialEngineeringResult:
    """Social engineering detection result."""
    
    detected: bool
    confidence: float
    tactics: List[str]
    psychological_triggers: List[str]


@dataclass
class DocumentResult:
    """Financial document analysis result."""
    
    is_fraudulent: bool
    confidence: float
    anomalies: List[str]
    entity_mismatches: List[Dict[str, str]]
    financial_inconsistencies: List[str]