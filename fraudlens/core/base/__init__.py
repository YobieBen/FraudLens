"""
Base classes for FraudLens components.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

from fraudlens.core.base.detector import FraudDetector
from fraudlens.core.base.processor import ModalityProcessor
from fraudlens.core.base.scorer import RiskScorer

__all__ = ["FraudDetector", "ModalityProcessor", "RiskScorer"]
