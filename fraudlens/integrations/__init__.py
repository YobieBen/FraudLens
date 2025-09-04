"""
External database integrations for enhanced fraud detection.

This module provides connectors to publicly available fraud detection databases,
threat intelligence feeds, and identity verification services.
"""

from .document_validator import DocumentValidator
from .phishing_db import PhishingDatabaseConnector
from .threat_intel import ThreatIntelligenceManager

__all__ = ["ThreatIntelligenceManager", "DocumentValidator", "PhishingDatabaseConnector"]
