"""
Visual fraud analyzers for different fraud types.
"""

from .deepfake_detector import DeepfakeDetector
from .document_forgery import DocumentForgeryDetector
from .logo_impersonation import LogoImpersonationDetector
from .manipulation_detector import ManipulationDetector
from .qr_code_analyzer import QRCodeAnalyzer

__all__ = [
    "DeepfakeDetector",
    "DocumentForgeryDetector",
    "LogoImpersonationDetector",
    "ManipulationDetector",
    "QRCodeAnalyzer",
]
