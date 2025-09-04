"""
Vision and Document Processing Module for FraudLens.

This module handles:
- Image fraud detection (deepfakes, manipulations, forgeries)
- PDF document analysis and fraud detection
- OCR and text extraction from images/documents
- Visual feature extraction and analysis
- Document authenticity verification

Author: Yobie Benjamin
Date: 2025-08-27 18:48:00 PDT
"""

from .detector import VisionFraudDetector
from .image_preprocessor import ImagePreprocessor
from .pdf_processor import PDFProcessor

__all__ = [
    "VisionFraudDetector",
    "PDFProcessor",
    "ImagePreprocessor",
]
