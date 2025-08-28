"""
Logo and brand impersonation detection.

Author: Yobie Benjamin
Date: 2025-08-27 18:48:00 PDT
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class LogoImpersonationDetector:
    """
    Detects logo/brand impersonation.
    
    Features:
    - Known brand logo detection
    - Similarity scoring
    - Typosquatting detection in logos
    """
    
    def __init__(self, brands_db_path: Optional[Path] = None):
        """Initialize detector."""
        self.brands_db_path = brands_db_path
        self.known_brands = self._load_known_brands()
        logger.info(f"LogoImpersonationDetector initialized with {len(self.known_brands)} brands")
    
    def _load_known_brands(self) -> Dict[str, Dict]:
        """Load known brand signatures."""
        # Simplified brand database
        return {
            "paypal": {"colors": [(0, 48, 135), (0, 159, 228)], "keywords": ["paypal", "payment"]},
            "amazon": {"colors": [(255, 153, 0), (35, 47, 62)], "keywords": ["amazon", "prime"]},
            "microsoft": {"colors": [(0, 120, 215), (255, 185, 0)], "keywords": ["microsoft", "windows"]},
            "google": {"colors": [(66, 133, 244), (234, 67, 53)], "keywords": ["google", "gmail"]},
            "apple": {"colors": [(0, 0, 0), (147, 147, 147)], "keywords": ["apple", "iphone"]},
            "facebook": {"colors": [(24, 119, 242)], "keywords": ["facebook", "meta"]},
            "bank of america": {"colors": [(220, 20, 60), (0, 0, 139)], "keywords": ["bank", "america"]},
        }
    
    async def initialize(self) -> None:
        """Initialize detector."""
        pass
    
    async def detect_logos(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect logo impersonation in image."""
        detected_brands = []
        impersonation_detected = False
        confidence = 0.0
        
        # Simple color-based brand detection
        for brand_name, brand_info in self.known_brands.items():
            brand_score = self._check_brand_presence(image, brand_info)
            
            # Only flag if significant presence (not just black text)
            if brand_score > 0.5:  # Raised threshold
                detected_brands.append({
                    "brand": brand_name,
                    "confidence": brand_score,
                    "suspicious": self._is_suspicious_context(image, brand_name),
                })
                
                # Require both high score AND suspicious context
                if brand_score > 0.7 and self._is_suspicious_context(image, brand_name):
                    impersonation_detected = True
                    confidence = max(confidence, brand_score)
        
        return {
            "impersonation_detected": impersonation_detected,
            "confidence": confidence,
            "detected_brands": detected_brands,
            "suspicious_elements": self._find_suspicious_elements(image, detected_brands),
        }
    
    def _check_brand_presence(self, image: np.ndarray, brand_info: Dict) -> float:
        """Check if brand colors/patterns are present."""
        score = 0.0
        
        # Check for brand colors
        for brand_color in brand_info.get("colors", []):
            color_presence = self._check_color_presence(image, brand_color)
            score = max(score, color_presence)
        
        return score
    
    def _check_color_presence(self, image: np.ndarray, target_color: tuple) -> float:
        """Check if specific color is prominently present."""
        if image.ndim != 3:
            return 0.0
        
        # Skip very common colors (black, white, gray)
        target = np.array(target_color)
        if np.all(target < 20) or np.all(target > 235):  # Skip near-black or near-white
            return 0.0
        if np.std(target) < 10:  # Skip grays (similar RGB values)
            return 0.0
            
        # Calculate color distance for each pixel
        distances = np.sqrt(np.sum((image - target) ** 2, axis=2))
        
        # Count pixels close to target color
        close_pixels = distances < 50  # Threshold
        presence_ratio = np.sum(close_pixels) / close_pixels.size
        
        return min(presence_ratio * 10, 1.0)  # Scale up and cap at 1.0
    
    def _is_suspicious_context(self, image: np.ndarray, brand_name: str) -> bool:
        """Check if brand appears in suspicious context."""
        # Simplified check - look for common phishing indicators
        # In production, would use OCR to check surrounding text
        
        # Check if image is low quality (often indicates phishing)
        # But avoid false positives on simple/uniform images
        if image.ndim == 3:
            quality_score = np.std(image) / 128.0
            # Only flag as suspicious if VERY low quality AND not uniform
            mean_val = np.mean(image)
            if quality_score < 0.1 and mean_val < 50:  # Very dark and low quality
                return True
        
        # Check for mixed brands (unusual)
        # This is simplified - in production would be more sophisticated
        return False
    
    def _find_suspicious_elements(self, image: np.ndarray, detected_brands: List[Dict]) -> List[str]:
        """Find suspicious elements related to brand impersonation."""
        suspicious = []
        
        if len(detected_brands) > 2:
            suspicious.append("Multiple brands detected (unusual)")
        
        for brand_data in detected_brands:
            if brand_data.get("suspicious"):
                suspicious.append(f"{brand_data['brand']} in suspicious context")
        
        return suspicious
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass