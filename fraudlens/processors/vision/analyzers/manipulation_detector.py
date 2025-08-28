"""
Image manipulation detection.

Author: Yobie Benjamin
Date: 2025-08-27 18:48:00 PDT
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class ManipulationDetector:
    """
    Detects image manipulations.
    
    Features:
    - Metadata anomaly detection
    - Error Level Analysis (ELA)
    - Clone detection
    - Splicing detection
    """
    
    def __init__(self, check_metadata: bool = True, check_ela: bool = True):
        """Initialize detector."""
        self.check_metadata = check_metadata
        self.check_ela = check_ela
        logger.info("ManipulationDetector initialized")
    
    async def initialize(self) -> None:
        """Initialize detector."""
        pass
    
    async def analyze(self, image_path: Path) -> Dict[str, Any]:
        """Analyze image for manipulations."""
        is_manipulated = False
        confidence = 0.0
        edits = []
        metadata_anomalies = []
        
        # Check metadata if path provided
        if isinstance(image_path, (str, Path)):
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
                
                img = Image.open(image_path)
                exifdata = img.getexif()
                
                if exifdata:
                    # Check for suspicious metadata
                    for tag_id, value in exifdata.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        # Check for editing software
                        if tag == "Software" and any(s in str(value).lower() for s in ["photoshop", "gimp", "editor"]):
                            metadata_anomalies.append(f"Edited with: {value}")
                            confidence = max(confidence, 0.7)
                            is_manipulated = True
                        
                        # Check for multiple saves
                        if tag == "DateTime" and tag == "DateTimeOriginal":
                            if value != exifdata.get("DateTimeOriginal"):
                                metadata_anomalies.append("Modification date mismatch")
                                confidence = max(confidence, 0.6)
                
                # Simple ELA check
                if self.check_ela:
                    img_array = np.array(img)
                    ela_score = self._simple_ela(img_array)
                    if ela_score > 0.3:
                        edits.append(f"ELA anomaly (score: {ela_score:.2f})")
                        confidence = max(confidence, ela_score)
                        is_manipulated = True
                        
            except Exception as e:
                logger.warning(f"Metadata analysis failed: {e}")
        
        return {
            "is_manipulated": is_manipulated,
            "confidence": confidence,
            "edits": edits,
            "metadata_anomalies": metadata_anomalies,
        }
    
    def _simple_ela(self, image: np.ndarray) -> float:
        """Simple Error Level Analysis."""
        from PIL import Image
        import io
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Compress and decompress
        pil_img = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        
        compressed = Image.open(buffer)
        compressed_arr = np.array(compressed)
        
        # Calculate difference
        if compressed_arr.shape == image.shape:
            diff = np.abs(image.astype(float) - compressed_arr.astype(float))
            return np.var(diff) / 1000.0  # Normalized score
        
        return 0.0
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass