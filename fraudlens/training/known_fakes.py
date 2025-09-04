"""
Database of known fake documents and their signatures.

Author: Yobie Benjamin
Date: 2025-08-28
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np
from loguru import logger


class KnownFakeDatabase:
    """
    Manages database of known fake documents for immediate detection.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize known fakes database."""
        self.db_path = db_path or Path("data/known_fakes.json")
        self.known_fakes = self._load_database()

        # Known fake patterns for specific document types
        self.fake_patterns = {
            "mclovin_license": {
                "type": "driver_license",
                "identifiers": [
                    "McLovin",
                    "MCLOVIN",
                    "06/03/1981",
                    "892 MOMONA ST",
                    "HONOLULU",
                    "123 456 789",
                ],
                "description": "Superbad movie fake ID",
                "confidence": 1.0,
            },
            "sample_passport": {
                "type": "passport",
                "identifiers": ["SPECIMEN", "SAMPLE", "TEST DOCUMENT", "00000000"],
                "description": "Sample/test passport",
                "confidence": 1.0,
            },
            "john_doe_license": {
                "type": "driver_license",
                "identifiers": ["John Doe", "Jane Doe", "123 Main St", "Anytown", "12345"],
                "description": "Generic placeholder ID",
                "confidence": 0.95,
            },
            "novelty_id_markers": {
                "type": "any",
                "identifiers": [
                    "FOR NOVELTY USE",
                    "NOT A GOVERNMENT",
                    "SOUVENIR",
                    "REPLICA",
                    "PROP USE ONLY",
                    "ENTERTAINMENT PURPOSES",
                ],
                "description": "Novelty ID disclaimer",
                "confidence": 1.0,
            },
        }

        # Known fake serial number patterns
        self.fake_serials = {
            "sequential": ["123456789", "111111111", "000000000", "999999999"],
            "test": ["TEST12345", "SAMPLE123", "DEMO12345", "FAKE12345"],
            "movie_refs": ["007", "420", "666", "1337", "8675309"],
        }

        # Image hashes of known fakes
        self.fake_hashes: Set[str] = set()
        self._load_known_hashes()

        logger.info(f"KnownFakeDatabase initialized with {len(self.fake_patterns)} patterns")

    def _load_database(self) -> Dict[str, Any]:
        """Load known fakes database from file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load known fakes database: {e}")
        return {}

    def _load_known_hashes(self):
        """Load hashes of known fake document images."""
        # These would be loaded from a database in production
        # Adding some known fake document hashes
        known_fake_hashes = [
            # McLovin license hash examples (would be actual hashes)
            "d41d8cd98f00b204e9800998ecf8427e",  # Example hash
            "098f6bcd4621d373cade4e832627b4f6",  # Example hash
        ]
        self.fake_hashes.update(known_fake_hashes)

    def check_document(
        self, image: np.ndarray, text_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if document is a known fake.

        Args:
            image: Document image
            text_content: OCR extracted text (optional)

        Returns:
            Detection result with confidence and reason
        """
        result = {"is_known_fake": False, "confidence": 0.0, "reasons": [], "fake_type": None}

        # Check image hash
        image_hash = self._compute_image_hash(image)
        if image_hash in self.fake_hashes:
            result["is_known_fake"] = True
            result["confidence"] = 1.0
            result["reasons"].append("Exact match to known fake document")
            result["fake_type"] = "hash_match"
            return result

        # If we have text content, check for known patterns
        if text_content:
            text_upper = text_content.upper()

            # Check each known fake pattern
            for fake_id, pattern in self.fake_patterns.items():
                matches = 0
                for identifier in pattern["identifiers"]:
                    if identifier.upper() in text_upper:
                        matches += 1

                # If we match multiple identifiers, it's likely fake
                match_ratio = matches / len(pattern["identifiers"])
                if match_ratio > 0.5:  # More than half the identifiers match
                    result["is_known_fake"] = True
                    result["confidence"] = max(
                        result["confidence"], pattern["confidence"] * match_ratio
                    )
                    result["reasons"].append(f"{pattern['description']} detected")
                    result["fake_type"] = fake_id

            # Check for fake serial numbers
            for serial_type, serials in self.fake_serials.items():
                for serial in serials:
                    if serial in text_upper:
                        result["is_known_fake"] = True
                        result["confidence"] = max(result["confidence"], 0.9)
                        result["reasons"].append(f"Known fake serial: {serial_type}")
                        result["fake_type"] = f"fake_serial_{serial_type}"

        # Visual analysis for specific known fakes
        visual_result = self._check_visual_signatures(image)
        if visual_result["is_fake"]:
            result["is_known_fake"] = True
            result["confidence"] = max(result["confidence"], visual_result["confidence"])
            result["reasons"].extend(visual_result["reasons"])
            if not result["fake_type"]:
                result["fake_type"] = visual_result.get("fake_type", "visual_match")

        return result

    def _compute_image_hash(self, image: np.ndarray) -> str:
        """Compute hash of image for comparison."""
        # Convert to bytes and compute hash
        if len(image.shape) == 3:
            # Convert to grayscale for consistent hashing
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Resize to standard size for consistent hashing
        resized = cv2.resize(gray, (256, 256))

        # Compute hash
        return hashlib.md5(resized.tobytes()).hexdigest()

    def _check_visual_signatures(self, image: np.ndarray) -> Dict[str, Any]:
        """Check for visual signatures of known fakes."""
        result = {"is_fake": False, "confidence": 0.0, "reasons": [], "fake_type": None}

        # Check for specific visual patterns
        # McLovin license has specific visual characteristics
        if self._check_mclovin_signature(image):
            result["is_fake"] = True
            result["confidence"] = 0.95
            result["reasons"].append("McLovin (Superbad) fake ID detected")
            result["fake_type"] = "mclovin"

        # Check for low-quality printing indicators
        if self._check_low_quality_indicators(image):
            result["confidence"] = max(result["confidence"], 0.7)
            result["reasons"].append("Low-quality printing detected")

        # Check for missing security features
        if self._check_missing_security_features(image):
            result["confidence"] = max(result["confidence"], 0.6)
            result["reasons"].append("Missing expected security features")

        return result

    def _check_mclovin_signature(self, image: np.ndarray) -> bool:
        """Check for McLovin fake ID signature."""
        # Specific checks for McLovin ID
        # In production, would use template matching or feature detection

        # Check for Hawaii theme colors (simplified)
        if len(image.shape) == 3:
            # Check for predominant red/orange tones (Hawaii license)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # Define range for red/orange colors
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)

            # If significant red/orange present, could be McLovin
            red_ratio = np.sum(mask > 0) / mask.size
            if red_ratio > 0.1:  # More than 10% red/orange
                return True

        return False

    def _check_low_quality_indicators(self, image: np.ndarray) -> bool:
        """Check for low-quality printing indicators."""
        try:
            # Check image quality metrics
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # Ensure proper dtype for Laplacian
            if gray.dtype != np.uint8:
                gray = gray.astype(np.uint8)

            # Check for pixelation/low resolution
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()

            # Low variance indicates blurry/low quality
            if variance < 100:
                return True

            # Check for JPEG artifacts (simplified)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Too many or too few edges indicate issues
            if edge_density < 0.01 or edge_density > 0.5:
                return True

        except Exception as e:
            logger.debug(f"Error in quality check: {e}")
            # If we can't check quality, assume it's okay
            return False

        return False

    def _check_missing_security_features(self, image: np.ndarray) -> bool:
        """Check for missing security features."""
        # Simplified check for security features
        # Real implementation would check for:
        # - Holograms (reflective areas)
        # - UV features
        # - Microprinting
        # - Raised text

        if len(image.shape) == 3:
            # Check for holographic shimmer (simplified - check for color variance)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]

            # Real IDs have areas of high saturation (holograms)
            high_sat_ratio = np.sum(saturation > 200) / saturation.size

            # If no high saturation areas, might be missing holograms
            if high_sat_ratio < 0.01:
                return True

        return False

    def add_known_fake(self, image: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """
        Add a new known fake to the database.

        Args:
            image: Fake document image
            metadata: Information about the fake

        Returns:
            Success status
        """
        try:
            # Compute hash
            image_hash = self._compute_image_hash(image)
            self.fake_hashes.add(image_hash)

            # Store metadata
            fake_id = f"fake_{datetime.now().timestamp()}"
            self.known_fakes[fake_id] = {
                "hash": image_hash,
                "metadata": metadata,
                "added_date": datetime.now().isoformat(),
            }

            # Save database
            self._save_database()

            logger.info(f"Added new known fake: {fake_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add known fake: {e}")
            return False

    def _save_database(self):
        """Save database to file."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, "w") as f:
                json.dump(self.known_fakes, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save database: {e}")

    def update_from_feedback(self, image_hash: str, is_fake: bool, confidence: float):
        """
        Update database based on user feedback.

        Args:
            image_hash: Hash of the document
            is_fake: Whether document is fake
            confidence: Confidence level
        """
        if is_fake and confidence > 0.9:
            self.fake_hashes.add(image_hash)
            self.known_fakes[f"feedback_{image_hash}"] = {
                "hash": image_hash,
                "is_fake": is_fake,
                "confidence": confidence,
                "source": "user_feedback",
                "date": datetime.now().isoformat(),
            }
            self._save_database()
