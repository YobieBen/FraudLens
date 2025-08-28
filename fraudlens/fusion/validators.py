"""
Cross-modal validation system for consistency checking.

Author: Yobie Benjamin
Date: 2025
"""

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image

from fraudlens.core.base.detector import DetectionResult


@dataclass
class ValidationResult:
    """Result from validation check."""
    is_valid: bool
    confidence: float
    inconsistencies: List[str]
    evidence: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "inconsistencies": self.inconsistencies,
            "evidence": self.evidence,
        }


@dataclass
class ConsistencyReport:
    """Comprehensive consistency report."""
    overall_consistency: float
    text_image_consistency: Optional[ValidationResult] = None
    audio_video_sync: Optional[ValidationResult] = None
    metadata_content_validation: Optional[ValidationResult] = None
    entity_consistency: Optional[ValidationResult] = None
    temporal_consistency: Optional[ValidationResult] = None
    inconsistency_count: int = 0
    high_risk_inconsistencies: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_consistency": self.overall_consistency,
            "text_image_consistency": self.text_image_consistency.to_dict() if self.text_image_consistency else None,
            "audio_video_sync": self.audio_video_sync.to_dict() if self.audio_video_sync else None,
            "metadata_content_validation": self.metadata_content_validation.to_dict() if self.metadata_content_validation else None,
            "entity_consistency": self.entity_consistency.to_dict() if self.entity_consistency else None,
            "temporal_consistency": self.temporal_consistency.to_dict() if self.temporal_consistency else None,
            "inconsistency_count": self.inconsistency_count,
            "high_risk_inconsistencies": self.high_risk_inconsistencies,
            "timestamp": self.timestamp.isoformat(),
        }


class CrossModalValidator:
    """
    Validates consistency across different modalities.
    
    Performs various cross-modal checks to identify inconsistencies
    that may indicate fraud or manipulation.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        sync_tolerance_ms: int = 100,
        entity_match_threshold: float = 0.8,
    ):
        """
        Initialize cross-modal validator.
        
        Args:
            similarity_threshold: Threshold for similarity checks
            sync_tolerance_ms: Tolerance for audio-video sync
            entity_match_threshold: Threshold for entity matching
        """
        self.similarity_threshold = similarity_threshold
        self.sync_tolerance_ms = sync_tolerance_ms
        self.entity_match_threshold = entity_match_threshold
        
        # Validation statistics
        self._validation_count = 0
        self._inconsistency_patterns = {}
        
        logger.info("CrossModalValidator initialized")
    
    async def validate_consistency(
        self,
        text_result: Optional[DetectionResult] = None,
        image_result: Optional[DetectionResult] = None,
        audio_result: Optional[DetectionResult] = None,
        video_result: Optional[DetectionResult] = None,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> ConsistencyReport:
        """
        Validate consistency across modalities.
        
        Args:
            text_result: Text analysis result
            image_result: Image analysis result
            audio_result: Audio analysis result
            video_result: Video analysis result
            raw_data: Raw data for detailed validation
            
        Returns:
            Comprehensive consistency report
        """
        self._validation_count += 1
        
        validation_tasks = []
        
        # Text-image consistency
        if text_result and image_result:
            validation_tasks.append(
                self._validate_text_image(text_result, image_result, raw_data)
            )
        
        # Audio-video sync
        if audio_result and video_result:
            validation_tasks.append(
                self._validate_audio_video_sync(audio_result, video_result, raw_data)
            )
        
        # Metadata-content validation
        if raw_data and raw_data.get("metadata"):
            validation_tasks.append(
                self._validate_metadata_content(raw_data)
            )
        
        # Entity consistency
        results = [r for r in [text_result, image_result, audio_result, video_result] if r]
        if len(results) > 1:
            validation_tasks.append(
                self._validate_entity_consistency(results, raw_data)
            )
        
        # Temporal consistency
        if any(r for r in [text_result, image_result, audio_result, video_result] if r):
            validation_tasks.append(
                self._validate_temporal_consistency(
                    [r for r in [text_result, image_result, audio_result, video_result] if r],
                    raw_data
                )
            )
        
        # Run validations in parallel
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process results
        report = ConsistencyReport(overall_consistency=1.0)
        consistency_scores = []
        
        # Map results to report fields based on what was actually validated
        result_idx = 0
        
        # Text-image consistency
        if text_result and image_result:
            if result_idx < len(validation_results) and not isinstance(validation_results[result_idx], Exception):
                report.text_image_consistency = validation_results[result_idx]
            result_idx += 1
        
        # Audio-video sync
        if audio_result and video_result:
            if result_idx < len(validation_results) and not isinstance(validation_results[result_idx], Exception):
                report.audio_video_sync = validation_results[result_idx]
            result_idx += 1
        
        # Metadata-content validation
        if raw_data and raw_data.get("metadata"):
            if result_idx < len(validation_results) and not isinstance(validation_results[result_idx], Exception):
                report.metadata_content_validation = validation_results[result_idx]
            result_idx += 1
        
        # Entity consistency
        if len(results) > 1:
            if result_idx < len(validation_results) and not isinstance(validation_results[result_idx], Exception):
                report.entity_consistency = validation_results[result_idx]
            result_idx += 1
        
        # Temporal consistency
        if any(r for r in [text_result, image_result, audio_result, video_result] if r):
            if result_idx < len(validation_results) and not isinstance(validation_results[result_idx], Exception):
                report.temporal_consistency = validation_results[result_idx]
            result_idx += 1
        
        # Calculate consistency scores
        for result in validation_results:
            if isinstance(result, Exception):
                logger.error(f"Validation failed: {result}")
                continue
            
            if result:
                consistency_scores.append(result.confidence if result.is_valid else 1 - result.confidence)
                if not result.is_valid:
                    report.inconsistency_count += len(result.inconsistencies)
                    # Check for high-risk inconsistencies
                    for inconsistency in result.inconsistencies:
                        if self._is_high_risk_inconsistency(inconsistency):
                            report.high_risk_inconsistencies.append(inconsistency)
        
        # Calculate overall consistency
        if consistency_scores:
            report.overall_consistency = float(np.mean(consistency_scores))
        
        return report
    
    async def _validate_text_image(
        self,
        text_result: DetectionResult,
        image_result: DetectionResult,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate consistency between text and image."""
        inconsistencies = []
        evidence = {}
        
        # Extract entities from text
        text_entities = self._extract_entities_from_text(text_result)
        
        # Extract visual elements from image
        image_entities = self._extract_entities_from_image(image_result)
        
        # Check for entity consistency
        common_entities = set(text_entities) & set(image_entities)
        text_only = set(text_entities) - set(image_entities)
        image_only = set(image_entities) - set(text_entities)
        
        if text_only:
            inconsistencies.append(f"Entities mentioned in text but not visible: {text_only}")
        if image_only and len(image_only) > 2:  # Some visual elements may not be mentioned
            inconsistencies.append(f"Visual elements not mentioned in text: {image_only}")
        
        # Check fraud type consistency
        text_fraud_types = set(text_result.fraud_types)
        image_fraud_types = set(image_result.fraud_types)
        
        if text_fraud_types and image_fraud_types:
            overlap = text_fraud_types & image_fraud_types
            if not overlap and (text_fraud_types or image_fraud_types):
                inconsistencies.append("Different fraud types detected in text vs image")
        
        # Check claims vs visual evidence
        if raw_data:
            text_claims = raw_data.get("text_claims", [])
            visual_evidence = raw_data.get("visual_evidence", [])
            
            for claim in text_claims:
                if not self._verify_claim_against_visual(claim, visual_evidence):
                    inconsistencies.append(f"Claim not supported by visual evidence: {claim}")
        
        # Calculate confidence
        if text_entities and image_entities:
            jaccard = len(common_entities) / len(set(text_entities) | set(image_entities))
            confidence = jaccard
        else:
            confidence = 0.5
        
        evidence["text_entities"] = text_entities
        evidence["image_entities"] = image_entities
        evidence["common_entities"] = list(common_entities)
        
        return ValidationResult(
            is_valid=len(inconsistencies) == 0,
            confidence=confidence,
            inconsistencies=inconsistencies,
            evidence=evidence,
        )
    
    async def _validate_audio_video_sync(
        self,
        audio_result: DetectionResult,
        video_result: DetectionResult,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate audio-video synchronization."""
        inconsistencies = []
        evidence = {}
        
        if raw_data:
            # Check timestamps
            audio_timestamps = raw_data.get("audio_timestamps", [])
            video_timestamps = raw_data.get("video_timestamps", [])
            
            if audio_timestamps and video_timestamps:
                # Simple sync check
                sync_errors = []
                for i, (a_ts, v_ts) in enumerate(zip(audio_timestamps, video_timestamps)):
                    diff = abs(a_ts - v_ts)
                    if diff > self.sync_tolerance_ms:
                        sync_errors.append((i, diff))
                
                if sync_errors:
                    inconsistencies.append(f"Audio-video sync issues at {len(sync_errors)} points")
                    evidence["sync_errors"] = sync_errors
            
            # Check for audio without corresponding video activity
            audio_peaks = raw_data.get("audio_peaks", [])
            video_motion = raw_data.get("video_motion", [])
            
            if audio_peaks and not video_motion:
                inconsistencies.append("Audio activity without video motion detected")
            
            # Check for lip-sync if applicable
            if raw_data.get("has_speech") and raw_data.get("has_face"):
                lip_sync_score = raw_data.get("lip_sync_score", 1.0)
                if lip_sync_score < 0.5:
                    inconsistencies.append(f"Poor lip-sync detected: {lip_sync_score:.2f}")
        
        # Check fraud score consistency
        score_diff = abs(audio_result.fraud_score - video_result.fraud_score)
        if score_diff > 0.3:
            inconsistencies.append(f"Large discrepancy in fraud scores: {score_diff:.2f}")
        
        confidence = 1.0 - score_diff if score_diff < 1.0 else 0.0
        
        return ValidationResult(
            is_valid=len(inconsistencies) == 0,
            confidence=confidence,
            inconsistencies=inconsistencies,
            evidence=evidence,
        )
    
    async def _validate_metadata_content(
        self,
        raw_data: Dict[str, Any],
    ) -> ValidationResult:
        """Validate document metadata against content."""
        inconsistencies = []
        evidence = {}
        
        metadata = raw_data.get("metadata", {})
        content = raw_data.get("content", "")
        
        # Check creation/modification dates
        creation_date = metadata.get("creation_date")
        modification_date = metadata.get("modification_date")
        
        if creation_date and modification_date:
            if modification_date < creation_date:
                inconsistencies.append("Modification date before creation date")
        
        # Check author consistency
        metadata_author = metadata.get("author", "")
        content_author = self._extract_author_from_content(content)
        
        if metadata_author and content_author:
            if metadata_author.lower() != content_author.lower():
                inconsistencies.append(f"Author mismatch: metadata={metadata_author}, content={content_author}")
        
        # Check file size consistency
        expected_size = metadata.get("file_size")
        actual_size = len(content.encode()) if content else 0
        
        if expected_size and actual_size:
            size_diff = abs(expected_size - actual_size)
            if size_diff > expected_size * 0.1:  # More than 10% difference
                inconsistencies.append(f"File size mismatch: expected={expected_size}, actual={actual_size}")
        
        # Check hash consistency
        metadata_hash = metadata.get("content_hash")
        if metadata_hash and content:
            actual_hash = hashlib.sha256(content.encode()).hexdigest()
            if metadata_hash != actual_hash:
                inconsistencies.append("Content hash mismatch")
        
        # Check format consistency
        metadata_format = metadata.get("format", "").lower()
        detected_format = self._detect_content_format(content)
        
        if metadata_format and detected_format:
            if metadata_format != detected_format:
                inconsistencies.append(f"Format mismatch: metadata={metadata_format}, detected={detected_format}")
        
        confidence = 1.0 - (len(inconsistencies) * 0.2)
        confidence = max(0.0, confidence)
        
        evidence["metadata"] = metadata
        evidence["content_length"] = len(content) if content else 0
        
        return ValidationResult(
            is_valid=len(inconsistencies) == 0,
            confidence=confidence,
            inconsistencies=inconsistencies,
            evidence=evidence,
        )
    
    async def _validate_entity_consistency(
        self,
        results: List[DetectionResult],
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate entity consistency across modalities."""
        inconsistencies = []
        evidence = {}
        
        # Extract entities from each result
        all_entities = []
        for result in results:
            entities = self._extract_entities_from_result(result)
            all_entities.append(set(entities))
        
        # Check for consistency
        if all_entities:
            common = set.intersection(*all_entities)
            union = set.union(*all_entities)
            
            if union:
                consistency_ratio = len(common) / len(union)
                
                if consistency_ratio < self.entity_match_threshold:
                    inconsistencies.append(f"Low entity consistency: {consistency_ratio:.2f}")
                
                # Check for conflicting entities
                for i, entities1 in enumerate(all_entities):
                    for j, entities2 in enumerate(all_entities[i+1:], i+1):
                        conflicts = self._find_conflicting_entities(entities1, entities2)
                        if conflicts:
                            inconsistencies.append(f"Conflicting entities between modalities {i} and {j}: {conflicts}")
            
            evidence["common_entities"] = list(common)
            evidence["all_entities"] = [list(e) for e in all_entities]
            confidence = consistency_ratio if union else 0.5
        else:
            confidence = 0.5
        
        return ValidationResult(
            is_valid=len(inconsistencies) == 0,
            confidence=confidence,
            inconsistencies=inconsistencies,
            evidence=evidence,
        )
    
    async def _validate_temporal_consistency(
        self,
        results: List[DetectionResult],
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate temporal consistency."""
        inconsistencies = []
        evidence = {}
        
        # Extract timestamps
        timestamps = []
        for result in results:
            if hasattr(result, "timestamp"):
                timestamps.append(result.timestamp)
        
        if raw_data:
            # Check for temporal anomalies
            content_dates = raw_data.get("content_dates", [])
            metadata_dates = raw_data.get("metadata_dates", [])
            
            if content_dates and metadata_dates:
                for c_date in content_dates:
                    for m_date in metadata_dates:
                        if c_date > m_date:
                            inconsistencies.append(f"Content date {c_date} after metadata date {m_date}")
            
            # Check for impossible temporal sequences
            events = raw_data.get("events", [])
            if events:
                sorted_events = sorted(events, key=lambda x: x.get("timestamp", 0))
                for i in range(len(sorted_events) - 1):
                    event1 = sorted_events[i]
                    event2 = sorted_events[i + 1]
                    
                    if not self._validate_event_sequence(event1, event2):
                        inconsistencies.append(f"Invalid event sequence: {event1['type']} -> {event2['type']}")
        
        # Check processing time consistency
        processing_times = [r.processing_time_ms for r in results if hasattr(r, "processing_time_ms")]
        if processing_times:
            mean_time = np.mean(processing_times)
            std_time = np.std(processing_times)
            
            for i, time in enumerate(processing_times):
                if abs(time - mean_time) > 3 * std_time:  # 3-sigma rule
                    inconsistencies.append(f"Anomalous processing time for modality {i}: {time}ms")
        
        confidence = 1.0 - (len(inconsistencies) * 0.15)
        confidence = max(0.0, confidence)
        
        evidence["timestamps"] = [ts.isoformat() if hasattr(ts, "isoformat") else str(ts) for ts in timestamps]
        evidence["processing_times"] = processing_times
        
        return ValidationResult(
            is_valid=len(inconsistencies) == 0,
            confidence=confidence,
            inconsistencies=inconsistencies,
            evidence=evidence,
        )
    
    def _extract_entities_from_text(self, result: DetectionResult) -> List[str]:
        """Extract entities from text result."""
        entities = []
        
        if result.evidence:
            # Extract from evidence
            entities.extend(result.evidence.get("entities", []))
            entities.extend(result.evidence.get("names", []))
            entities.extend(result.evidence.get("organizations", []))
        
        return entities
    
    def _extract_entities_from_image(self, result: DetectionResult) -> List[str]:
        """Extract entities from image result."""
        entities = []
        
        if result.evidence:
            # Extract from evidence
            entities.extend(result.evidence.get("objects", []))
            entities.extend(result.evidence.get("text_in_image", []))
            entities.extend(result.evidence.get("logos", []))
        
        return entities
    
    def _extract_entities_from_result(self, result: DetectionResult) -> List[str]:
        """Extract entities from any result."""
        if result.modality.value == "text":
            return self._extract_entities_from_text(result)
        elif result.modality.value in ["image", "vision"]:
            return self._extract_entities_from_image(result)
        else:
            # Generic extraction
            entities = []
            if result.evidence:
                for key in ["entities", "objects", "names", "keywords"]:
                    entities.extend(result.evidence.get(key, []))
            return entities
    
    def _verify_claim_against_visual(self, claim: str, visual_evidence: List[Any]) -> bool:
        """Verify if claim is supported by visual evidence."""
        # Simple keyword matching for now
        claim_lower = claim.lower()
        
        for evidence in visual_evidence:
            if isinstance(evidence, str):
                if evidence.lower() in claim_lower or claim_lower in evidence.lower():
                    return True
            elif isinstance(evidence, dict):
                evidence_text = str(evidence.get("description", "")).lower()
                if evidence_text and (evidence_text in claim_lower or claim_lower in evidence_text):
                    return True
        
        return False
    
    def _extract_author_from_content(self, content: str) -> Optional[str]:
        """Extract author from content."""
        # Simple pattern matching
        patterns = [
            r"Author:\s*([^\n]+)",
            r"By:\s*([^\n]+)",
            r"Written by:\s*([^\n]+)",
            r"Signed:\s*([^\n]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _detect_content_format(self, content: str) -> Optional[str]:
        """Detect content format."""
        if content.startswith("%PDF"):
            return "pdf"
        elif content.startswith("<?xml"):
            return "xml"
        elif content.startswith("{") or content.startswith("["):
            return "json"
        elif "<html" in content.lower():
            return "html"
        else:
            return "text"
    
    def _find_conflicting_entities(self, entities1: set, entities2: set) -> List[str]:
        """Find conflicting entities between sets."""
        conflicts = []
        
        # Simple conflict detection based on opposites
        opposites = {
            "legitimate": "fraudulent",
            "authentic": "fake",
            "original": "copied",
            "verified": "unverified",
        }
        
        for entity1 in entities1:
            for entity2 in entities2:
                for word1, word2 in opposites.items():
                    if word1 in entity1.lower() and word2 in entity2.lower():
                        conflicts.append(f"{entity1} vs {entity2}")
                    elif word2 in entity1.lower() and word1 in entity2.lower():
                        conflicts.append(f"{entity1} vs {entity2}")
        
        return conflicts
    
    def _validate_event_sequence(self, event1: Dict, event2: Dict) -> bool:
        """Validate if event sequence is possible."""
        # Simple validation based on event types
        invalid_sequences = [
            ("payment_completed", "payment_initiated"),
            ("account_closed", "account_opened"),
            ("delivery_completed", "order_placed"),
        ]
        
        type1 = event1.get("type", "")
        type2 = event2.get("type", "")
        
        return (type1, type2) not in invalid_sequences
    
    def _is_high_risk_inconsistency(self, inconsistency: str) -> bool:
        """Check if inconsistency is high risk."""
        high_risk_keywords = [
            "hash mismatch",
            "author mismatch",
            "signature",
            "tampering",
            "modification date",
            "conflicting entities",
            "sync issues",
        ]
        
        inconsistency_lower = inconsistency.lower()
        return any(keyword in inconsistency_lower for keyword in high_risk_keywords)