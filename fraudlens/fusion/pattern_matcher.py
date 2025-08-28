"""
Fraud pattern matching and library system.

Author: Yobie Benjamin
Date: 2025
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger


class PatternType(Enum):
    """Types of fraud patterns."""
    TEXT_REGEX = "text_regex"
    IMAGE_TEMPLATE = "image_template"
    BEHAVIORAL = "behavioral"
    SEQUENCE = "sequence"
    NETWORK = "network"
    COMPOSITE = "composite"


@dataclass
class Pattern:
    """Fraud pattern definition."""
    pattern_id: str
    name: str
    type: PatternType
    description: str
    pattern_data: Dict[str, Any]
    confidence_threshold: float = 0.7
    severity: str = "medium"  # low, medium, high, critical
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_matched: Optional[datetime] = None
    match_count: int = 0
    false_positive_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "pattern_data": self.pattern_data,
            "confidence_threshold": self.confidence_threshold,
            "severity": self.severity,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "last_matched": self.last_matched.isoformat() if self.last_matched else None,
            "match_count": self.match_count,
            "false_positive_count": self.false_positive_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        """Create from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            name=data["name"],
            type=PatternType(data["type"]),
            description=data["description"],
            pattern_data=data["pattern_data"],
            confidence_threshold=data.get("confidence_threshold", 0.7),
            severity=data.get("severity", "medium"),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            last_matched=datetime.fromisoformat(data["last_matched"]) if data.get("last_matched") else None,
            match_count=data.get("match_count", 0),
            false_positive_count=data.get("false_positive_count", 0),
        )


@dataclass
class Match:
    """Pattern match result."""
    pattern: Pattern
    confidence: float
    location: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern.pattern_id,
            "pattern_name": self.pattern.name,
            "confidence": self.confidence,
            "location": self.location,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
        }


class PatternLibrary:
    """
    Library of fraud patterns.
    
    Manages a collection of fraud patterns with CRUD operations
    and performance tracking.
    """
    
    def __init__(self, library_path: Optional[Path] = None):
        """
        Initialize pattern library.
        
        Args:
            library_path: Path to pattern library file
        """
        self.library_path = library_path or Path("fraud_patterns.json")
        self.patterns: Dict[str, Pattern] = {}
        self.pattern_index: Dict[str, Set[str]] = {}  # tag -> pattern_ids
        
        # Load existing patterns
        self.load_library()
        
        # Initialize default patterns if empty
        if not self.patterns:
            self._initialize_default_patterns()
        
        logger.info(f"PatternLibrary initialized with {len(self.patterns)} patterns")
    
    def load_library(self) -> None:
        """Load patterns from file."""
        if self.library_path.exists():
            try:
                with open(self.library_path, "r") as f:
                    data = json.load(f)
                    for pattern_data in data.get("patterns", []):
                        pattern = Pattern.from_dict(pattern_data)
                        self.add_pattern(pattern)
                logger.info(f"Loaded {len(self.patterns)} patterns from {self.library_path}")
            except Exception as e:
                logger.error(f"Failed to load pattern library: {e}")
    
    def save_library(self) -> None:
        """Save patterns to file."""
        try:
            data = {
                "patterns": [p.to_dict() for p in self.patterns.values()],
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "pattern_count": len(self.patterns),
                },
            }
            with open(self.library_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.patterns)} patterns to {self.library_path}")
        except Exception as e:
            logger.error(f"Failed to save pattern library: {e}")
    
    def add_pattern(self, pattern: Pattern) -> None:
        """Add pattern to library."""
        self.patterns[pattern.pattern_id] = pattern
        
        # Update index
        for tag in pattern.tags:
            if tag not in self.pattern_index:
                self.pattern_index[tag] = set()
            self.pattern_index[tag].add(pattern.pattern_id)
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove pattern from library."""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            
            # Update index
            for tag in pattern.tags:
                if tag in self.pattern_index:
                    self.pattern_index[tag].discard(pattern_id)
            
            del self.patterns[pattern_id]
            return True
        return False
    
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get pattern by ID."""
        return self.patterns.get(pattern_id)
    
    def search_patterns(
        self,
        tags: Optional[List[str]] = None,
        type: Optional[PatternType] = None,
        severity: Optional[str] = None,
    ) -> List[Pattern]:
        """Search patterns by criteria."""
        results = []
        
        # Start with all patterns
        candidates = set(self.patterns.keys())
        
        # Filter by tags
        if tags:
            tag_matches = set()
            for tag in tags:
                tag_matches.update(self.pattern_index.get(tag, set()))
            candidates &= tag_matches
        
        # Filter by type and severity
        for pattern_id in candidates:
            pattern = self.patterns[pattern_id]
            
            if type and pattern.type != type:
                continue
            if severity and pattern.severity != severity:
                continue
            
            results.append(pattern)
        
        return results
    
    def update_pattern_stats(
        self,
        pattern_id: str,
        matched: bool,
        false_positive: bool = False,
    ) -> None:
        """Update pattern statistics."""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            
            if matched:
                pattern.match_count += 1
                pattern.last_matched = datetime.now()
            
            if false_positive:
                pattern.false_positive_count += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pattern performance statistics."""
        stats = {
            "total_patterns": len(self.patterns),
            "patterns_by_type": {},
            "patterns_by_severity": {},
            "most_matched": [],
            "high_false_positive": [],
        }
        
        # Count by type and severity
        for pattern in self.patterns.values():
            type_key = pattern.type.value
            stats["patterns_by_type"][type_key] = stats["patterns_by_type"].get(type_key, 0) + 1
            stats["patterns_by_severity"][pattern.severity] = stats["patterns_by_severity"].get(pattern.severity, 0) + 1
        
        # Find most matched patterns
        sorted_by_matches = sorted(self.patterns.values(), key=lambda p: p.match_count, reverse=True)
        stats["most_matched"] = [
            {"id": p.pattern_id, "name": p.name, "matches": p.match_count}
            for p in sorted_by_matches[:5]
        ]
        
        # Find high false positive patterns
        high_fp = [p for p in self.patterns.values() if p.false_positive_count > p.match_count * 0.3]
        stats["high_false_positive"] = [
            {"id": p.pattern_id, "name": p.name, "fp_rate": p.false_positive_count / max(p.match_count, 1)}
            for p in high_fp
        ]
        
        return stats
    
    def _initialize_default_patterns(self) -> None:
        """Initialize with default fraud patterns."""
        default_patterns = [
            # Text patterns
            Pattern(
                pattern_id="txt_phish_001",
                name="Urgent Payment Request",
                type=PatternType.TEXT_REGEX,
                description="Detects urgent payment request language",
                pattern_data={
                    "regex": r"(?i)(urgent|immediate|act now).*(payment|transfer|wire)",
                    "min_matches": 1,
                },
                confidence_threshold=0.8,
                severity="high",
                tags=["phishing", "text", "financial"],
            ),
            Pattern(
                pattern_id="txt_phish_002",
                name="Account Suspension Threat",
                type=PatternType.TEXT_REGEX,
                description="Detects account suspension threats",
                pattern_data={
                    "regex": r"(?i)(account|service).*(suspend|terminat|clos|lock|restrict)",
                    "min_matches": 1,
                },
                confidence_threshold=0.7,
                severity="medium",
                tags=["phishing", "text", "threat"],
            ),
            Pattern(
                pattern_id="txt_typo_001",
                name="Domain Typosquatting",
                type=PatternType.TEXT_REGEX,
                description="Detects typosquatted domains",
                pattern_data={
                    "regex": r"(?i)(goog1e|arnazon|microsofy|app1e|payp4l)\.com",
                    "min_matches": 1,
                },
                confidence_threshold=0.9,
                severity="critical",
                tags=["phishing", "text", "domain"],
            ),
            
            # Behavioral patterns
            Pattern(
                pattern_id="beh_ml_001",
                name="Rapid Sequential Transfers",
                type=PatternType.BEHAVIORAL,
                description="Multiple transfers in rapid succession",
                pattern_data={
                    "behavior": "rapid_transfers",
                    "threshold": 5,
                    "time_window_minutes": 10,
                },
                confidence_threshold=0.75,
                severity="high",
                tags=["money_laundering", "behavioral", "transaction"],
            ),
            Pattern(
                pattern_id="beh_ml_002",
                name="Structuring Pattern",
                type=PatternType.BEHAVIORAL,
                description="Transactions structured to avoid reporting",
                pattern_data={
                    "behavior": "structuring",
                    "amount_threshold": 9000,
                    "count_threshold": 3,
                },
                confidence_threshold=0.8,
                severity="high",
                tags=["money_laundering", "behavioral", "structuring"],
            ),
            
            # Sequence patterns
            Pattern(
                pattern_id="seq_fraud_001",
                name="Account Takeover Sequence",
                type=PatternType.SEQUENCE,
                description="Typical account takeover sequence",
                pattern_data={
                    "sequence": [
                        "password_reset",
                        "email_change",
                        "large_withdrawal",
                    ],
                    "max_time_between_steps_hours": 24,
                },
                confidence_threshold=0.85,
                severity="critical",
                tags=["account_takeover", "sequence", "identity"],
            ),
            
            # Network patterns
            Pattern(
                pattern_id="net_fraud_001",
                name="Fraud Ring Network",
                type=PatternType.NETWORK,
                description="Connected accounts in fraud ring",
                pattern_data={
                    "min_connections": 3,
                    "shared_attributes": ["ip_address", "device_id", "payment_method"],
                },
                confidence_threshold=0.7,
                severity="high",
                tags=["fraud_ring", "network", "organized"],
            ),
        ]
        
        for pattern in default_patterns:
            self.add_pattern(pattern)


class FraudPatternMatcher:
    """
    Matches fraud patterns against multi-modal data.
    """
    
    def __init__(self, library: Optional[PatternLibrary] = None):
        """
        Initialize pattern matcher.
        
        Args:
            library: Pattern library to use
        """
        self.library = library or PatternLibrary()
        self._compiled_regexes = {}
        self._match_cache = {}
        
        # Compile regex patterns
        self._compile_text_patterns()
        
        logger.info("FraudPatternMatcher initialized")
    
    def _compile_text_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        for pattern in self.library.patterns.values():
            if pattern.type == PatternType.TEXT_REGEX:
                regex_str = pattern.pattern_data.get("regex")
                if regex_str:
                    try:
                        self._compiled_regexes[pattern.pattern_id] = re.compile(regex_str)
                    except re.error as e:
                        logger.error(f"Failed to compile regex for {pattern.pattern_id}: {e}")
    
    async def match_patterns(
        self,
        data: Dict[str, Any],
        pattern_types: Optional[List[PatternType]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Match]:
        """
        Match patterns against multi-modal data.
        
        Args:
            data: Multi-modal data dictionary
            pattern_types: Types of patterns to match
            tags: Pattern tags to filter by
            
        Returns:
            List of pattern matches
        """
        matches = []
        
        # Get relevant patterns
        if tags:
            patterns = self.library.search_patterns(tags=tags)
        else:
            patterns = list(self.library.patterns.values())
        
        # Filter by type if specified
        if pattern_types:
            patterns = [p for p in patterns if p.type in pattern_types]
        
        # Match each pattern
        for pattern in patterns:
            match = await self._match_single_pattern(pattern, data)
            if match and match.confidence >= pattern.confidence_threshold:
                matches.append(match)
                # Update pattern statistics
                self.library.update_pattern_stats(pattern.pattern_id, matched=True)
        
        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        return matches
    
    async def _match_single_pattern(
        self,
        pattern: Pattern,
        data: Dict[str, Any],
    ) -> Optional[Match]:
        """Match single pattern against data."""
        if pattern.type == PatternType.TEXT_REGEX:
            return self._match_text_regex(pattern, data)
        elif pattern.type == PatternType.BEHAVIORAL:
            return self._match_behavioral(pattern, data)
        elif pattern.type == PatternType.SEQUENCE:
            return self._match_sequence(pattern, data)
        elif pattern.type == PatternType.NETWORK:
            return self._match_network(pattern, data)
        elif pattern.type == PatternType.COMPOSITE:
            return await self._match_composite(pattern, data)
        else:
            return None
    
    def _match_text_regex(self, pattern: Pattern, data: Dict[str, Any]) -> Optional[Match]:
        """Match text regex pattern."""
        text = data.get("text", "")
        if not text:
            return None
        
        regex = self._compiled_regexes.get(pattern.pattern_id)
        if not regex:
            return None
        
        matches = regex.findall(text)
        min_matches = pattern.pattern_data.get("min_matches", 1)
        
        if len(matches) >= min_matches:
            confidence = min(1.0, len(matches) / max(min_matches, 1))
            
            return Match(
                pattern=pattern,
                confidence=confidence,
                location="text",
                evidence={
                    "matches": matches[:5],  # Limit to first 5
                    "match_count": len(matches),
                },
            )
        
        return None
    
    def _match_behavioral(self, pattern: Pattern, data: Dict[str, Any]) -> Optional[Match]:
        """Match behavioral pattern."""
        behavior_type = pattern.pattern_data.get("behavior")
        
        if behavior_type == "rapid_transfers":
            transactions = data.get("transactions", [])
            if not transactions:
                return None
            
            threshold = pattern.pattern_data.get("threshold", 5)
            time_window = pattern.pattern_data.get("time_window_minutes", 10)
            
            # Check for rapid transfers
            rapid_count = 0
            for i in range(len(transactions) - 1):
                time_diff = transactions[i+1].get("timestamp", 0) - transactions[i].get("timestamp", 0)
                if time_diff < time_window * 60:  # Convert to seconds
                    rapid_count += 1
            
            if rapid_count >= threshold:
                confidence = min(1.0, rapid_count / threshold)
                return Match(
                    pattern=pattern,
                    confidence=confidence,
                    location="transactions",
                    evidence={"rapid_count": rapid_count},
                )
        
        elif behavior_type == "structuring":
            transactions = data.get("transactions", [])
            amount_threshold = pattern.pattern_data.get("amount_threshold", 9000)
            count_threshold = pattern.pattern_data.get("count_threshold", 3)
            
            structured_count = sum(
                1 for t in transactions
                if amount_threshold * 0.9 <= t.get("amount", 0) < amount_threshold * 1.1
            )
            
            if structured_count >= count_threshold:
                confidence = min(1.0, structured_count / count_threshold)
                return Match(
                    pattern=pattern,
                    confidence=confidence,
                    location="transactions",
                    evidence={"structured_count": structured_count},
                )
        
        return None
    
    def _match_sequence(self, pattern: Pattern, data: Dict[str, Any]) -> Optional[Match]:
        """Match sequence pattern."""
        events = data.get("events", [])
        if not events:
            return None
        
        expected_sequence = pattern.pattern_data.get("sequence", [])
        max_time_between = pattern.pattern_data.get("max_time_between_steps_hours", 24)
        
        # Look for sequence in events
        event_types = [e.get("type") for e in events]
        
        # Check if expected sequence exists
        seq_index = 0
        last_match_time = None
        matched_events = []
        
        for event in events:
            if seq_index < len(expected_sequence):
                if event.get("type") == expected_sequence[seq_index]:
                    # Check time constraint
                    if last_match_time:
                        time_diff = event.get("timestamp", 0) - last_match_time
                        if time_diff > max_time_between * 3600:  # Convert to seconds
                            # Reset sequence
                            seq_index = 0
                            matched_events = []
                            continue
                    
                    matched_events.append(event)
                    last_match_time = event.get("timestamp", 0)
                    seq_index += 1
        
        if seq_index == len(expected_sequence):
            confidence = 1.0
            return Match(
                pattern=pattern,
                confidence=confidence,
                location="events",
                evidence={"matched_sequence": matched_events},
            )
        elif seq_index > 0:
            # Partial match
            confidence = seq_index / len(expected_sequence)
            if confidence >= 0.5:  # At least half the sequence matched
                return Match(
                    pattern=pattern,
                    confidence=confidence,
                    location="events",
                    evidence={
                        "partial_sequence": matched_events,
                        "completion": f"{seq_index}/{len(expected_sequence)}",
                    },
                )
        
        return None
    
    def _match_network(self, pattern: Pattern, data: Dict[str, Any]) -> Optional[Match]:
        """Match network pattern."""
        entities = data.get("entities", [])
        relationships = data.get("relationships", {})
        
        if not entities or not relationships:
            return None
        
        min_connections = pattern.pattern_data.get("min_connections", 3)
        shared_attributes = pattern.pattern_data.get("shared_attributes", [])
        
        # Find connected components
        suspicious_clusters = []
        
        for entity in entities:
            connections = relationships.get(entity.get("id", ""), [])
            
            if len(connections) >= min_connections:
                # Check for shared attributes
                cluster = [entity]
                for conn_id in connections:
                    conn_entity = next((e for e in entities if e.get("id") == conn_id), None)
                    if conn_entity:
                        cluster.append(conn_entity)
                
                # Check shared attributes
                shared_count = 0
                for attr in shared_attributes:
                    values = [e.get(attr) for e in cluster if e.get(attr)]
                    if values and len(set(values)) < len(values) * 0.5:  # More than half share value
                        shared_count += 1
                
                if shared_count > 0:
                    suspicious_clusters.append({
                        "center": entity.get("id"),
                        "size": len(cluster),
                        "shared_attributes": shared_count,
                    })
        
        if suspicious_clusters:
            confidence = min(1.0, len(suspicious_clusters) * 0.3)
            return Match(
                pattern=pattern,
                confidence=confidence,
                location="network",
                evidence={"suspicious_clusters": suspicious_clusters[:3]},  # Limit to 3
            )
        
        return None
    
    async def _match_composite(self, pattern: Pattern, data: Dict[str, Any]) -> Optional[Match]:
        """Match composite pattern (combination of multiple patterns)."""
        sub_patterns = pattern.pattern_data.get("sub_patterns", [])
        require_all = pattern.pattern_data.get("require_all", False)
        
        sub_matches = []
        for sub_pattern_id in sub_patterns:
            sub_pattern = self.library.get_pattern(sub_pattern_id)
            if sub_pattern:
                match = await self._match_single_pattern(sub_pattern, data)
                if match:
                    sub_matches.append(match)
        
        if require_all:
            if len(sub_matches) == len(sub_patterns):
                confidence = np.mean([m.confidence for m in sub_matches])
                return Match(
                    pattern=pattern,
                    confidence=confidence,
                    location="composite",
                    evidence={
                        "sub_matches": [m.to_dict() for m in sub_matches],
                    },
                )
        else:
            if sub_matches:
                confidence = np.mean([m.confidence for m in sub_matches])
                return Match(
                    pattern=pattern,
                    confidence=confidence,
                    location="composite",
                    evidence={
                        "sub_matches": [m.to_dict() for m in sub_matches],
                        "match_rate": f"{len(sub_matches)}/{len(sub_patterns)}",
                    },
                )
        
        return None
    
    def add_pattern(self, pattern: Pattern) -> None:
        """Add new pattern to library."""
        self.library.add_pattern(pattern)
        
        # Compile if text pattern
        if pattern.type == PatternType.TEXT_REGEX:
            regex_str = pattern.pattern_data.get("regex")
            if regex_str:
                try:
                    self._compiled_regexes[pattern.pattern_id] = re.compile(regex_str)
                except re.error as e:
                    logger.error(f"Failed to compile regex for {pattern.pattern_id}: {e}")
    
    def evaluate_pattern_performance(
        self,
        pattern_id: str,
        time_period_days: int = 30,
    ) -> Dict[str, Any]:
        """Evaluate pattern performance."""
        pattern = self.library.get_pattern(pattern_id)
        if not pattern:
            return {}
        
        total_matches = pattern.match_count
        false_positives = pattern.false_positive_count
        
        if total_matches > 0:
            precision = 1.0 - (false_positives / total_matches)
        else:
            precision = 0.0
        
        return {
            "pattern_id": pattern_id,
            "pattern_name": pattern.name,
            "total_matches": total_matches,
            "false_positives": false_positives,
            "precision": precision,
            "last_matched": pattern.last_matched.isoformat() if pattern.last_matched else None,
            "severity": pattern.severity,
            "confidence_threshold": pattern.confidence_threshold,
        }