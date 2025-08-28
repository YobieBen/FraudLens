"""
Social engineering tactics detection analyzer.

Author: Yobie Benjamin  
Date: 2025-08-26 18:34:00 PDT
"""

import re
from typing import Any, Dict, List, Set

# Result classes defined in detector.py


class SocialEngineeringAnalyzer:
    """
    Analyzer for detecting social engineering tactics.
    
    Features:
    - Psychological manipulation detection
    - Authority and fear appeals
    - Trust exploitation patterns
    - Pretexting identification
    - Baiting and quid pro quo detection
    """
    
    def __init__(self, llm_manager: Any, feature_extractor: Any):
        """Initialize social engineering analyzer."""
        self.llm_manager = llm_manager
        self.feature_extractor = feature_extractor
        
        # Psychological triggers
        self.psychological_triggers = {
            "fear": [
                "threat", "danger", "risk", "vulnerable", "exposed",
                "breach", "hacked", "compromised", "illegal", "arrest",
                "lawsuit", "penalty", "fine", "suspended", "terminated",
                "security breach", "data loss", "prevent", "serious consequences"
            ],
            "greed": [
                "free", "prize", "winner", "cash", "money", "profit",
                "bonus", "reward", "gift", "lottery", "jackpot",
                "opportunity", "exclusive", "limited offer"
            ],
            "curiosity": [
                "secret", "confidential", "exclusive", "leaked",
                "shocking", "unbelievable", "you won't believe",
                "find out", "discover", "reveal"
            ],
            "urgency": [
                "immediately", "urgent", "now", "today", "expires",
                "deadline", "last chance", "final", "hurry",
                "act fast", "don't wait", "time sensitive"
            ],
            "authority": [
                "official", "government", "police", "irs", "fbi",
                "court", "legal", "mandatory", "required",
                "compliance", "regulation", "authorized"
            ],
            "trust": [
                "trusted", "verified", "certified", "guaranteed",
                "secure", "safe", "protected", "recommended",
                "endorsed", "approved"
            ]
        }
        
        # Social engineering tactics
        self.tactics = {
            "pretexting": [
                "verify your identity", "confirm your account",
                "update your information", "security check",
                "routine verification", "account maintenance",
                "it support", "tech support", "it department",
                "remote access", "give us access"
            ],
            "baiting": [
                "free download", "click here", "see attachment",
                "open immediately", "exclusive access", "special offer"
            ],
            "quid_pro_quo": [
                "in exchange", "help us help you", "mutual benefit",
                "cooperation needed", "assist us", "participate"
            ],
            "tailgating": [
                "follow this link", "continue here", "proceed to",
                "next step", "complete process", "finish setup"
            ],
            "watering_hole": [
                "popular site", "trusted source", "official page",
                "verified link", "secure portal", "member area"
            ]
        }
        
    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text for social engineering tactics."""
        detected_tactics = []
        detected_triggers = []
        confidence_scores = []
        
        # Check psychological triggers
        trigger_results = self._analyze_psychological_triggers(text)
        for trigger_type, score in trigger_results.items():
            if score > 0.3:
                detected_triggers.append(trigger_type)
                confidence_scores.append(score)
        
        # Check specific tactics
        tactic_results = self._analyze_tactics(text)
        for tactic, score in tactic_results.items():
            if score > 0.3:
                detected_tactics.append(tactic)
                confidence_scores.append(score)
        
        # Check manipulation patterns
        manipulation_score = self._detect_manipulation_patterns(text)
        if manipulation_score > 0.4:
            detected_tactics.append("psychological_manipulation")
            confidence_scores.append(manipulation_score)
        
        # Check for pretexting
        if self._detect_pretexting(text):
            detected_tactics.append("pretexting")
            confidence_scores.append(0.6)
        
        # Use LLM for advanced analysis
        if detected_tactics or detected_triggers:
            llm_result = await self.llm_manager.analyze_fraud(text, "social_engineering")
            if llm_result.get("detected"):
                llm_confidence = llm_result.get("confidence", 0.5)
                confidence_scores.append(llm_confidence)
                llm_tactics = llm_result.get("tactics", [])
                detected_tactics.extend([t for t in llm_tactics if t not in detected_tactics])
        
        # Calculate overall results
        detected = len(detected_tactics) > 0 or len(detected_triggers) > 0
        confidence = max(confidence_scores) if confidence_scores else 0.0
        
        # Determine risk level
        if confidence > 0.7:
            risk_level = "high"
        elif confidence > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "detected": detected,
            "confidence": confidence,
            "tactics": detected_tactics[:5],
            "psychological_triggers": detected_triggers[:5],
            "risk_level": risk_level,
        }
    
    def _analyze_psychological_triggers(self, text: str) -> Dict[str, float]:
        """Analyze psychological triggers in text."""
        text_lower = text.lower()
        results = {}
        
        for trigger_type, keywords in self.psychological_triggers.items():
            score = 0.0
            matches = 0
            
            for keyword in keywords:
                if keyword in text_lower:
                    matches += 1
                    score += 0.15
            
            if matches > 0:
                # Boost score if multiple keywords from same category
                if matches >= 3:
                    score *= 1.5
                results[trigger_type] = min(score, 1.0)
        
        return results
    
    def _analyze_tactics(self, text: str) -> Dict[str, float]:
        """Analyze specific social engineering tactics."""
        text_lower = text.lower()
        results = {}
        
        for tactic, phrases in self.tactics.items():
            score = 0.0
            matches = 0
            
            for phrase in phrases:
                if phrase in text_lower:
                    matches += 1
                    score += 0.25
            
            if matches > 0:
                results[tactic] = min(score, 1.0)
        
        return results
    
    def _detect_manipulation_patterns(self, text: str) -> float:
        """Detect manipulation language patterns."""
        manipulation_patterns = [
            # Reciprocity
            (r'since you|because you|after you', 0.1),
            (r'return the favor|pay it forward', 0.2),
            
            # Social proof
            (r'everyone is|most people|others have', 0.15),
            (r'join \d+|millions of users', 0.15),
            
            # Commitment
            (r'you agreed|you promised|you committed', 0.2),
            (r'as discussed|per our conversation', 0.15),
            
            # Scarcity
            (r'only \d+ left|limited quantity|while supplies', 0.2),
            (r'exclusive|rare|unique opportunity', 0.15),
            
            # Authority
            (r'expert|professional|specialist', 0.1),
            (r'certified|licensed|authorized', 0.15),
            
            # Liking
            (r'friend|buddy|dear|valued', 0.1),
            (r'we care|we understand|we appreciate', 0.1),
        ]
        
        score = 0.0
        text_lower = text.lower()
        
        for pattern, weight in manipulation_patterns:
            if re.search(pattern, text_lower):
                score += weight
        
        return min(score, 1.0)
    
    def _detect_pretexting(self, text: str) -> bool:
        """Detect pretexting attempts."""
        pretext_indicators = [
            r'calling from|representing|on behalf of',
            r'IT department|tech support|security team',
            r'verify your|confirm your|update your',
            r'routine check|security audit|compliance review',
            r'your account shows|our records indicate',
            r'help desk|customer service|support team',
        ]
        
        text_lower = text.lower()
        matches = 0
        
        for pattern in pretext_indicators:
            if re.search(pattern, text_lower):
                matches += 1
        
        return matches >= 2  # Multiple indicators suggest pretexting