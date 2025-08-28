"""
Money laundering pattern detection analyzer.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import re
from typing import Any, Dict, List, Set


class MoneyLaunderingAnalyzer:
    """Analyzer for detecting money laundering patterns in text."""
    
    def __init__(self, llm_manager: Any, feature_extractor: Any):
        """Initialize money laundering analyzer."""
        self.llm_manager = llm_manager
        self.feature_extractor = feature_extractor
        
        # Money laundering indicators
        self.ml_keywords = {
            "structuring": ["split", "multiple transfers", "below threshold", "avoid reporting"],
            "layering": ["offshore", "shell company", "complex transaction", "multiple accounts"],
            "integration": ["legitimate business", "real estate", "investment", "clean money"],
            "crypto": ["bitcoin", "cryptocurrency", "wallet", "mixer", "tumbler", "exchange"],
        }
        
        # Suspicious transaction patterns
        self.suspicious_patterns = [
            r'\$9,\d{3}',  # Just below $10k reporting threshold
            r'wire transfer',
            r'cash deposit',
            r'nominee|straw man',
            r'smurfing',
            r'hawala',
        ]
        
        # High-risk jurisdictions
        self.high_risk_countries = [
            "cayman", "panama", "switzerland", "cyprus", "malta",
            "seychelles", "bahamas", "virgin islands", "luxembourg",
        ]
        
    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text for money laundering indicators."""
        patterns_detected = []
        risk_indicators = []
        confidence_scores = []
        
        # Check for ML keywords
        keyword_results = self._check_ml_keywords(text)
        if keyword_results:
            patterns_detected.extend(keyword_results)
            confidence_scores.append(0.4 * len(keyword_results))
        
        # Check for suspicious patterns
        pattern_results = self._check_suspicious_patterns(text)
        if pattern_results:
            risk_indicators.extend(pattern_results)
            confidence_scores.append(0.5)
        
        # Check for high-risk jurisdictions
        jurisdiction_results = self._check_jurisdictions(text)
        if jurisdiction_results:
            risk_indicators.append(f"High-risk jurisdictions mentioned: {', '.join(jurisdiction_results)}")
            confidence_scores.append(0.3)
        
        # Check for structuring patterns
        if self._detect_structuring(text):
            patterns_detected.append("structuring")
            confidence_scores.append(0.6)
        
        # Check for crypto-related patterns (often used for layering)
        if self._detect_crypto_ml(text):
            patterns_detected.append("layering")
            confidence_scores.append(0.5)
        
        # Check for rapid movement patterns
        if self._detect_rapid_movement(text):
            patterns_detected.append("rapid_movement")
            confidence_scores.append(0.6)
        
        # Use LLM for advanced analysis
        if patterns_detected or risk_indicators:
            llm_result = await self.llm_manager.analyze_fraud(text[:1500], "money_laundering")
            if llm_result.get("detected"):
                confidence_scores.append(llm_result.get("confidence", 0.5))
        
        # Calculate overall results
        detected = len(patterns_detected) > 0 or len(risk_indicators) > 0
        confidence = max(confidence_scores) if confidence_scores else 0.0
        
        return {
            "detected": detected,
            "confidence": min(confidence, 1.0),
            "patterns": patterns_detected[:5],
            "risk_indicators": risk_indicators[:5],
        }
    
    def _check_ml_keywords(self, text: str) -> List[str]:
        """Check for money laundering keywords."""
        text_lower = text.lower()
        detected = []
        
        for category, keywords in self.ml_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected.append(f"{category}: {keyword}")
        
        return detected
    
    def _check_suspicious_patterns(self, text: str) -> List[str]:
        """Check for suspicious transaction patterns."""
        detected = []
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(f"Pattern: {pattern}")
        
        return detected
    
    def _check_jurisdictions(self, text: str) -> List[str]:
        """Check for high-risk jurisdiction mentions."""
        text_lower = text.lower()
        mentioned = []
        
        for country in self.high_risk_countries:
            if country in text_lower:
                mentioned.append(country.title())
        
        return mentioned
    
    def _detect_structuring(self, text: str) -> bool:
        """Detect potential structuring patterns."""
        text_lower = text.lower()
        
        # Look for split transfers or rapid sequential transfers
        if "split transfer" in text_lower or "multiple transfer" in text_lower:
            return True
        
        # Look for multiple transactions just below reporting threshold
        amounts = re.findall(r'\$[\d,]+\.?\d{0,2}', text)
        
        if len(amounts) > 2:
            below_threshold = 0
            same_amounts = {}
            for amount in amounts:
                value = float(amount.replace('$', '').replace(',', ''))
                # Check for structuring below $10,000
                if 9000 <= value < 10000:
                    below_threshold += 1
                # Check for identical or similar amounts (splitting)
                rounded = round(value, -3)  # Round to nearest 1000
                same_amounts[rounded] = same_amounts.get(rounded, 0) + 1
            
            # Detect if multiple similar amounts (indicating splitting)
            if any(count >= 2 for count in same_amounts.values()):
                return True
            
            return below_threshold >= 2
        
        return False
    
    def _detect_crypto_ml(self, text: str) -> bool:
        """Detect cryptocurrency money laundering patterns."""
        text_lower = text.lower()
        
        # Check for crypto mentions which often indicate layering
        if "crypto" in text_lower or "btc" in text_lower or "bitcoin" in text_lower:
            return True
        
        crypto_ml_indicators = [
            "mixer", "tumbler", "coinjoin", "privacy coin",
            "anonymous wallet", "dark web", "no kyc",
        ]
        
        for indicator in crypto_ml_indicators:
            if indicator in text_lower:
                return True
        
        return False
    
    def _detect_rapid_movement(self, text: str) -> bool:
        """Detect rapid movement of funds."""
        text_lower = text.lower()
        
        # Look for immediate or rapid transfers
        rapid_indicators = [
            "immediate transfer", "rapid transfer", "same day",
            "within minutes", "within hours", "immediately",
        ]
        
        for indicator in rapid_indicators:
            if indicator in text_lower:
                return True
        
        # Check for timestamps showing rapid succession
        import re
        timestamps = re.findall(r'\d{1,2}:\d{2}:\d{2}', text)
        if len(timestamps) >= 3:
            # Multiple transactions in short time indicates rapid movement
            return True
        
        return False