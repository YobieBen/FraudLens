"""
Phishing detection analyzer for email and SMS fraud.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

# Result classes defined in detector.py


class PhishingAnalyzer:
    """
    Specialized analyzer for phishing detection.

    Features:
    - URL analysis and reputation checking
    - Email pattern detection
    - Urgency and pressure tactics identification
    - Brand impersonation detection
    - Typosquatting detection
    """

    def __init__(self, llm_manager: Any, feature_extractor: Any):
        """
        Initialize phishing analyzer.

        Args:
            llm_manager: LLM manager for advanced analysis
            feature_extractor: Feature extraction utility
        """
        self.llm_manager = llm_manager
        self.feature_extractor = feature_extractor

        # Known legitimate domains
        self.legitimate_domains = {
            "paypal.com",
            "amazon.com",
            "microsoft.com",
            "google.com",
            "apple.com",
            "facebook.com",
            "twitter.com",
            "linkedin.com",
            "chase.com",
            "bankofamerica.com",
            "wellsfargo.com",
            "citi.com",
            "americanexpress.com",
            "discover.com",
            "capitalone.com",
        }

        # Phishing keywords and phrases
        self.phishing_keywords = {
            "urgent": 0.3,
            "immediate action": 0.4,
            "verify account": 0.3,
            "suspended": 0.4,
            "click here": 0.3,
            "act now": 0.3,
            "limited time": 0.2,
            "confirm identity": 0.3,
            "update payment": 0.3,
            "security alert": 0.2,
            "unusual activity": 0.2,
            "prize": 0.3,
            "winner": 0.3,
            "congratulations": 0.2,
            "tax refund": 0.4,
            "irs": 0.3,
            "invoice": 0.2,
            "receipt": 0.2,
            "package delivery": 0.2,
            "tracking": 0.2,
        }

        # Suspicious TLDs
        self.suspicious_tlds = {
            ".tk",
            ".ml",
            ".ga",
            ".cf",
            ".click",
            ".download",
            ".review",
            ".top",
            ".win",
            ".bid",
            ".club",
            ".fake",
        }

        # URL shorteners
        self.url_shorteners = {
            "bit.ly",
            "tinyurl.com",
            "goo.gl",
            "ow.ly",
            "short.link",
            "buff.ly",
            "is.gd",
            "adf.ly",
            "bc.vc",
            "cutt.ly",
        }

    async def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for phishing indicators.

        Args:
            text: Text to analyze

        Returns:
            Phishing analysis result
        """
        indicators = []
        confidence_scores = []

        # Extract URLs and check for URL-like patterns
        text_lower = text.lower()
        urls = self._extract_urls(text)
        suspicious_urls = self._analyze_urls(urls)

        # Check for common phishing phrases that suggest URLs
        phishing_phrases = [
            "click here",
            "click this link",
            "verify your account",
            "account has been compromised",
            "secure your account",
            "suspended account",
            "confirm your identity",
            "update your information",
        ]

        for phrase in phishing_phrases:
            if phrase in text_lower:
                indicators.append(f"Contains suspicious phrase: '{phrase}'")
                confidence_scores.append(0.8)
                if not suspicious_urls:
                    suspicious_urls = ["implicit_link"]
                break

        if suspicious_urls:
            indicators.append(f"Found {len(suspicious_urls)} suspicious URLs/patterns")
            confidence_scores.append(min(0.4 * len(suspicious_urls), 0.9))

        # Check for urgency and pressure tactics
        urgency_score = self._calculate_urgency_score(text)
        if urgency_score > 0.5:
            indicators.append(f"High urgency score: {urgency_score:.2f}")
            confidence_scores.append(urgency_score)

        # Check for phishing keywords
        keyword_score = self._check_phishing_keywords(text)
        if keyword_score > 0.3:
            indicators.append(f"Phishing keyword score: {keyword_score:.2f}")
            confidence_scores.append(keyword_score)

        # Check for lottery/prize scams
        lottery_words = [
            "won",
            "winner",
            "prize",
            "lottery",
            "congratulations",
            "claim",
            "million",
            "$1,000,000",
        ]
        lottery_count = sum(1 for word in lottery_words if word in text_lower)
        if lottery_count >= 2:
            indicators.append("Lottery/prize scam pattern detected")
            confidence_scores.append(0.9)

        # Check for impersonation
        impersonated_entities = self._detect_impersonation(text)
        if impersonated_entities:
            indicators.append(f"Possible impersonation of: {', '.join(impersonated_entities)}")
            confidence_scores.append(0.7)

        # Check for credential requests
        if self._has_credential_request(text):
            indicators.append("Requests sensitive information")
            confidence_scores.append(0.6)

        # Grammar and spelling analysis
        grammar_score = await self._analyze_grammar(text)
        if grammar_score > 0.5:
            indicators.append(f"Poor grammar score: {grammar_score:.2f}")
            confidence_scores.append(grammar_score * 0.5)

        # Use LLM for advanced analysis if needed
        if confidence_scores:
            llm_result = await self.llm_manager.analyze_fraud(text, "phishing")
            if llm_result.get("is_phishing"):
                llm_confidence = llm_result.get("confidence", 0.5)
                confidence_scores.append(llm_confidence)
                llm_indicators = llm_result.get("indicators", [])
                indicators.extend(llm_indicators[:3])

        # Calculate overall confidence
        is_phishing = False
        confidence = 0.0

        # If we have any indicators, be more aggressive about detection
        if indicators or suspicious_urls or impersonated_entities:
            # Even weak signals should trigger detection if multiple indicators present
            if len(indicators) > 2:
                confidence = max(0.7, max(confidence_scores) if confidence_scores else 0.7)
            elif confidence_scores:
                confidence = max(confidence_scores)
            else:
                confidence = 0.6  # Default moderate confidence if indicators exist

            is_phishing = confidence > 0.3  # Lower threshold for detection
        elif confidence_scores:
            confidence = max(confidence_scores)
            is_phishing = confidence > 0.5

        return {
            "is_phishing": is_phishing,
            "confidence": confidence,
            "indicators": indicators[:10],  # Limit to top 10
            "suspicious_urls": suspicious_urls[:5],  # Top 5 suspicious URLs
            "impersonated_entities": impersonated_entities[:3],  # Top 3 entities
            "urgency_score": urgency_score,
        }

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        # Regex for URL extraction
        url_pattern = r"https?://(?:[-\w.])+(?::\d+)?(?:[/\w\s.,?!@#$%&()=~-]*)?"
        urls = re.findall(url_pattern, text, re.IGNORECASE)

        # Also look for obfuscated URLs
        obfuscated_pattern = r"(?:hxxp|hXXp|h\*\*p)[s]?://[^\s]+"
        obfuscated = re.findall(obfuscated_pattern, text, re.IGNORECASE)
        urls.extend(
            [
                url.replace("hxxp", "http").replace("hXXp", "http").replace("h**p", "http")
                for url in obfuscated
            ]
        )

        # Look for URLs without protocol
        domain_pattern = r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b"
        domains = re.findall(domain_pattern, text)
        urls.extend(
            [f"http://{domain}" for domain in domains if domain not in self.legitimate_domains]
        )

        return list(set(urls))

    def _analyze_urls(self, urls: List[str]) -> List[str]:
        """Analyze URLs for suspicious characteristics."""
        suspicious = []

        for url in urls:
            try:
                parsed = urlparse(url.lower())
                domain = parsed.netloc

                # Check for URL shorteners
                if any(shortener in domain for shortener in self.url_shorteners):
                    suspicious.append(url)
                    continue

                # Check for suspicious TLDs
                if any(domain.endswith(tld) for tld in self.suspicious_tlds):
                    suspicious.append(url)
                    continue

                # Check for typosquatting
                for legit in self.legitimate_domains:
                    if self._is_typosquatting(domain, legit):
                        suspicious.append(url)
                        break

                # Check for suspicious patterns
                if self._has_suspicious_patterns(url):
                    suspicious.append(url)

            except:
                # Malformed URL is suspicious
                suspicious.append(url)

        return suspicious

    def _is_typosquatting(self, domain: str, legitimate: str) -> bool:
        """Check if domain is typosquatting on legitimate domain."""
        # Remove TLD for comparison
        domain_base = domain.split(".")[0]
        legit_base = legitimate.split(".")[0]

        # Check for common typosquatting patterns
        if domain_base == legit_base:
            return False

        # Character substitution
        if len(domain_base) == len(legit_base):
            diff_count = sum(1 for a, b in zip(domain_base, legit_base) if a != b)
            if diff_count <= 2:
                return True

        # Character insertion/deletion
        if abs(len(domain_base) - len(legit_base)) == 1:
            if domain_base in legit_base or legit_base in domain_base:
                return True

        # Homoglyphs (visual similarity)
        homoglyphs = {
            "o": "0",
            "i": "1",
            "l": "1",
            "e": "3",
            "a": "@",
            "s": "$",
            "g": "9",
        }
        for char, replacement in homoglyphs.items():
            if domain_base == legit_base.replace(char, replacement):
                return True

        return False

    def _has_suspicious_patterns(self, url: str) -> bool:
        """Check for suspicious URL patterns."""
        suspicious_patterns = [
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IP address
            r"@",  # @ symbol in URL
            r"-{2,}",  # Multiple dashes
            r"\.{2,}",  # Multiple dots
            r"[^\x00-\x7F]",  # Non-ASCII characters
            r"\.\./",  # Directory traversal
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, url):
                return True

        return False

    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency/pressure score."""
        text_lower = text.lower()
        urgency_phrases = [
            ("immediate", 0.3),
            ("urgent", 0.3),
            ("act now", 0.4),
            ("expires", 0.2),
            ("limited time", 0.3),
            ("final notice", 0.4),
            ("last chance", 0.3),
            ("24 hours", 0.2),
            ("today only", 0.3),
            ("asap", 0.2),
            ("quickly", 0.2),
            ("hurry", 0.2),
        ]

        score = 0.0
        for phrase, weight in urgency_phrases:
            if phrase in text_lower:
                score += weight

        return min(score, 1.0)

    def _check_phishing_keywords(self, text: str) -> float:
        """Check for phishing keywords."""
        text_lower = text.lower()
        score = 0.0

        for keyword, weight in self.phishing_keywords.items():
            if keyword in text_lower:
                score += weight

        return min(score, 1.0)

    def _detect_impersonation(self, text: str) -> List[str]:
        """Detect possible brand/organization impersonation."""
        text_lower = text.lower()
        impersonated = []

        # Check for brand mentions
        brands = [
            "paypal",
            "amazon",
            "microsoft",
            "google",
            "apple",
            "facebook",
            "twitter",
            "netflix",
            "spotify",
            "adobe",
            "chase",
            "bank of america",
            "wells fargo",
            "citibank",
            "american express",
            "discover",
            "capital one",
            "irs",
            "social security",
            "medicare",
            "fedex",
            "ups",
            "usps",
        ]

        for brand in brands:
            if brand in text_lower:
                # Check if it's in a suspicious context
                context_words = ["verify", "confirm", "update", "suspended", "locked"]
                for word in context_words:
                    if word in text_lower:
                        impersonated.append(brand.title())
                        break

        return list(set(impersonated))

    def _has_credential_request(self, text: str) -> bool:
        """Check if text requests credentials or sensitive info."""
        credential_patterns = [
            r"password",
            r"ssn|social security",
            r"credit card",
            r"bank account",
            r"pin\b",
            r"cvv",
            r"routing number",
            r"account number",
            r"date of birth|dob",
            r"mother\'s maiden",
            r"security question",
        ]

        text_lower = text.lower()
        for pattern in credential_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    async def _analyze_grammar(self, text: str) -> float:
        """Analyze grammar and spelling quality."""
        # Simple heuristic-based grammar check
        errors = 0

        # Check for common phishing grammar mistakes
        grammar_issues = [
            (r"\s+[,.]", 0.1),  # Space before punctuation
            (r"[.!?]{2,}", 0.1),  # Multiple punctuation
            (r"\b[A-Z]{2,}\b", 0.05),  # All caps words
            (r"\bi\b", 0.1),  # Lowercase "i" as pronoun
            (r"(recieve|loose|there money|you\'re account)", 0.2),  # Common misspellings
        ]

        for pattern, weight in grammar_issues:
            matches = len(re.findall(pattern, text))
            errors += matches * weight

        # Normalize by text length
        word_count = len(text.split())
        if word_count > 0:
            return min(errors / (word_count / 100), 1.0)

        return 0.0
