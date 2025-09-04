"""
Feature extraction for text fraud detection.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import hashlib
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import json


class FeatureExtractor:
    """
    Extract features from text for fraud detection.

    Features:
    - Named entity recognition for financial entities
    - Sentiment analysis with financial context
    - Urgency/pressure tactics detection
    - Grammatical anomaly scoring
    - Cross-reference checking
    """

    def __init__(self):
        """Initialize feature extractor."""
        # Financial entity patterns
        self.financial_entities = {
            "bank_names": [
                "chase",
                "bank of america",
                "wells fargo",
                "citibank",
                "capital one",
                "pnc",
                "us bank",
                "truist",
                "goldman sachs",
                "morgan stanley",
                "jpmorgan",
                "hsbc",
                "barclays",
            ],
            "payment_services": [
                "paypal",
                "venmo",
                "zelle",
                "cashapp",
                "stripe",
                "square",
                "western union",
                "moneygram",
                "wise",
            ],
            "credit_cards": [
                "visa",
                "mastercard",
                "amex",
                "american express",
                "discover",
                "diners club",
            ],
            "cryptocurrencies": [
                "bitcoin",
                "btc",
                "ethereum",
                "eth",
                "crypto",
                "blockchain",
                "wallet address",
                "coinbase",
                "binance",
            ],
        }

        # Financial terms
        self.financial_terms = [
            "account",
            "balance",
            "transfer",
            "transaction",
            "payment",
            "deposit",
            "withdrawal",
            "credit",
            "debit",
            "loan",
            "mortgage",
            "interest",
            "fee",
            "charge",
            "refund",
            "invoice",
            "statement",
            "routing number",
            "swift",
            "iban",
            "wire transfer",
            "ach",
            "direct deposit",
        ]

        # Urgency indicators
        self.urgency_words = [
            "urgent",
            "immediate",
            "asap",
            "now",
            "quickly",
            "hurry",
            "fast",
            "expire",
            "deadline",
            "limited",
            "final",
            "last chance",
            "act now",
            "don't wait",
        ]

        # Grammar quality indicators
        self.grammar_issues = {
            "capitalization": r"^[a-z]|[.!?]\s+[a-z]",
            "punctuation": r"\s+[,.]|[.!?]{2,}",
            "spacing": r"\s{2,}|\t",
            "spelling": r"recieve|loose|there money|you\'re account|wont",
        }

        self._initialized = False
        self._cache = {}

    async def initialize(self) -> None:
        """Initialize feature extractor."""
        self._initialized = True

    async def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract comprehensive features from text.

        Args:
            text: Input text

        Returns:
            Dictionary of extracted features
        """
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._cache:
            return self._cache[text_hash]

        features = {}

        # Basic statistics
        features.update(self._extract_basic_stats(text))

        # Financial entities
        features.update(self._extract_financial_entities(text))

        # URLs and emails
        features.update(self._extract_urls_emails(text))

        # Urgency and sentiment
        features["urgency_score"] = self._calculate_urgency_score(text)
        features["sentiment"] = self._analyze_sentiment(text)

        # Grammar quality
        features["grammar_score"] = self._calculate_grammar_score(text)

        # Financial terms
        features["financial_term_density"] = self._calculate_financial_density(text)
        features["has_financial_terms"] = (
            features["financial_term_density"] > 0.02
        )  # Lowered threshold for invoices

        # Transaction patterns
        features["has_transaction_patterns"] = self._detect_transaction_patterns(text)

        # Credential requests
        features["requests_credentials"] = self._detect_credential_requests(text)

        # Cache features
        self._cache[text_hash] = features
        if len(self._cache) > 1000:
            # Remove oldest entries
            self._cache = dict(list(self._cache.items())[-500:])

        return features

    def _extract_basic_stats(self, text: str) -> Dict[str, Any]:
        """Extract basic text statistics."""
        words = text.split()
        sentences = re.split(r"[.!?]+", text)

        return {
            "text_length": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            "digit_ratio": sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            "special_char_ratio": (
                sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
                if text
                else 0
            ),
        }

    def _extract_financial_entities(self, text: str) -> Dict[str, Any]:
        """Extract financial entities from text."""
        text_lower = text.lower()
        entities = {
            "banks_mentioned": [],
            "payment_services_mentioned": [],
            "credit_cards_mentioned": [],
            "crypto_mentioned": False,
        }

        for bank in self.financial_entities["bank_names"]:
            if bank in text_lower:
                entities["banks_mentioned"].append(bank)

        for service in self.financial_entities["payment_services"]:
            if service in text_lower:
                entities["payment_services_mentioned"].append(service)

        for card in self.financial_entities["credit_cards"]:
            if card in text_lower:
                entities["credit_cards_mentioned"].append(card)

        for crypto in self.financial_entities["cryptocurrencies"]:
            if crypto in text_lower:
                entities["crypto_mentioned"] = True
                break

        return entities

    def _extract_urls_emails(self, text: str) -> Dict[str, Any]:
        """Extract URLs and email addresses."""
        # URL pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text, re.IGNORECASE)

        # Email pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, text)

        # URL shortener pattern
        shortener_pattern = r"(?:bit\.ly|tinyurl\.com|goo\.gl|ow\.ly|short\.link)/[^\s]+"
        shortened_urls = re.findall(shortener_pattern, text, re.IGNORECASE)

        return {
            "url_count": len(urls),
            "email_count": len(emails),
            "has_urls": len(urls) > 0,
            "has_email_patterns": len(emails) > 0,
            "has_shortened_urls": len(shortened_urls) > 0,
            "urls": urls[:5],  # First 5 URLs
            "emails": emails[:3],  # First 3 emails
        }

    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency score based on keywords."""
        text_lower = text.lower()
        score = 0.0
        word_count = len(text.split())

        for word in self.urgency_words:
            occurrences = text_lower.count(word)
            score += occurrences * 0.1

        # Check for time references
        time_patterns = [
            r"\b\d+\s*(?:hours?|minutes?|days?|weeks?)\b",
            r"\btoday\b|\btomorrow\b|\bnow\b",
            r"\bexpir\w+\b|\bdeadline\b",
        ]

        for pattern in time_patterns:
            if re.search(pattern, text_lower):
                score += 0.15

        # Normalize by text length
        if word_count > 0:
            score = score / (word_count / 100)

        return min(score, 1.0)

    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment with financial context."""
        text_lower = text.lower()

        positive_words = [
            "congratulations",
            "winner",
            "prize",
            "free",
            "bonus",
            "reward",
            "refund",
            "approved",
            "eligible",
            "qualified",
        ]

        negative_words = [
            "suspended",
            "blocked",
            "locked",
            "frozen",
            "terminated",
            "expired",
            "overdue",
            "penalty",
            "fine",
            "violation",
        ]

        neutral_words = ["update", "verify", "confirm", "review", "check"]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        neutral_count = sum(1 for word in neutral_words if word in text_lower)

        total = positive_count + negative_count + neutral_count

        if total == 0:
            return "neutral"

        if positive_count > negative_count and positive_count > neutral_count:
            return "positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            return "negative"
        else:
            return "neutral"

    def _calculate_grammar_score(self, text: str) -> float:
        """Calculate grammar quality score."""
        issues = 0
        word_count = len(text.split())

        for issue_type, pattern in self.grammar_issues.items():
            matches = len(re.findall(pattern, text))
            issues += matches

        # Check for all caps
        if text.isupper():
            issues += 5

        # Check for no capitalization
        if text.islower() and len(text) > 50:
            issues += 3

        # Check for repeated punctuation
        if re.search(r"[!?]{3,}", text):
            issues += 2

        # Normalize
        if word_count > 0:
            score = issues / word_count
            return min(score, 1.0)

        return 0.0

    def _calculate_financial_density(self, text: str) -> float:
        """Calculate density of financial terms."""
        text_lower = text.lower()
        words = text_lower.split()

        if not words:
            return 0.0

        financial_count = sum(1 for word in words if word in self.financial_terms)
        return financial_count / len(words)

    def _detect_transaction_patterns(self, text: str) -> bool:
        """Detect transaction-related patterns."""
        transaction_patterns = [
            r"\$[\d,]+\.?\d*",  # Dollar amounts
            r"USD|EUR|GBP|JPY|CNY",  # Currency codes
            r"transfer|wire|deposit|withdraw",  # Transaction verbs
            r"account\s*(?:number|#)?\s*[:]*\s*\d+",  # Account numbers
            r"routing\s*(?:number|#)?\s*[:]*\s*\d+",  # Routing numbers
            r"reference\s*(?:number|#|code)",  # Reference numbers
        ]

        text_lower = text.lower()
        for pattern in transaction_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False

    def _detect_credential_requests(self, text: str) -> bool:
        """Detect requests for credentials."""
        credential_patterns = [
            r"enter your password",
            r"verify your identity",
            r"confirm your (?:pin|password|ssn)",
            r"provide your (?:account|card) (?:number|details)",
            r"click here to (?:verify|confirm|update)",
            r"security question",
        ]

        text_lower = text.lower()
        for pattern in credential_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._cache.clear()
        self._initialized = False

    def get_memory_usage(self) -> int:
        """Get estimated memory usage."""
        # Estimate cache size
        cache_size = len(json.dumps(self._cache)) if self._cache else 0
        return cache_size + 1024 * 1024  # Cache + 1MB overhead
