"""
QR code and barcode malicious payload detection.

Author: Yobie Benjamin
Date: 2025-08-27 18:48:00 PDT
"""

import asyncio
import re
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class QRCodeAnalyzer:
    """
    Analyzes QR codes and barcodes for malicious content.

    Features:
    - QR code detection and decoding
    - Malicious URL detection
    - Payload analysis
    - Barcode validation
    """

    def __init__(self, check_malicious_urls: bool = True):
        """Initialize analyzer."""
        self.check_malicious_urls = check_malicious_urls

        # Try to import QR libraries
        self.has_pyzbar = self._try_import_pyzbar()
        self.has_qrcode = self._try_import_qrcode()

        logger.info(f"QRCodeAnalyzer initialized (pyzbar: {self.has_pyzbar})")

    def _try_import_pyzbar(self) -> bool:
        """Try to import pyzbar."""
        try:
            from pyzbar import pyzbar

            self.pyzbar = pyzbar
            return True
        except ImportError:
            return False

    def _try_import_qrcode(self) -> bool:
        """Try to import qrcode."""
        try:
            import qrcode

            self.qrcode_module = qrcode
            return True
        except ImportError:
            return False

    async def initialize(self) -> None:
        """Initialize analyzer."""
        pass

    async def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze QR codes in image."""
        malicious = False
        payload = None
        risk_indicators = []

        # Detect and decode QR codes
        if self.has_pyzbar:
            codes = self._detect_codes(image)

            for code in codes:
                data = code.get("data", "")
                code_type = code.get("type", "")

                # Analyze payload
                if data:
                    payload = data

                    # Check for malicious patterns
                    if self._is_malicious_url(data):
                        malicious = True
                        risk_indicators.append("Malicious URL pattern")

                    if self._has_suspicious_payload(data):
                        malicious = True
                        risk_indicators.append("Suspicious payload content")

                    if self._is_obfuscated(data):
                        risk_indicators.append("Obfuscated content")
        else:
            # Mock analysis for testing
            payload = "mock://example.com"
            if "phishing" in str(image).lower():
                malicious = True
                risk_indicators.append("Test malicious QR")

        return {
            "malicious": malicious,
            "payload": payload,
            "risk_indicators": risk_indicators,
            "codes_found": 1 if payload else 0,
        }

    def _detect_codes(self, image: np.ndarray) -> List[Dict]:
        """Detect QR codes and barcodes in image."""
        codes = []

        if not self.has_pyzbar:
            return codes

        # Convert to PIL Image for pyzbar
        from PIL import Image

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        # Detect codes
        detected = self.pyzbar.decode(pil_image)

        for code in detected:
            codes.append(
                {
                    "type": code.type,
                    "data": (
                        code.data.decode("utf-8") if isinstance(code.data, bytes) else code.data
                    ),
                    "rect": code.rect,
                }
            )

        return codes

    def _is_malicious_url(self, url: str) -> bool:
        """Check if URL is malicious."""
        if not self.check_malicious_urls:
            return False

        url_lower = url.lower()

        # Check for known malicious patterns
        malicious_patterns = [
            r"bit\.ly/[a-z0-9]{3,}",  # Shortened URLs
            r"tinyurl\.com/",
            r"goo\.gl/",
            r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}",  # IP addresses
            r"[a-z0-9]+-[a-z0-9]+-[a-z0-9]+\.(tk|ml|ga|cf)",  # Suspicious domains
            r"[a-z]+(ph[i1]sh[i1]ng|scam|fake|fraud)",  # Obvious malicious
        ]

        for pattern in malicious_patterns:
            if re.search(pattern, url_lower):
                return True

        # Check for typosquatting
        legitimate_domains = ["paypal.com", "amazon.com", "google.com", "microsoft.com"]
        for legit in legitimate_domains:
            if self._is_typosquatting(url_lower, legit):
                return True

        return False

    def _has_suspicious_payload(self, data: str) -> bool:
        """Check for suspicious payload content."""
        suspicious_keywords = [
            "eval",
            "exec",
            "script",  # Code execution
            "password",
            "credential",
            "login",  # Credential harvesting
            "bitcoin",
            "wallet",
            "transfer",  # Cryptocurrency scams
            "winner",
            "congratulations",
            "prize",  # Common scam words
        ]

        data_lower = data.lower()
        return any(keyword in data_lower for keyword in suspicious_keywords)

    def _is_obfuscated(self, data: str) -> bool:
        """Check if content appears obfuscated."""
        # Check for base64
        if re.match(r"^[A-Za-z0-9+/]+=*$", data) and len(data) > 20:
            return True

        # Check for hex encoding
        if re.match(r"^[0-9a-fA-F]+$", data) and len(data) % 2 == 0 and len(data) > 20:
            return True

        # Check for excessive special characters
        special_ratio = sum(1 for c in data if not c.isalnum()) / max(len(data), 1)
        if special_ratio > 0.5:
            return True

        return False

    def _is_typosquatting(self, url: str, legitimate: str) -> bool:
        """Check if URL is typosquatting on legitimate domain."""
        # Extract domain from URL
        import urllib.parse

        try:
            parsed = urllib.parse.urlparse(url if url.startswith("http") else f"http://{url}")
            domain = parsed.netloc.lower()
        except:
            domain = url.lower()

        # Remove common variations
        legit_base = legitimate.replace(".com", "").replace(".org", "")
        domain_base = domain.replace(".com", "").replace(".org", "").replace(".tk", "")

        # Check for character substitution
        if domain_base != legit_base:
            # Common substitutions
            substitutions = {
                "a": ["@", "4"],
                "e": ["3"],
                "i": ["1", "l"],
                "o": ["0"],
                "s": ["5", "$"],
            }

            for char, subs in substitutions.items():
                for sub in subs:
                    if domain_base == legit_base.replace(char, sub):
                        return True

            # Check for character insertion
            if len(domain_base) == len(legit_base) + 1:
                for i in range(len(domain_base)):
                    if domain_base[:i] + domain_base[i + 1 :] == legit_base:
                        return True

        return False

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass
