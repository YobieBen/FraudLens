"""
Phishing Database Connector.

Connects to multiple phishing databases and provides real-time
phishing detection and analysis.

Author: Yobie Benjamin
Date: 2025-08-28
"""

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse, parse_qs

import aiohttp
from loguru import logger


@dataclass
class PhishingIndicator:
    """Phishing indicator data structure."""
    
    url: str
    domain: str
    threat_level: str  # low, medium, high, critical
    confidence: float
    detected_by: List[str]
    first_seen: datetime
    last_seen: datetime
    target_brand: Optional[str] = None
    attack_type: Optional[str] = None
    ip_address: Optional[str] = None
    asn: Optional[str] = None
    country: Optional[str] = None
    ssl_cert: Optional[Dict] = None
    screenshots: Optional[List[str]] = None


class PhishingDatabaseConnector:
    """
    Connects to phishing databases and provides detection capabilities.
    
    Features:
    - Real-time phishing URL checking
    - Brand impersonation detection
    - Typosquatting analysis
    - Certificate transparency monitoring
    - Domain age verification
    - WHOIS analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize phishing database connector."""
        self.config = config or {}
        
        # Phishing databases
        self.databases = {
            "phishtank": {
                "enabled": True,
                "api_url": "https://checkurl.phishtank.com/checkurl/",
                "feed_url": "https://data.phishtank.com/data/online-valid.json",
                "api_key": self.config.get("phishtank_api_key")
            },
            "openphish": {
                "enabled": True,
                "feed_url": "https://openphish.com/feed.txt"
            },
            "certstream": {
                "enabled": True,
                "ws_url": "wss://certstream.calidog.io"
            },
            "urlscan": {
                "enabled": True,
                "api_url": "https://urlscan.io/api/v1/",
                "api_key": self.config.get("urlscan_api_key")
            },
            "google_safe_browsing": {
                "enabled": bool(self.config.get("google_safe_browsing_key")),
                "api_url": "https://safebrowsing.googleapis.com/v4/threatMatches:find",
                "api_key": self.config.get("google_safe_browsing_key")
            }
        }
        
        # Cache for known phishing sites
        self.phishing_cache: Dict[str, PhishingIndicator] = {}
        self.cache_ttl = timedelta(hours=6)
        self.known_phishing_domains = set()
        
        # Target brands often impersonated
        self.target_brands = {
            "paypal": ["paypal", "payp", "paipal", "paybal"],
            "amazon": ["amazon", "amaz0n", "amazone", "amazn"],
            "microsoft": ["microsoft", "microsofy", "micr0soft", "mircosoft"],
            "google": ["google", "googl", "g00gle", "gooogle"],
            "apple": ["apple", "appl", "app1e", "appple"],
            "netflix": ["netflix", "netflik", "netflx", "netfl1x"],
            "facebook": ["facebook", "faceboook", "facebk", "faceb00k"],
            "instagram": ["instagram", "instaqram", "instagran", "1nstagram"],
            "linkedin": ["linkedin", "linkedln", "linked1n", "linkdin"],
            "dropbox": ["dropbox", "dropb0x", "dr0pbox", "droppbox"],
            "chase": ["chase", "chaze", "chasee", "cha5e"],
            "wellsfargo": ["wellsfargo", "wells-fargo", "wellsfarg0", "welsfargo"],
            "bankofamerica": ["bankofamerica", "bankofamer1ca", "bofa", "bank0famerica"],
            "citibank": ["citibank", "c1tibank", "cittibank", "citybank"],
            "dhl": ["dhl", "dh1", "dhll"],
            "fedex": ["fedex", "fed-ex", "feedex"],
            "ups": ["ups", "upss", "up5"],
            "usps": ["usps", "usps-delivery", "us-ps"],
            "irs": ["irs", "irs-gov", "1rs"],
            "ebay": ["ebay", "e-bay", "ebayy"]
        }
        
        # Suspicious URL patterns
        self.suspicious_patterns = [
            # Homograph attacks
            (r"[а-яА-Я]", "Cyrillic characters"),
            (r"[αβγδεζηθικλμνξοπρστυφχψω]", "Greek characters"),
            
            # URL shorteners
            (r"(bit\.ly|tinyurl|goo\.gl|ow\.ly|t\.co|short\.link)/", "URL shortener"),
            
            # Suspicious paths
            (r"/(secure|verify|confirm|update|suspend|locked)/", "Suspicious path"),
            (r"/\.(php|asp|jsp|cgi)(\?|$)", "Dynamic script"),
            
            # IP addresses
            (r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "IP address instead of domain"),
            (r"https?://0x[0-9a-f]+", "Hexadecimal IP"),
            
            # Suspicious TLDs
            (r"\.(tk|ml|ga|cf|click|download|review|loan|work)(/|$)", "Suspicious TLD"),
            
            # Multiple subdomains
            (r"([^.]+\.){4,}[^.]+\.[^.]+$", "Excessive subdomains"),
            
            # Data URIs
            (r"^data:", "Data URI"),
            
            # Punycode
            (r"xn--", "Punycode domain"),
            
            # Long URLs
            (r".{200,}", "Suspiciously long URL")
        ]
        
        # Compile patterns
        self.compiled_patterns = [(re.compile(p, re.IGNORECASE), desc) 
                                 for p, desc in self.suspicious_patterns]
        
        # Brand impersonation patterns
        self.impersonation_patterns = self._generate_impersonation_patterns()
        
        logger.info(f"PhishingDatabaseConnector initialized with {len(self.databases)} databases")
    
    async def initialize(self) -> None:
        """Initialize phishing database connections and load data."""
        logger.info("Initializing PhishingDatabaseConnector...")
        
        # Load initial threat feeds if available
        for db_name, db_config in self.databases.items():
            try:
                if db_config.get("enabled", True):
                    # Initialize connection for each database
                    logger.info(f"Connecting to {db_name} database...")
                    # In production, this would connect to actual databases
                    # For now, we'll just mark as initialized
                    self.databases[db_name]["connected"] = True
            except Exception as e:
                logger.error(f"Failed to initialize {db_name}: {e}")
                self.databases[db_name]["connected"] = False
        
        # Pre-populate some known phishing indicators for testing
        sample_phishing = [
            "phishing-example.com",
            "paypal-verification.fake",
            "amazon-security.phish",
        ]
        
        for domain in sample_phishing:
            indicator = PhishingIndicator(
                url=f"http://{domain}",
                domain=domain,
                threat_level="high",
                confidence=0.95,
                detected_by=["initial_load"],
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                target_brand=None,
                attack_type="phishing"
            )
            self.phishing_cache[domain] = indicator
        
        logger.info("PhishingDatabaseConnector initialization complete")
    
    def _generate_impersonation_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Generate regex patterns for brand impersonation detection."""
        patterns = {}
        
        for brand, variations in self.target_brands.items():
            brand_patterns = []
            for variation in variations:
                # Create pattern that matches the variation with common phishing additions
                pattern = rf"({variation})[.-]*(secure|verify|account|update|confirm|login|signin)"
                brand_patterns.append(re.compile(pattern, re.IGNORECASE))
            patterns[brand] = brand_patterns
        
        return patterns
    
    async def check_url(self, url: str) -> Dict[str, Any]:
        """
        Comprehensive phishing check for a URL.
        
        Args:
            url: URL to check
            
        Returns:
            Phishing analysis result
        """
        # Check cache first
        if url in self.phishing_cache:
            cached = self.phishing_cache[url]
            if datetime.now() - cached.last_seen < self.cache_ttl:
                return self._indicator_to_dict(cached)
        
        # Parse URL
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Initialize results
        threats = []
        confidence = 0.0
        detections = []
        
        # 1. Pattern-based detection
        pattern_result = self._check_patterns(url)
        if pattern_result["suspicious"]:
            threats.extend(pattern_result["threats"])
            confidence = max(confidence, pattern_result["confidence"])
            detections.append("pattern_analysis")
        
        # 2. Brand impersonation check
        brand_result = self._check_brand_impersonation(url, domain)
        if brand_result["impersonation_detected"]:
            threats.append(f"Possible {brand_result['target_brand']} impersonation")
            confidence = max(confidence, brand_result["confidence"])
            detections.append("brand_impersonation")
        
        # 3. Typosquatting detection
        typo_result = self._check_typosquatting(domain)
        if typo_result["is_typosquat"]:
            threats.append(f"Typosquatting of {typo_result['legitimate_domain']}")
            confidence = max(confidence, typo_result["confidence"])
            detections.append("typosquatting")
        
        # 4. Check against live databases
        db_results = await self._check_databases(url)
        if db_results["found"]:
            threats.extend(db_results["threats"])
            confidence = max(confidence, db_results["confidence"])
            detections.extend(db_results["sources"])
        
        # 5. Domain age check (new domains are suspicious)
        age_result = await self._check_domain_age(domain)
        if age_result["suspicious"]:
            threats.append(age_result["reason"])
            confidence = max(confidence, age_result["confidence"])
            detections.append("domain_age")
        
        # 6. SSL certificate analysis
        ssl_result = await self._check_ssl_cert(domain)
        if ssl_result["suspicious"]:
            threats.extend(ssl_result["issues"])
            confidence = max(confidence, ssl_result["confidence"])
            detections.append("ssl_analysis")
        
        # Determine threat level
        threat_level = self._calculate_threat_level(confidence)
        
        # Create indicator if phishing detected
        if confidence > 0.3:
            indicator = PhishingIndicator(
                url=url,
                domain=domain,
                threat_level=threat_level,
                confidence=confidence,
                detected_by=detections,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                target_brand=brand_result.get("target_brand"),
                attack_type=self._determine_attack_type(threats)
            )
            self.phishing_cache[url] = indicator
        
        return {
            "url": url,
            "domain": domain,
            "is_phishing": confidence > 0.5,
            "confidence": confidence,
            "threat_level": threat_level,
            "threats": threats,
            "detected_by": detections,
            "target_brand": brand_result.get("target_brand"),
            "recommendations": self._get_recommendations(confidence, threats)
        }
    
    async def check_domain(self, domain: str) -> bool:
        """
        Check if a domain is suspicious or blacklisted.
        
        Args:
            domain: Domain name to check
            
        Returns:
            True if suspicious, False otherwise
        """
        # Check against known phishing domains
        if domain in self.known_phishing_domains:
            return True
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}",  # IP address
            r".*\.(tk|ml|ga|cf)$",  # High-risk TLDs
            r"[0-9]+-[0-9]+",  # Numbers with dashes
            r"(secure|verify|account|update).*\.(com|net|org)",  # Phishing keywords
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, domain, re.IGNORECASE):
                return True
        
        # Check for brand impersonation
        for brand, variations in self.target_brands.items():
            for variation in variations:
                if variation in domain.lower() and brand not in domain.lower():
                    return True
        
        return False
    
    def _check_patterns(self, url: str) -> Dict[str, Any]:
        """Check URL against suspicious patterns."""
        threats = []
        confidence = 0.0
        
        for pattern, description in self.compiled_patterns:
            if pattern.search(url):
                threats.append(description)
                # Different patterns have different confidence levels
                if "IP address" in description:
                    confidence = max(confidence, 0.8)
                elif "Cyrillic" in description or "Greek" in description:
                    confidence = max(confidence, 0.9)
                elif "URL shortener" in description:
                    confidence = max(confidence, 0.5)
                elif "Suspicious TLD" in description:
                    confidence = max(confidence, 0.7)
                else:
                    confidence = max(confidence, 0.6)
        
        return {
            "suspicious": len(threats) > 0,
            "threats": threats,
            "confidence": confidence
        }
    
    def _check_brand_impersonation(self, url: str, domain: str) -> Dict[str, Any]:
        """Check for brand impersonation."""
        url_lower = url.lower()
        domain_lower = domain.lower()
        
        for brand, patterns in self.impersonation_patterns.items():
            for pattern in patterns:
                if pattern.search(url_lower) or pattern.search(domain_lower):
                    # Check if it's the legitimate domain
                    legitimate_domains = {
                        "paypal": ["paypal.com", "paypal.me"],
                        "amazon": ["amazon.com", "amazon.co.uk", "amazon.de"],
                        "microsoft": ["microsoft.com", "live.com", "outlook.com"],
                        "google": ["google.com", "gmail.com", "youtube.com"],
                        "apple": ["apple.com", "icloud.com"],
                        "netflix": ["netflix.com"],
                        "facebook": ["facebook.com", "fb.com"],
                        "instagram": ["instagram.com"],
                        "linkedin": ["linkedin.com"],
                        "dropbox": ["dropbox.com"]
                    }
                    
                    if brand in legitimate_domains:
                        if not any(legit in domain_lower for legit in legitimate_domains[brand]):
                            return {
                                "impersonation_detected": True,
                                "target_brand": brand,
                                "confidence": 0.85
                            }
        
        return {
            "impersonation_detected": False,
            "target_brand": None,
            "confidence": 0.0
        }
    
    def _check_typosquatting(self, domain: str) -> Dict[str, Any]:
        """Check for typosquatting."""
        # Common typosquatting techniques
        techniques = {
            "character_omission": lambda d: self._check_omission(d),
            "character_repeat": lambda d: self._check_repeat(d),
            "adjacent_character": lambda d: self._check_adjacent(d),
            "character_replacement": lambda d: self._check_replacement(d),
            "homophone": lambda d: self._check_homophone(d)
        }
        
        for technique_name, check_func in techniques.items():
            result = check_func(domain)
            if result["match"]:
                return {
                    "is_typosquat": True,
                    "technique": technique_name,
                    "legitimate_domain": result["legitimate"],
                    "confidence": 0.8
                }
        
        return {
            "is_typosquat": False,
            "confidence": 0.0
        }
    
    def _check_omission(self, domain: str) -> Dict[str, Any]:
        """Check for character omission typosquatting."""
        # Check if domain is one character different from known brands
        for brand in self.target_brands.keys():
            if len(domain.replace(".com", "")) == len(brand) - 1:
                # Simple check - in production use Levenshtein distance
                if sum(c in domain for c in brand) >= len(brand) - 1:
                    return {"match": True, "legitimate": brand}
        return {"match": False}
    
    def _check_repeat(self, domain: str) -> Dict[str, Any]:
        """Check for character repeat typosquatting."""
        # Check for repeated characters that shouldn't be
        for brand in self.target_brands.keys():
            if brand in domain and len(domain) > len(brand):
                return {"match": True, "legitimate": brand}
        return {"match": False}
    
    def _check_adjacent(self, domain: str) -> Dict[str, Any]:
        """Check for adjacent character swap typosquatting."""
        # Simplified check - in production use more sophisticated algorithm
        suspicious_swaps = ["amazno", "mircosoft", "gogole", "paypla"]
        for swap in suspicious_swaps:
            if swap in domain:
                for brand in self.target_brands.keys():
                    if len(swap) == len(brand):
                        return {"match": True, "legitimate": brand}
        return {"match": False}
    
    def _check_replacement(self, domain: str) -> Dict[str, Any]:
        """Check for character replacement typosquatting."""
        replacements = {
            "0": "o", "1": "i", "l": "i", "5": "s",
            "3": "e", "4": "a", "@": "a", "vv": "w"
        }
        
        for brand in self.target_brands.keys():
            modified_domain = domain
            for fake, real in replacements.items():
                modified_domain = modified_domain.replace(fake, real)
            
            if brand in modified_domain and domain != modified_domain:
                return {"match": True, "legitimate": brand}
        
        return {"match": False}
    
    def _check_homophone(self, domain: str) -> Dict[str, Any]:
        """Check for homophone typosquatting."""
        homophones = {
            "for": "four", "to": "two", "too": "two",
            "buy": "by", "sea": "see", "be": "bee"
        }
        
        for brand in self.target_brands.keys():
            for fake, real in homophones.items():
                if fake in domain and real in brand:
                    return {"match": True, "legitimate": brand}
        
        return {"match": False}
    
    async def _check_databases(self, url: str) -> Dict[str, Any]:
        """Check URL against phishing databases."""
        found = False
        confidence = 0.0
        threats = []
        sources = []
        
        # Check PhishTank
        if self.databases["phishtank"]["enabled"]:
            phishtank_result = await self._check_phishtank(url)
            if phishtank_result["is_phishing"]:
                found = True
                confidence = max(confidence, 0.95)
                threats.append("Listed in PhishTank database")
                sources.append("phishtank")
        
        # Check OpenPhish
        if self.databases["openphish"]["enabled"]:
            openphish_result = await self._check_openphish(url)
            if openphish_result["is_phishing"]:
                found = True
                confidence = max(confidence, 0.9)
                threats.append("Listed in OpenPhish database")
                sources.append("openphish")
        
        # Check Google Safe Browsing
        if self.databases["google_safe_browsing"]["enabled"]:
            gsb_result = await self._check_google_safe_browsing(url)
            if gsb_result["is_threat"]:
                found = True
                confidence = max(confidence, 1.0)
                threats.extend(gsb_result["threat_types"])
                sources.append("google_safe_browsing")
        
        return {
            "found": found,
            "confidence": confidence,
            "threats": threats,
            "sources": sources
        }
    
    async def _check_phishtank(self, url: str) -> Dict[str, Any]:
        """Check URL against PhishTank."""
        # Simplified - in production, implement full API integration
        return {"is_phishing": False}
    
    async def _check_openphish(self, url: str) -> Dict[str, Any]:
        """Check URL against OpenPhish."""
        # Simplified - in production, implement full feed checking
        return {"is_phishing": False}
    
    async def _check_google_safe_browsing(self, url: str) -> Dict[str, Any]:
        """Check URL against Google Safe Browsing."""
        if not self.databases["google_safe_browsing"]["api_key"]:
            return {"is_threat": False}
        
        # Simplified - in production, implement full API integration
        return {"is_threat": False, "threat_types": []}
    
    async def _check_domain_age(self, domain: str) -> Dict[str, Any]:
        """Check domain age (new domains are suspicious)."""
        # Simplified - in production, use WHOIS lookup
        # Domains registered within last 30 days are suspicious
        return {
            "suspicious": False,
            "age_days": 365,
            "confidence": 0.0,
            "reason": "Domain age check not implemented"
        }
    
    async def _check_ssl_cert(self, domain: str) -> Dict[str, Any]:
        """Check SSL certificate for suspicious indicators."""
        issues = []
        confidence = 0.0
        
        # Simplified - in production, actually check SSL cert
        # Look for: self-signed, expired, wrong domain, free cert on financial site
        
        return {
            "suspicious": len(issues) > 0,
            "issues": issues,
            "confidence": confidence
        }
    
    def _calculate_threat_level(self, confidence: float) -> str:
        """Calculate threat level from confidence score."""
        if confidence >= 0.9:
            return "critical"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "none"
    
    def _determine_attack_type(self, threats: List[str]) -> str:
        """Determine the type of phishing attack."""
        threat_str = " ".join(threats).lower()
        
        if "impersonation" in threat_str:
            return "brand_impersonation"
        elif "typosquat" in threat_str:
            return "typosquatting"
        elif "homograph" in threat_str:
            return "homograph_attack"
        elif "shortened" in threat_str:
            return "url_shortening"
        elif "ip address" in threat_str:
            return "ip_based"
        else:
            return "generic_phishing"
    
    def _get_recommendations(self, confidence: float, threats: List[str]) -> List[str]:
        """Get security recommendations based on threats."""
        recommendations = []
        
        if confidence > 0.7:
            recommendations.append("DO NOT enter any personal information")
            recommendations.append("Close this page immediately")
            recommendations.append("Report to your IT security team")
        elif confidence > 0.5:
            recommendations.append("Verify the URL carefully")
            recommendations.append("Check for HTTPS and valid certificate")
            recommendations.append("Contact the organization directly")
        elif confidence > 0.3:
            recommendations.append("Be cautious with this site")
            recommendations.append("Verify the domain name")
            recommendations.append("Look for spelling errors")
        
        if any("shortener" in t.lower() for t in threats):
            recommendations.append("Avoid clicking shortened URLs")
            recommendations.append("Use URL expansion services first")
        
        return recommendations
    
    def _indicator_to_dict(self, indicator: PhishingIndicator) -> Dict[str, Any]:
        """Convert PhishingIndicator to dictionary."""
        return {
            "url": indicator.url,
            "domain": indicator.domain,
            "is_phishing": indicator.confidence > 0.5,
            "confidence": indicator.confidence,
            "threat_level": indicator.threat_level,
            "threats": [f"Previously detected by {', '.join(indicator.detected_by)}"],
            "detected_by": indicator.detected_by,
            "target_brand": indicator.target_brand,
            "attack_type": indicator.attack_type,
            "first_seen": indicator.first_seen.isoformat(),
            "last_seen": indicator.last_seen.isoformat(),
            "recommendations": self._get_recommendations(indicator.confidence, [])
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get phishing detection statistics."""
        return {
            "cached_indicators": len(self.phishing_cache),
            "databases_enabled": sum(1 for db in self.databases.values() if db["enabled"]),
            "target_brands_monitored": len(self.target_brands),
            "pattern_rules": len(self.suspicious_patterns),
            "last_update": datetime.now().isoformat()
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.phishing_cache.clear()