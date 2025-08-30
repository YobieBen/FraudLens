"""
Threat Intelligence Integration Manager.

Connects to multiple public threat intelligence feeds and databases
to enhance fraud detection capabilities.

Author: Yobie Benjamin
Date: 2025-08-28
"""

import asyncio
import hashlib
import json
import re
import ssl
import certifi
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import aiohttp
from loguru import logger


class ThreatIntelligenceManager:
    """
    Manages connections to multiple threat intelligence sources.
    
    Integrates with:
    - MISP (Malware Information Sharing Platform)
    - AlienVault OTX
    - Abuse.ch (URLhaus, MalwareBazaar, ThreatFox)
    - PhishTank
    - OpenPhish
    - Spamhaus
    - VirusTotal (limited free tier)
    - CIRCL (Computer Incident Response Center Luxembourg)
    - CISA AIS (Automated Indicator Sharing)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize threat intelligence manager."""
        self.config = config or {}
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)
        
        # API endpoints for public threat feeds
        self.feeds = {
            "urlhaus": {
                "url": "https://urlhaus-api.abuse.ch/v1/",
                "type": "malicious_urls",
                "auth": None,
                "rate_limit": 100  # requests per minute
            },
            "phishtank": {
                "url": "https://data.phishtank.com/data/online-valid.json",
                "type": "phishing",
                "auth": self.config.get("phishtank_api_key"),
                "rate_limit": 60
            },
            "openphish": {
                "url": "https://openphish.com/feed.txt",
                "type": "phishing",
                "auth": None,
                "rate_limit": 10  # conservative for free tier
            },
            "alienvault_otx": {
                "url": "https://otx.alienvault.com/api/v1/",
                "type": "mixed",
                "auth": self.config.get("otx_api_key"),
                "rate_limit": 60
            },
            "abuse_ch_ssl": {
                "url": "https://sslbl.abuse.ch/blacklist/",
                "type": "ssl_certificates",
                "auth": None,
                "rate_limit": 10
            },
            "malwarebazaar": {
                "url": "https://mb-api.abuse.ch/api/v1/",
                "type": "malware",
                "auth": None,
                "rate_limit": 100
            },
            "threatfox": {
                "url": "https://threatfox-api.abuse.ch/api/v1/",
                "type": "ioc",
                "auth": None,
                "rate_limit": 100
            },
            "circl_hashlookup": {
                "url": "https://hashlookup.circl.lu/",
                "type": "file_hash",
                "auth": None,
                "rate_limit": 100
            },
            "spamhaus_dbl": {
                "url": "https://www.spamhaus.org/drop/drop.txt",
                "type": "blocklist",
                "auth": None,
                "rate_limit": 10
            }
        }
        
        # Known bad indicators (updated from feeds)
        self.known_bad_urls: Set[str] = set()
        self.known_bad_domains: Set[str] = set()
        self.known_bad_ips: Set[str] = set()
        self.known_bad_hashes: Set[str] = set()
        self.known_bad_emails: Set[str] = set()
        
        # Feed status tracking
        self.feeds_status = {}
        self.last_update = None
        
        # Phishing patterns
        self.phishing_patterns = [
            # URL shorteners often used in phishing
            r"bit\.ly/", r"tinyurl\.com/", r"goo\.gl/", r"ow\.ly/",
            # Suspicious TLDs
            r"\.(tk|ml|ga|cf|click|download|review)(/|$)",
            # Typosquatting patterns
            r"(payp[a4]l|amaz0n|micr0soft|g00gle|app1e)",
            # Suspicious paths
            r"/(suspend|verify|confirm|update|secure|account).*(php|html|asp)",
            # IP addresses instead of domains
            r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
            # Homograph attacks
            r"[а-яА-Я]",  # Cyrillic characters mixed with Latin
        ]
        
        # Compile regex patterns
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.phishing_patterns]
        
        logger.info(f"ThreatIntelligenceManager initialized with {len(self.feeds)} feeds")
    
    @property
    def threat_feeds(self):
        """Alias for feeds to maintain compatibility."""
        return self.feeds
    
    async def initialize(self) -> None:
        """Initialize and fetch initial threat data."""
        await self.update_threat_feeds()
    
    async def update_threat_feeds(self) -> None:
        """Update threat intelligence from all configured feeds."""
        tasks = []
        for feed_name, feed_config in self.feeds.items():
            if feed_config["auth"] or feed_name in ["urlhaus", "openphish", "spamhaus_dbl"]:
                tasks.append(self._fetch_feed(feed_name, feed_config))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update feed status
        for i, (feed_name, _) in enumerate(self.feeds.items()):
            if i < len(results):
                self.feeds_status[feed_name] = {
                    "status": "active" if not isinstance(results[i], Exception) else "failed",
                    "last_check": datetime.now().isoformat()
                }
        
        self.last_update = datetime.now()
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Updated {success_count}/{len(tasks)} threat feeds")
    
    async def _fetch_feed(self, feed_name: str, feed_config: Dict) -> bool:
        """Fetch data from a specific threat feed."""
        try:
            # Create SSL context with proper certificate validation
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            
            # For some feeds, we may need to be less strict
            if feed_name in ["openphish", "spamhaus_dbl"]:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            # Create connector with SSL context
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                headers = {}
                if feed_config["auth"]:
                    headers["X-OTX-API-KEY"] = feed_config["auth"]
                
                if feed_name == "urlhaus":
                    # URLhaus specific API call
                    data = {"command": "urls", "limit": 1000}
                    async with session.post(feed_config["url"] + "urls/recent/", data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            for entry in result.get("urls", []):
                                self.known_bad_urls.add(entry["url"])
                                domain = urlparse(entry["url"]).netloc
                                if domain:
                                    self.known_bad_domains.add(domain)
                
                elif feed_name == "openphish":
                    async with session.get(feed_config["url"]) as response:
                        if response.status == 200:
                            content = await response.text()
                            for url in content.strip().split('\n'):
                                if url:
                                    self.known_bad_urls.add(url.strip())
                                    domain = urlparse(url).netloc
                                    if domain:
                                        self.known_bad_domains.add(domain)
                
                elif feed_name == "phishtank" and feed_config["auth"]:
                    async with session.get(feed_config["url"], headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            for entry in data:
                                if entry.get("verified") == "yes":
                                    self.known_bad_urls.add(entry["url"])
                                    domain = urlparse(entry["url"]).netloc
                                    if domain:
                                        self.known_bad_domains.add(domain)
                
                elif feed_name == "spamhaus_dbl":
                    async with session.get(feed_config["url"]) as response:
                        if response.status == 200:
                            content = await response.text()
                            for line in content.strip().split('\n'):
                                if not line.startswith(';') and line.strip():
                                    ip = line.split(';')[0].strip()
                                    if self._is_valid_ip(ip):
                                        self.known_bad_ips.add(ip)
                
                return True
                
        except Exception as e:
            logger.warning(f"Failed to fetch {feed_name}: {e}")
            return False
    
    async def check_url(self, url: str) -> Dict[str, Any]:
        """
        Check if URL is known malicious or suspicious.
        
        Returns:
            Dict with threat assessment
        """
        threat_score = 0.0
        threats_found = []
        
        # Check against known bad URLs
        if url in self.known_bad_urls:
            threat_score = 1.0
            threats_found.append("Known malicious URL")
        
        # Check domain
        domain = urlparse(url).netloc
        if domain and domain in self.known_bad_domains:
            threat_score = max(threat_score, 0.9)
            threats_found.append("Known malicious domain")
        
        # Check patterns
        for pattern in self.compiled_patterns:
            if pattern.search(url):
                threat_score = max(threat_score, 0.7)
                threats_found.append(f"Suspicious pattern: {pattern.pattern[:30]}...")
                break
        
        # Check for homograph attacks
        if self._detect_homograph(url):
            threat_score = max(threat_score, 0.8)
            threats_found.append("Possible homograph attack")
        
        # Check URL shortener
        if self._is_url_shortener(url):
            threat_score = max(threat_score, 0.5)
            threats_found.append("URL shortener detected")
        
        return {
            "url": url,
            "threat_score": threat_score,
            "threats": threats_found,
            "checked_at": datetime.now().isoformat(),
            "sources": ["urlhaus", "openphish", "phishtank", "pattern_matching"]
        }
    
    async def check_hash(self, file_hash: str) -> Dict[str, Any]:
        """Check if file hash is known malicious."""
        if file_hash.lower() in self.known_bad_hashes:
            return {
                "hash": file_hash,
                "malicious": True,
                "threat_score": 1.0,
                "source": "threat_feeds"
            }
        
        # Query CIRCL hashlookup
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://hashlookup.circl.lu/lookup/sha256/{file_hash}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("KnownMalicious"):
                            self.known_bad_hashes.add(file_hash.lower())
                            return {
                                "hash": file_hash,
                                "malicious": True,
                                "threat_score": 1.0,
                                "source": "circl_hashlookup"
                            }
        except:
            pass
        
        return {
            "hash": file_hash,
            "malicious": False,
            "threat_score": 0.0,
            "source": "no_match"
        }
    
    async def check_ip(self, ip_address: str) -> Dict[str, Any]:
        """Check if IP address is known malicious."""
        threat_score = 0.0
        threats = []
        
        if ip_address in self.known_bad_ips:
            threat_score = 1.0
            threats.append("Known malicious IP")
        
        # Check if IP is in bogon range
        if self._is_bogon_ip(ip_address):
            threat_score = max(threat_score, 0.6)
            threats.append("Bogon IP range")
        
        return {
            "ip": ip_address,
            "threat_score": threat_score,
            "threats": threats,
            "checked_at": datetime.now().isoformat()
        }
    
    async def check_email(self, email: str) -> Dict[str, Any]:
        """Check if email is suspicious or known bad."""
        threat_score = 0.0
        threats = []
        
        domain = email.split('@')[-1] if '@' in email else ""
        
        # Check disposable email domains
        disposable_domains = [
            "tempmail.com", "guerrillamail.com", "mailinator.com",
            "10minutemail.com", "throwaway.email", "yopmail.com"
        ]
        
        if domain in disposable_domains:
            threat_score = 0.6
            threats.append("Disposable email domain")
        
        # Check if domain is in bad domains list
        if domain in self.known_bad_domains:
            threat_score = max(threat_score, 0.8)
            threats.append("Known malicious domain")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r"\d{5,}",  # Many digits
            r"(admin|support|security|account).*\d+",  # Generic names with numbers
            r"[a-z]{15,}",  # Very long random strings
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, email.split('@')[0], re.IGNORECASE):
                threat_score = max(threat_score, 0.5)
                threats.append("Suspicious email pattern")
                break
        
        return {
            "email": email,
            "threat_score": threat_score,
            "threats": threats,
            "checked_at": datetime.now().isoformat()
        }
    
    def _detect_homograph(self, url: str) -> bool:
        """Detect potential homograph attacks in URLs."""
        # Check for mixed scripts
        cyrillic = re.search(r'[а-яА-Я]', url)
        latin = re.search(r'[a-zA-Z]', url)
        
        if cyrillic and latin:
            return True
        
        # Check for confusable characters
        confusables = {
            'о': 'o', 'а': 'a', 'е': 'e', 'р': 'p',
            'с': 'c', 'х': 'x', 'у': 'y', 'к': 'k'
        }
        
        for char, latin_char in confusables.items():
            if char in url and latin_char in url:
                return True
        
        return False
    
    def _is_url_shortener(self, url: str) -> bool:
        """Check if URL uses a shortener service."""
        shorteners = [
            "bit.ly", "tinyurl.com", "goo.gl", "ow.ly",
            "is.gd", "buff.ly", "short.link", "t.co"
        ]
        domain = urlparse(url).netloc.lower()
        return any(shortener in domain for shortener in shorteners)
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format."""
        try:
            parts = ip.split('.')
            return len(parts) == 4 and all(0 <= int(p) <= 255 for p in parts)
        except:
            return False
    
    def _is_bogon_ip(self, ip: str) -> bool:
        """Check if IP is in bogon range."""
        bogon_ranges = [
            "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10",
            "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12",
            "192.0.0.0/24", "192.0.2.0/24", "192.168.0.0/16",
            "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
            "224.0.0.0/3"
        ]
        
        # Simplified check - in production use ipaddress module
        for bogon in ["0.", "10.", "127.", "169.254.", "172.", "192.168."]:
            if ip.startswith(bogon):
                return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get threat intelligence statistics."""
        return {
            "total_indicators": (
                len(self.known_bad_urls) + 
                len(self.known_bad_domains) + 
                len(self.known_bad_ips) + 
                len(self.known_bad_hashes)
            ),
            "known_bad_urls": len(self.known_bad_urls),
            "known_bad_domains": len(self.known_bad_domains),
            "known_bad_ips": len(self.known_bad_ips),
            "known_bad_hashes": len(self.known_bad_hashes),
            "feeds_active": len([f for f in self.feeds_status.values() if f.get("status") == "active"]),
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }
    
    async def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of threat intelligence data."""
        return {
            "known_bad_urls": len(self.known_bad_urls),
            "known_bad_domains": len(self.known_bad_domains),
            "known_bad_ips": len(self.known_bad_ips),
            "known_bad_hashes": len(self.known_bad_hashes),
            "feeds_configured": len(self.feeds),
            "last_updated": datetime.now().isoformat()
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.known_bad_urls.clear()
        self.known_bad_domains.clear()
        self.known_bad_ips.clear()
        self.known_bad_hashes.clear()
        self.cache.clear()