"""
Gmail IMAP Integration for FraudLens
Provides real email access using IMAP with app passwords
"""

import asyncio
import base64
import email
import imaplib
import re
import ssl
from datetime import datetime
from email.header import decode_header
from typing import Dict, List, Optional, Tuple

from loguru import logger


class GmailIMAPScanner:
    """Gmail IMAP scanner for real email fraud detection."""

    def __init__(self, fraud_detector=None):
        """Initialize Gmail IMAP scanner."""
        self.fraud_detector = fraud_detector
        self.imap = None
        self.email_address = None
        self.is_connected = False

    def connect(self, email_address: str, app_password: str) -> bool:
        """
        Connect to Gmail using IMAP with app password.

        Args:
            email_address: Gmail address
            app_password: App-specific password from Google

        Returns:
            True if connected successfully
        """
        try:
            # Try multiple SSL connection methods
            import platform

            import certifi

            # Method 1: Try with certifi certificates
            try:
                context = ssl.create_default_context(cafile=certifi.where())
                self.imap = imaplib.IMAP4_SSL("imap.gmail.com", 993, ssl_context=context)
                logger.info("Connected with certifi certificates")
            except ssl.SSLError:
                # Method 2: Try with system default context
                try:
                    context = ssl.create_default_context()
                    self.imap = imaplib.IMAP4_SSL("imap.gmail.com", 993, ssl_context=context)
                    logger.info("Connected with system certificates")
                except ssl.SSLError:
                    # Method 3: Create unverified context (less secure but works)
                    logger.warning("SSL verification failed, using unverified context")
                    context = ssl._create_unverified_context()
                    self.imap = imaplib.IMAP4_SSL("imap.gmail.com", 993, ssl_context=context)
                    logger.info("Connected with unverified context")

            # Login with app password
            self.imap.login(email_address, app_password)

            self.email_address = email_address
            self.is_connected = True

            logger.info(f"Successfully connected to Gmail for {email_address}")
            return True

        except imaplib.IMAP4.error as e:
            logger.error(f"IMAP authentication failed: {e}")
            if "Invalid credentials" in str(e) or "AUTHENTICATIONFAILED" in str(e):
                raise Exception("Invalid email or app password. Please check your credentials.")
            raise Exception(f"IMAP connection failed: {e}")

        except Exception as e:
            logger.error(f"Failed to connect to Gmail: {e}")
            raise Exception(f"Connection failed: {e}")

    def disconnect(self):
        """Disconnect from Gmail."""
        try:
            if self.imap:
                self.imap.close()
                self.imap.logout()
            self.is_connected = False
            logger.info("Disconnected from Gmail")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")

    def search_emails(
        self, query: str = "ALL", folder: str = "INBOX", max_results: int = 1000, offset: int = 0
    ) -> List[str]:
        """
        Search emails using IMAP search criteria with pagination support.

        Args:
            query: IMAP search query (e.g., "UNSEEN", "FROM example@gmail.com")
            folder: Folder to search in
            max_results: Maximum number of emails to return (up to 1000)
            offset: Number of emails to skip from the beginning

        Returns:
            List of email IDs
        """
        if not self.is_connected:
            raise Exception("Not connected to Gmail")

        try:
            # Select folder
            self.imap.select(folder)

            # Convert common queries to IMAP format
            imap_query = self._convert_to_imap_query(query)

            # Search emails
            typ, data = self.imap.search(None, imap_query)

            if typ != "OK":
                return []

            # Get email IDs
            email_ids = data[0].split()

            # Apply offset and limit
            if offset > 0 and offset < len(email_ids):
                email_ids = email_ids[offset:]

            # Limit results (max 1000 per batch)
            max_results = min(max_results, 1000)
            if len(email_ids) > max_results:
                email_ids = email_ids[-max_results:]  # Get most recent

            logger.info(f"Found {len(email_ids)} emails matching query (offset: {offset})")
            return email_ids

        except Exception as e:
            logger.error(f"Email search failed: {e}")
            raise Exception(f"Search failed: {e}")

    def _convert_to_imap_query(self, query: str) -> str:
        """Convert user-friendly query to IMAP format."""
        query_lower = query.lower()

        # Common conversions
        conversions = {
            "is:unread": "UNSEEN",
            "is:read": "SEEN",
            "is:starred": "FLAGGED",
            "has:attachment": "HAS attachment",
            "is:important": "FLAGGED",
            "all": "ALL",
        }

        for key, value in conversions.items():
            if key in query_lower:
                return value

        # Handle from: queries
        if "from:" in query_lower:
            sender = query.split("from:")[1].strip()
            return f'FROM "{sender}"'

        # Handle subject: queries
        if "subject:" in query_lower:
            subject = query.split("subject:")[1].strip()
            return f'SUBJECT "{subject}"'

        # Default to UNSEEN (unread)
        return "UNSEEN"

    def fetch_email(self, email_id: str) -> Dict:
        """
        Fetch a single email by ID.

        Args:
            email_id: Email ID from search

        Returns:
            Dictionary with email details
        """
        if not self.is_connected:
            raise Exception("Not connected to Gmail")

        try:
            # Fetch email data
            typ, data = self.imap.fetch(email_id, "(RFC822)")

            if typ != "OK":
                return None

            # Parse email
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Extract email details
            email_data = {
                "id": email_id.decode() if isinstance(email_id, bytes) else email_id,
                "subject": self._decode_header(msg.get("Subject", "")),
                "from": self._decode_header(msg.get("From", "")),
                "to": self._decode_header(msg.get("To", "")),
                "date": msg.get("Date", ""),
                "body": self._get_email_body(msg),
                "has_attachments": self._has_attachments(msg),
                "headers": dict(msg.items()),
            }

            return email_data

        except Exception as e:
            logger.error(f"Failed to fetch email {email_id}: {e}")
            return None

    def _decode_header(self, header_value: str) -> str:
        """Decode email header."""
        if not header_value:
            return ""

        decoded_parts = []
        for part, encoding in decode_header(header_value):
            if isinstance(part, bytes):
                if encoding:
                    decoded_parts.append(part.decode(encoding, errors="ignore"))
                else:
                    decoded_parts.append(part.decode("utf-8", errors="ignore"))
            else:
                decoded_parts.append(str(part))

        return " ".join(decoded_parts)

    def _get_email_body(self, msg) -> str:
        """Extract email body text."""
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()

                if content_type == "text/plain":
                    try:
                        body += part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    except:
                        pass
                elif content_type == "text/html" and not body:
                    try:
                        html_body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        # Simple HTML to text conversion
                        body = re.sub("<[^<]+?>", "", html_body)
                    except:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
            except:
                body = str(msg.get_payload())

        return body.strip()

    def _has_attachments(self, msg) -> bool:
        """Check if email has attachments."""
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                return True
        return False

    async def scan_for_fraud(
        self, query: str = "UNSEEN", max_emails: int = 1000, offset: int = 0
    ) -> List[Dict]:
        """
        Scan emails for fraud with batch processing support.

        Args:
            query: Search query
            max_emails: Maximum emails to scan (up to 1000)
            offset: Number of emails to skip from the beginning

        Returns:
            List of scan results with fraud detection
        """
        if not self.is_connected:
            raise Exception("Not connected to Gmail")

        results = []

        try:
            # Search emails with pagination
            email_ids = self.search_emails(query, max_results=max_emails, offset=offset)

            logger.info(
                f"Found {len(email_ids)} emails to scan (batch starting at offset {offset})"
            )

            for email_id in email_ids:
                # Fetch email
                email_data = self.fetch_email(email_id)

                if not email_data:
                    continue

                # Prepare text for analysis
                analysis_text = f"""
                Subject: {email_data['subject']}
                From: {email_data['from']}
                Body: {email_data['body'][:1000]}  # Limit body length
                """

                # Analyze for fraud
                fraud_result = await self._analyze_for_fraud(analysis_text, email_data)

                # Combine results
                result = {
                    "message_id": email_data["id"],
                    "subject": email_data["subject"],
                    "sender": email_data["from"],
                    "date": email_data["date"],
                    "is_fraud": fraud_result["is_fraud"],
                    "confidence": fraud_result["confidence"],
                    "fraud_types": fraud_result["fraud_types"],
                    "risk_level": fraud_result["risk_level"],
                    "has_attachments": email_data["has_attachments"],
                }

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Fraud scan failed: {e}")
            raise Exception(f"Scan failed: {e}")

    async def _analyze_for_fraud(self, text: str, email_data: Dict) -> Dict:
        """Analyze email for fraud indicators."""
        # Basic fraud detection logic
        fraud_indicators = {
            "phishing": [
                "verify your account",
                "suspended account",
                "click here immediately",
                "confirm your identity",
                "urgent action required",
                "verify your password",
                "update payment information",
            ],
            "scam": [
                "congratulations",
                "you have won",
                "claim your prize",
                "nigerian prince",
                "inheritance",
                "lottery",
                "tax refund",
            ],
            "suspicious": [
                "act now",
                "limited time",
                "don't miss out",
                "exclusive offer",
                "risk-free",
                "guaranteed",
            ],
        }

        text_lower = text.lower()
        detected_types = []
        confidence = 0.0

        # Check for fraud indicators
        for fraud_type, indicators in fraud_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    detected_types.append(fraud_type)
                    confidence += 0.2
                    break

        # Check sender domain
        sender = email_data.get("from", "").lower()
        suspicious_domains = ["no-reply", "noreply", "do-not-reply", "notification"]

        for domain in suspicious_domains:
            if domain in sender:
                confidence += 0.1
                if "suspicious_sender" not in detected_types:
                    detected_types.append("suspicious_sender")

        # Check for suspicious URLs
        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, text_lower)

        for url in urls:
            if any(sus in url for sus in ["bit.ly", "tinyurl", "shorturl", "ow.ly"]):
                confidence += 0.15
                if "suspicious_url" not in detected_types:
                    detected_types.append("suspicious_url")

        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)

        # Determine if fraud
        is_fraud = confidence > 0.3

        # Determine risk level
        if confidence > 0.7:
            risk_level = "High"
        elif confidence > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # If fraud detector is available, use it for better analysis
        if self.fraud_detector:
            try:
                detector_result = await self.fraud_detector.detect(text)
                if hasattr(detector_result, "fraud_score"):
                    confidence = max(confidence, detector_result.fraud_score)
                    is_fraud = detector_result.fraud_score > 0.5
                    if detector_result.fraud_types:
                        detected_types.extend(detector_result.fraud_types)
            except Exception as e:
                logger.warning(f"Fraud detector failed, using basic detection: {e}")

        return {
            "is_fraud": is_fraud,
            "confidence": confidence,
            "fraud_types": list(set(detected_types)),
            "risk_level": risk_level,
        }

    def mark_as_spam(self, email_id: str):
        """Mark email as spam."""
        if not self.is_connected:
            raise Exception("Not connected to Gmail")

        try:
            # Move to spam folder
            self.imap.store(email_id, "+X-GM-LABELS", "\\Spam")
            logger.info(f"Marked email {email_id} as spam")
        except Exception as e:
            logger.error(f"Failed to mark as spam: {e}")

    def delete_email(self, email_id: str):
        """Delete email (move to trash)."""
        if not self.is_connected:
            raise Exception("Not connected to Gmail")

        try:
            # Mark for deletion
            self.imap.store(email_id, "+FLAGS", "\\Deleted")
            # Move to trash
            self.imap.store(email_id, "+X-GM-LABELS", "\\Trash")
            logger.info(f"Moved email {email_id} to trash")
        except Exception as e:
            logger.error(f"Failed to delete email: {e}")
