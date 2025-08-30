"""
Gmail Integration for FraudLens
Processes emails with attachments for fraud detection
"""

import asyncio
import base64
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import mimetypes
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle

from loguru import logger
from dataclasses import dataclass, asdict
from enum import Enum

from ..core.pipeline import FraudDetectionPipeline
from ..core.base.detector import FraudType


class EmailAction(Enum):
    """Actions to take on fraudulent emails."""
    NONE = "none"
    FLAG = "flag"
    SPAM = "spam"
    TRASH = "trash"
    QUARANTINE = "quarantine"


@dataclass
class EmailAnalysisResult:
    """Result of email fraud analysis."""
    message_id: str
    subject: str
    sender: str
    recipient: str
    date: datetime
    fraud_score: float
    fraud_types: List[str]
    confidence: float
    explanation: str
    attachments_analyzed: List[Dict[str, Any]]
    action_taken: EmailAction
    processing_time_ms: float
    raw_content_score: float
    attachment_scores: List[float]
    combined_score: float
    flagged: bool
    error: Optional[str] = None


class GmailFraudScanner:
    """Gmail integration for fraud detection."""
    
    # Gmail API scopes
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.modify',
        'https://www.googleapis.com/auth/gmail.labels',
    ]
    
    def __init__(
        self,
        credentials_file: str = "credentials.json",
        token_file: str = "token.pickle",
        fraud_threshold: float = 0.7,
        auto_action: bool = False,
        action_threshold: Dict[EmailAction, float] = None,
    ):
        """
        Initialize Gmail fraud scanner.
        
        Args:
            credentials_file: Path to OAuth2 credentials
            token_file: Path to store token
            fraud_threshold: Threshold for flagging as fraud
            auto_action: Whether to automatically take actions
            action_threshold: Thresholds for different actions
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.fraud_threshold = fraud_threshold
        self.auto_action = auto_action
        self.action_threshold = action_threshold or {
            EmailAction.FLAG: 0.5,
            EmailAction.SPAM: 0.7,
            EmailAction.TRASH: 0.9,
            EmailAction.QUARANTINE: 0.8,
        }
        
        self.service = None
        self.pipeline = FraudDetectionPipeline()
        self.labels = {}
        self.stats = {
            "total_processed": 0,
            "fraud_detected": 0,
            "attachments_processed": 0,
            "actions_taken": {},
            "processing_time_total": 0,
        }
        
    async def initialize(self):
        """Initialize Gmail service and fraud detection pipeline."""
        logger.info("Initializing Gmail Fraud Scanner...")
        
        # Initialize fraud detection pipeline
        await self.pipeline.initialize()
        
        # Authenticate Gmail
        self._authenticate()
        
        # Create custom labels
        await self._create_fraud_labels()
        
        logger.info("Gmail Fraud Scanner initialized successfully")
        
    def _authenticate(self):
        """Authenticate with Gmail API."""
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"Credentials file not found: {self.credentials_file}\n"
                        "Please download OAuth2 credentials from Google Cloud Console"
                    )
                    
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail authentication successful")
        
    async def _create_fraud_labels(self):
        """Create custom labels for fraud detection."""
        try:
            # Get existing labels
            results = self.service.users().labels().list(userId='me').execute()
            existing_labels = {label['name']: label['id'] for label in results.get('labels', [])}
            
            # Labels to create
            fraud_labels = [
                "FraudLens/Analyzed",
                "FraudLens/Suspicious",
                "FraudLens/Fraud",
                "FraudLens/Quarantine",
                "FraudLens/Safe",
            ]
            
            for label_name in fraud_labels:
                if label_name not in existing_labels:
                    label_object = {
                        'name': label_name,
                        'labelListVisibility': 'labelShow',
                        'messageListVisibility': 'show',
                    }
                    
                    created_label = self.service.users().labels().create(
                        userId='me',
                        body=label_object
                    ).execute()
                    
                    self.labels[label_name] = created_label['id']
                    logger.info(f"Created label: {label_name}")
                else:
                    self.labels[label_name] = existing_labels[label_name]
                    
        except HttpError as error:
            logger.error(f"Error creating labels: {error}")
            
    async def stream_emails(
        self,
        query: str = "is:unread",
        max_results: int = 100,
        process_attachments: bool = True,
        since_days: int = 7,
    ) -> List[EmailAnalysisResult]:
        """
        Stream and process emails from Gmail.
        
        Args:
            query: Gmail search query
            max_results: Maximum emails to process
            process_attachments: Whether to process attachments
            since_days: Process emails from last N days
            
        Returns:
            List of analysis results
        """
        logger.info(f"Streaming emails with query: {query}")
        
        # Add date filter to query
        date_filter = (datetime.now() - timedelta(days=since_days)).strftime('%Y/%m/%d')
        full_query = f"{query} after:{date_filter}"
        
        try:
            # Get message list
            results = self.service.users().messages().list(
                userId='me',
                q=full_query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            logger.info(f"Found {len(messages)} emails to process")
            
            # Process emails
            analysis_results = []
            for message in messages:
                result = await self.process_email(
                    message['id'],
                    process_attachments=process_attachments
                )
                analysis_results.append(result)
                
                # Take action if configured
                if self.auto_action and result.fraud_score > self.fraud_threshold:
                    await self._take_action(result)
            
            return analysis_results
            
        except HttpError as error:
            logger.error(f"Error streaming emails: {error}")
            return []
            
    async def process_email(
        self,
        message_id: str,
        process_attachments: bool = True,
    ) -> EmailAnalysisResult:
        """
        Process a single email for fraud detection.
        
        Args:
            message_id: Gmail message ID
            process_attachments: Whether to process attachments
            
        Returns:
            Analysis result
        """
        start_time = datetime.now()
        
        try:
            # Get message details
            message = self.service.users().messages().get(
                userId='me',
                id=message_id
            ).execute()
            
            # Extract email metadata
            headers = message['payload'].get('headers', [])
            subject = self._get_header(headers, 'Subject')
            sender = self._get_header(headers, 'From')
            recipient = self._get_header(headers, 'To')
            date_str = self._get_header(headers, 'Date')
            
            # Extract body
            body = self._extract_body(message['payload'])
            
            # Analyze email content
            content_result = await self.pipeline.process(
                f"Subject: {subject}\n\nFrom: {sender}\n\n{body}",
                modality="text"
            )
            
            content_score = content_result.fraud_score if content_result else 0.0
            
            # Process attachments if requested
            attachment_scores = []
            attachments_analyzed = []
            
            if process_attachments:
                attachments = self._extract_attachments(message['payload'], message_id)
                for attachment in attachments:
                    att_result = await self._process_attachment(attachment)
                    if att_result:
                        attachment_scores.append(att_result['fraud_score'])
                        attachments_analyzed.append(att_result)
            
            # Calculate combined score
            all_scores = [content_score] + attachment_scores
            combined_score = max(all_scores) if all_scores else 0.0
            
            # Determine fraud types
            fraud_types = []
            if content_result and content_result.fraud_types:
                fraud_types = [str(ft) for ft in content_result.fraud_types]
            
            # Determine action
            action = self._determine_action(combined_score)
            
            # Apply label
            await self._label_email(message_id, combined_score)
            
            # Update stats
            self.stats["total_processed"] += 1
            if combined_score > self.fraud_threshold:
                self.stats["fraud_detected"] += 1
            self.stats["attachments_processed"] += len(attachments_analyzed)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return EmailAnalysisResult(
                message_id=message_id,
                subject=subject,
                sender=sender,
                recipient=recipient,
                date=self._parse_date(date_str),
                fraud_score=combined_score,
                fraud_types=fraud_types,
                confidence=content_result.confidence if content_result else 0.0,
                explanation=content_result.explanation if content_result else "No analysis available",
                attachments_analyzed=attachments_analyzed,
                action_taken=action,
                processing_time_ms=processing_time,
                raw_content_score=content_score,
                attachment_scores=attachment_scores,
                combined_score=combined_score,
                flagged=combined_score > self.fraud_threshold,
                error=None,
            )
            
        except Exception as e:
            logger.error(f"Error processing email {message_id}: {e}")
            return EmailAnalysisResult(
                message_id=message_id,
                subject="Error",
                sender="Unknown",
                recipient="Unknown",
                date=datetime.now(),
                fraud_score=0.0,
                fraud_types=[],
                confidence=0.0,
                explanation=f"Error processing email: {str(e)}",
                attachments_analyzed=[],
                action_taken=EmailAction.NONE,
                processing_time_ms=0.0,
                raw_content_score=0.0,
                attachment_scores=[],
                combined_score=0.0,
                flagged=False,
                error=str(e),
            )
            
    def _extract_body(self, payload: Dict) -> str:
        """Extract email body from payload."""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data', '')
                    if data:
                        body += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                elif part['mimeType'] == 'text/html' and not body:
                    data = part['body'].get('data', '')
                    if data:
                        # Basic HTML stripping
                        import re
                        html = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                        body = re.sub('<[^<]+?>', '', html)
        else:
            # Single part message
            data = payload['body'].get('data', '')
            if data:
                body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        
        return body
        
    def _extract_attachments(self, payload: Dict, message_id: str) -> List[Dict]:
        """Extract attachments from email."""
        attachments = []
        
        def extract_parts(parts):
            for part in parts:
                if part.get('filename'):
                    attachment = {
                        'filename': part['filename'],
                        'mime_type': part['mimeType'],
                        'size': part['body'].get('size', 0),
                        'attachment_id': part['body'].get('attachmentId'),
                        'message_id': message_id,
                    }
                    attachments.append(attachment)
                    
                if 'parts' in part:
                    extract_parts(part['parts'])
        
        if 'parts' in payload:
            extract_parts(payload['parts'])
            
        return attachments
        
    async def _process_attachment(self, attachment: Dict) -> Optional[Dict]:
        """Process an attachment for fraud detection."""
        try:
            # Download attachment
            att_data = self.service.users().messages().attachments().get(
                userId='me',
                messageId=attachment['message_id'],
                id=attachment['attachment_id']
            ).execute()
            
            data = base64.urlsafe_b64decode(att_data['data'])
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(attachment['filename']).suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            
            # Determine modality based on mime type
            mime_type = attachment['mime_type']
            modality = self._get_modality_from_mime(mime_type)
            
            # Process with pipeline
            result = await self.pipeline.process(tmp_path, modality=modality)
            
            # Clean up
            os.unlink(tmp_path)
            
            if result:
                return {
                    'filename': attachment['filename'],
                    'mime_type': mime_type,
                    'size': attachment['size'],
                    'fraud_score': result.fraud_score,
                    'fraud_types': [str(ft) for ft in result.fraud_types] if result.fraud_types else [],
                    'confidence': result.confidence,
                    'explanation': result.explanation,
                }
            
        except Exception as e:
            logger.error(f"Error processing attachment {attachment['filename']}: {e}")
            
        return None
        
    def _get_modality_from_mime(self, mime_type: str) -> str:
        """Determine processing modality from MIME type."""
        if mime_type.startswith('image/'):
            return 'image'
        elif mime_type.startswith('video/'):
            return 'video'
        elif mime_type in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument']:
            return 'document'
        else:
            return 'text'
            
    def _determine_action(self, fraud_score: float) -> EmailAction:
        """Determine action based on fraud score."""
        if fraud_score >= self.action_threshold[EmailAction.TRASH]:
            return EmailAction.TRASH
        elif fraud_score >= self.action_threshold[EmailAction.QUARANTINE]:
            return EmailAction.QUARANTINE
        elif fraud_score >= self.action_threshold[EmailAction.SPAM]:
            return EmailAction.SPAM
        elif fraud_score >= self.action_threshold[EmailAction.FLAG]:
            return EmailAction.FLAG
        else:
            return EmailAction.NONE
            
    async def _label_email(self, message_id: str, fraud_score: float):
        """Apply labels to email based on fraud score."""
        try:
            # Determine label
            if fraud_score >= 0.8:
                label = "FraudLens/Fraud"
            elif fraud_score >= 0.5:
                label = "FraudLens/Suspicious"
            else:
                label = "FraudLens/Safe"
            
            # Add analyzed label
            labels_to_add = [
                self.labels.get("FraudLens/Analyzed"),
                self.labels.get(label),
            ]
            
            # Apply labels
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'addLabelIds': [l for l in labels_to_add if l]}
            ).execute()
            
        except Exception as e:
            logger.error(f"Error labeling email: {e}")
            
    async def _take_action(self, result: EmailAnalysisResult):
        """Take action on fraudulent email."""
        try:
            if result.action_taken == EmailAction.TRASH:
                # Move to trash
                self.service.users().messages().trash(
                    userId='me',
                    id=result.message_id
                ).execute()
                logger.info(f"Moved email {result.message_id} to trash")
                
            elif result.action_taken == EmailAction.SPAM:
                # Move to spam
                self.service.users().messages().modify(
                    userId='me',
                    id=result.message_id,
                    body={'addLabelIds': ['SPAM']}
                ).execute()
                logger.info(f"Marked email {result.message_id} as spam")
                
            elif result.action_taken == EmailAction.QUARANTINE:
                # Move to quarantine label
                quarantine_label = self.labels.get("FraudLens/Quarantine")
                if quarantine_label:
                    self.service.users().messages().modify(
                        userId='me',
                        id=result.message_id,
                        body={
                            'addLabelIds': [quarantine_label],
                            'removeLabelIds': ['INBOX']
                        }
                    ).execute()
                    logger.info(f"Quarantined email {result.message_id}")
                    
            # Update stats
            action_name = result.action_taken.value
            self.stats["actions_taken"][action_name] = self.stats["actions_taken"].get(action_name, 0) + 1
            
        except Exception as e:
            logger.error(f"Error taking action on email: {e}")
            
    def _get_header(self, headers: List[Dict], name: str) -> str:
        """Get header value from headers list."""
        for header in headers:
            if header['name'] == name:
                return header['value']
        return ""
        
    def _parse_date(self, date_str: str) -> datetime:
        """Parse email date string."""
        from email.utils import parsedate_to_datetime
        try:
            return parsedate_to_datetime(date_str)
        except:
            return datetime.now()
            
    async def bulk_process(
        self,
        queries: List[str],
        parallel: bool = True,
        max_workers: int = 5,
    ) -> Dict[str, List[EmailAnalysisResult]]:
        """
        Process multiple email queries in bulk.
        
        Args:
            queries: List of Gmail queries
            parallel: Whether to process in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            Dictionary of results by query
        """
        results = {}
        
        if parallel:
            # Process queries in parallel
            tasks = []
            for query in queries:
                task = asyncio.create_task(self.stream_emails(query))
                tasks.append((query, task))
            
            for query, task in tasks:
                results[query] = await task
        else:
            # Process sequentially
            for query in queries:
                results[query] = await self.stream_emails(query)
        
        return results
        
    async def monitor_inbox(
        self,
        interval_seconds: int = 60,
        query: str = "is:unread",
    ):
        """
        Continuously monitor inbox for new emails.
        
        Args:
            interval_seconds: Check interval
            query: Gmail query for monitoring
        """
        logger.info(f"Starting inbox monitoring (interval: {interval_seconds}s)")
        
        processed_ids = set()
        
        while True:
            try:
                # Get recent emails
                results = await self.stream_emails(query, max_results=50)
                
                # Process only new emails
                for result in results:
                    if result.message_id not in processed_ids:
                        processed_ids.add(result.message_id)
                        
                        if result.fraud_score > self.fraud_threshold:
                            logger.warning(
                                f"Fraud detected in email from {result.sender}: "
                                f"Score={result.fraud_score:.2%}, Types={result.fraud_types}"
                            )
                            
                            if self.auto_action:
                                await self._take_action(result)
                
                # Clean up old IDs to prevent memory growth
                if len(processed_ids) > 1000:
                    processed_ids = set(list(processed_ids)[-500:])
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            "avg_processing_time_ms": (
                self.stats["processing_time_total"] / self.stats["total_processed"]
                if self.stats["total_processed"] > 0 else 0
            ),
            "fraud_rate": (
                self.stats["fraud_detected"] / self.stats["total_processed"]
                if self.stats["total_processed"] > 0 else 0
            ),
        }


async def main():
    """Example usage."""
    scanner = GmailFraudScanner(
        fraud_threshold=0.6,
        auto_action=True,
        action_threshold={
            EmailAction.FLAG: 0.5,
            EmailAction.SPAM: 0.7,
            EmailAction.TRASH: 0.95,
        }
    )
    
    await scanner.initialize()
    
    # Process unread emails
    results = await scanner.stream_emails(
        query="is:unread",
        max_results=10,
        process_attachments=True
    )
    
    # Print results
    for result in results:
        print(f"\nEmail: {result.subject}")
        print(f"  From: {result.sender}")
        print(f"  Fraud Score: {result.fraud_score:.2%}")
        print(f"  Action: {result.action_taken.value}")
        
        if result.attachments_analyzed:
            print(f"  Attachments: {len(result.attachments_analyzed)}")
            for att in result.attachments_analyzed:
                print(f"    - {att['filename']}: {att['fraud_score']:.2%}")
    
    # Print statistics
    print(f"\nStatistics: {json.dumps(scanner.get_statistics(), indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())