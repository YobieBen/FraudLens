"""
Unit tests for Gmail integration
Tests Gmail API integration functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime
import base64
import json
from typing import Dict, Any

# Add parent directory to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fraudlens.api.gmail_integration import GmailFraudScanner, EmailAnalysisResult, EmailAction
from dataclasses import asdict


class TestGmailFraudScanner:
    """Test suite for Gmail fraud scanner"""

    @pytest.fixture
    def mock_credentials(self):
        """Mock Gmail credentials"""
        with patch("fraudlens.api.gmail_integration.Credentials") as mock_creds:
            yield mock_creds

    @pytest.fixture
    def mock_service(self):
        """Mock Gmail service"""
        with patch("fraudlens.api.gmail_integration.build") as mock_build:
            mock_service = MagicMock()
            mock_build.return_value = mock_service
            yield mock_service

    @pytest.fixture
    def mock_pipeline(self):
        """Mock fraud detection pipeline"""
        with patch("fraudlens.api.gmail_integration.FraudDetectionPipeline") as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_pipeline.process.return_value = {
                "fraud_score": 0.3,
                "fraud_types": [],
                "confidence": 0.9,
                "risk_indicators": [],
                "analysis_text": "Low risk email",
            }
            yield mock_pipeline

    @pytest.fixture
    async def scanner(self, mock_credentials, mock_service, mock_pipeline):
        """Create a GmailFraudScanner instance"""
        scanner = GmailFraudScanner()
        # Mock authentication and labels
        with patch.object(scanner, "_authenticate"):
            with patch.object(scanner, "_create_fraud_labels"):
                scanner.service = mock_service  # Set service before init
                await scanner.initialize()
        return scanner

    @pytest.mark.asyncio
    async def test_initialization(self, mock_credentials, mock_service):
        """Test scanner initialization"""
        scanner = GmailFraudScanner()
        # Mock the service and authentication
        with patch.object(scanner, "_authenticate") as mock_auth:
            with patch.object(scanner, "_create_fraud_labels") as mock_labels:
                # Set the service before initialization
                scanner.service = mock_service
                await scanner.initialize()

        assert scanner.service is not None
        assert scanner.pipeline is not None

    @pytest.mark.asyncio
    async def test_process_email(self, mock_service, mock_pipeline):
        """Test processing a single email"""
        scanner = GmailFraudScanner()
        scanner.service = mock_service
        scanner.pipeline = mock_pipeline

        # Mock pipeline process method
        mock_pipeline.process.return_value = AsyncMock()
        mock_pipeline.process.return_value.fraud_score = 0.3
        mock_pipeline.process.return_value.fraud_types = []
        mock_pipeline.process.return_value.confidence = 0.9
        mock_pipeline.process.return_value.explanation = "Safe email"

        # Mock email message
        mock_message = {
            "id": "test_message_id",
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Test Email"},
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "recipient@example.com"},
                    {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
                ],
                "body": {"data": base64.urlsafe_b64encode(b"Test email body").decode()},
            },
        }

        mock_service.users().messages().get().execute.return_value = mock_message

        # Mock label operations
        mock_service.users().messages().modify().execute.return_value = {}

        result = await scanner.process_email("test_message_id")

        assert isinstance(result, EmailAnalysisResult)
        assert result.message_id == "test_message_id"
        assert result.subject == "Test Email"
        assert result.sender == "sender@example.com"

    @pytest.mark.asyncio
    async def test_stream_emails(self, scanner, mock_service):
        """Test streaming emails from inbox"""
        # Mock list messages response
        mock_list_response = {"messages": [{"id": "msg1"}, {"id": "msg2"}]}
        mock_service.users().messages().list().execute.return_value = mock_list_response

        # Mock process_email to return valid results
        with patch.object(scanner, "process_email") as mock_process:
            mock_process.return_value = EmailAnalysisResult(
                message_id="msg1",
                subject="Test Email",
                sender="sender@example.com",
                recipient="recipient@example.com",
                date=datetime.now(),
                fraud_score=0.1,
                fraud_types=[],
                confidence=0.9,
                explanation="Safe email",
                attachments_analyzed=[],
                action_taken=EmailAction.NONE,
                processing_time_ms=100,
                raw_content_score=0.1,
                attachment_scores=[],
                combined_score=0.1,
                flagged=False,
                error=None,
            )

            results = await scanner.stream_emails(max_results=2)
            assert isinstance(results, list)
            assert len(results) <= 2

    def test_determine_action(self):
        """Test action determination based on fraud score"""
        scanner = GmailFraudScanner()

        assert scanner._determine_action(0.1) == EmailAction.NONE
        assert scanner._determine_action(0.6) == EmailAction.FLAG
        assert scanner._determine_action(0.75) == EmailAction.SPAM
        assert scanner._determine_action(0.95) == EmailAction.TRASH

    def test_extract_body(self):
        """Test email body extraction"""
        scanner = GmailFraudScanner()

        # Test plain text body
        payload = {"body": {"data": base64.urlsafe_b64encode(b"Test body").decode()}}
        body = scanner._extract_body(payload)
        assert body == "Test body"

        # Test multipart body
        payload = {
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": base64.urlsafe_b64encode(b"Plain text").decode()},
                },
                {
                    "mimeType": "text/html",
                    "body": {"data": base64.urlsafe_b64encode(b"<html>HTML text</html>").decode()},
                },
            ]
        }
        body = scanner._extract_body(payload)
        assert "Plain text" in body or "HTML text" in body

    def test_get_modality_from_mime(self):
        """Test MIME type to modality mapping"""
        scanner = GmailFraudScanner()

        assert scanner._get_modality_from_mime("image/jpeg") == "image"
        assert scanner._get_modality_from_mime("text/plain") == "text"
        assert scanner._get_modality_from_mime("application/pdf") == "document"
        assert scanner._get_modality_from_mime("video/mp4") == "video"
        assert scanner._get_modality_from_mime("unknown/type") == "text"

    @pytest.mark.asyncio
    async def test_bulk_process(self, scanner, mock_service):
        """Test bulk email processing"""
        # Mock stream_emails method instead
        with patch.object(scanner, "stream_emails") as mock_stream:
            mock_stream.return_value = [
                EmailAnalysisResult(
                    message_id="msg1",
                    subject="Test",
                    sender="test@example.com",
                    recipient="user@example.com",
                    date=datetime.now(),
                    fraud_score=0.1,
                    fraud_types=[],
                    confidence=0.9,
                    explanation="Safe email",
                    attachments_analyzed=[],
                    action_taken=EmailAction.NONE,
                    processing_time_ms=100,
                    raw_content_score=0.1,
                    attachment_scores=[],
                    combined_score=0.1,
                    flagged=False,
                    error=None,
                )
            ]

            results = await scanner.bulk_process(["is:unread"])

        assert isinstance(results, dict)
        assert "is:unread" in results

    def test_get_statistics(self):
        """Test statistics retrieval"""
        scanner = GmailFraudScanner()
        # Set up stats directly
        scanner.stats["total_processed"] = 10
        scanner.stats["fraud_detected"] = 2
        scanner.stats["processing_time_total"] = 1000

        stats = scanner.get_statistics()

        assert stats["total_processed"] == 10
        assert stats["fraud_detected"] == 2
        assert "fraud_rate" in stats

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in email processing"""
        scanner = GmailFraudScanner()
        scanner.service = MagicMock()
        scanner.service.users().messages().get().execute.side_effect = Exception("API Error")

        # Initialize pipeline to avoid NoneType errors
        scanner.pipeline = MagicMock()

        result = await scanner.process_email("error_message")

        # Should handle error gracefully and return an error result
        assert isinstance(result, EmailAnalysisResult)
        assert result.error == "API Error"


class TestEmailAnalysisResult:
    """Test EmailAnalysisResult dataclass"""

    def test_creation(self):
        """Test creating an EmailAnalysisResult"""
        result = EmailAnalysisResult(
            message_id="test_id",
            subject="Test Subject",
            sender="sender@example.com",
            recipient="recipient@example.com",
            date=datetime.now(),
            fraud_score=0.5,
            fraud_types=["phishing"],
            confidence=0.9,
            explanation="Potential phishing email",
            attachments_analyzed=[],
            action_taken=EmailAction.FLAG,
            processing_time_ms=100.0,
            raw_content_score=0.5,
            attachment_scores=[],
            combined_score=0.5,
            flagged=True,
            error=None,
        )

        assert result.message_id == "test_id"
        assert result.fraud_score == 0.5
        assert result.action_taken == EmailAction.FLAG

    def test_to_dict(self):
        """Test converting EmailAnalysisResult to dictionary"""
        result = EmailAnalysisResult(
            message_id="test_id",
            subject="Test Subject",
            sender="sender@example.com",
            recipient="recipient@example.com",
            date=datetime.now(),
            fraud_score=0.5,
            fraud_types=["phishing"],
            confidence=0.9,
            explanation="Potential phishing email",
            attachments_analyzed=[],
            action_taken=EmailAction.FLAG,
            processing_time_ms=100.0,
            raw_content_score=0.5,
            attachment_scores=[],
            combined_score=0.5,
            flagged=True,
            error=None,
        )

        result_dict = asdict(result)

        assert isinstance(result_dict, dict)
        assert result_dict["message_id"] == "test_id"
        assert result_dict["fraud_score"] == 0.5


class TestEmailAction:
    """Test EmailAction enum"""

    def test_action_values(self):
        """Test EmailAction enum values"""
        assert EmailAction.NONE.value == "none"
        assert EmailAction.FLAG.value == "flag"
        assert EmailAction.SPAM.value == "spam"
        assert EmailAction.TRASH.value == "trash"
        assert EmailAction.QUARANTINE.value == "quarantine"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
