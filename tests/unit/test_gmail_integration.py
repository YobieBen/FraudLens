"""
Unit tests for Gmail integration
Tests Gmail API integration functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call
from datetime import datetime, timedelta
import base64
import json
from typing import List, Dict, Any

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fraudlens.api.gmail_integration import (
    GmailIntegration, EmailMessage, EmailScanResult,
    EmailFilter, BatchEmailProcessor
)


class TestGmailIntegration:
    """Test suite for Gmail integration"""
    
    @pytest.fixture
    def mock_credentials(self):
        """Mock Gmail credentials"""
        mock_creds = Mock()
        mock_creds.valid = True
        mock_creds.expired = False
        mock_creds.refresh_token = "mock_refresh_token"
        return mock_creds
    
    @pytest.fixture
    def mock_service(self):
        """Mock Gmail service"""
        mock_service = Mock()
        
        # Mock messages().list()
        mock_list = Mock()
        mock_list.execute.return_value = {
            'messages': [
                {'id': 'msg1', 'threadId': 'thread1'},
                {'id': 'msg2', 'threadId': 'thread2'}
            ],
            'nextPageToken': None
        }
        
        # Mock messages().get()
        mock_get = Mock()
        mock_get.execute.return_value = {
            'id': 'msg1',
            'threadId': 'thread1',
            'labelIds': ['INBOX'],
            'snippet': 'Test email snippet',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'sender@example.com'},
                    {'name': 'To', 'value': 'recipient@example.com'},
                    {'name': 'Subject', 'value': 'Test Subject'},
                    {'name': 'Date', 'value': 'Mon, 1 Jan 2024 12:00:00 +0000'}
                ],
                'body': {
                    'data': base64.urlsafe_b64encode(b'Test email body').decode()
                }
            }
        }
        
        # Mock attachments().get()
        mock_attachment = Mock()
        mock_attachment.execute.return_value = {
            'data': base64.urlsafe_b64encode(b'attachment content').decode()
        }
        
        # Wire up the mocks
        mock_service.users().messages().list.return_value = mock_list
        mock_service.users().messages().get.return_value = mock_get
        mock_service.users().messages().attachments().get.return_value = mock_attachment
        
        return mock_service
    
    @pytest.fixture
    def gmail_integration(self, mock_credentials, mock_service):
        """Create GmailIntegration instance with mocks"""
        with patch('fraudlens.api.gmail_integration.build') as mock_build:
            mock_build.return_value = mock_service
            integration = GmailIntegration(credentials=mock_credentials)
            integration.service = mock_service
            return integration
    
    def test_initialization(self, mock_credentials):
        """Test GmailIntegration initialization"""
        with patch('fraudlens.api.gmail_integration.build') as mock_build:
            integration = GmailIntegration(credentials=mock_credentials)
            
            assert integration.credentials == mock_credentials
            assert integration.service is not None
            mock_build.assert_called_once()
    
    def test_list_messages(self, gmail_integration):
        """Test listing messages"""
        messages = gmail_integration.list_messages(max_results=10)
        
        assert len(messages) == 2
        assert messages[0]['id'] == 'msg1'
        assert messages[1]['id'] == 'msg2'
        
        gmail_integration.service.users().messages().list.assert_called_with(
            userId='me',
            maxResults=10,
            q=None,
            pageToken=None
        )
    
    def test_list_messages_with_query(self, gmail_integration):
        """Test listing messages with query"""
        query = "from:sender@example.com subject:urgent"
        messages = gmail_integration.list_messages(query=query, max_results=5)
        
        gmail_integration.service.users().messages().list.assert_called_with(
            userId='me',
            maxResults=5,
            q=query,
            pageToken=None
        )
    
    def test_get_message(self, gmail_integration):
        """Test getting a single message"""
        message = gmail_integration.get_message('msg1')
        
        assert message['id'] == 'msg1'
        assert message['threadId'] == 'thread1'
        assert 'payload' in message
        
        gmail_integration.service.users().messages().get.assert_called_with(
            userId='me',
            id='msg1',
            format='full'
        )
    
    def test_parse_message_headers(self, gmail_integration):
        """Test parsing message headers"""
        message = gmail_integration.get_message('msg1')
        headers = gmail_integration._parse_headers(message['payload']['headers'])
        
        assert headers['from'] == 'sender@example.com'
        assert headers['to'] == 'recipient@example.com'
        assert headers['subject'] == 'Test Subject'
        assert headers['date'] == 'Mon, 1 Jan 2024 12:00:00 +0000'
    
    def test_extract_body(self, gmail_integration):
        """Test extracting message body"""
        message = gmail_integration.get_message('msg1')
        body = gmail_integration._extract_body(message['payload'])
        
        assert body == 'Test email body'
    
    @pytest.mark.asyncio
    async def test_scan_email_for_fraud(self, gmail_integration):
        """Test scanning email for fraud"""
        with patch.object(gmail_integration, 'fraud_detector') as mock_detector:
            mock_detector.detect_fraud = AsyncMock(return_value={
                'is_fraud': True,
                'confidence': 0.95,
                'fraud_type': 'phishing',
                'risk_score': 8.5
            })
            
            message_data = {
                'id': 'msg1',
                'payload': {
                    'headers': [
                        {'name': 'Subject', 'value': 'Urgent: Verify your account'}
                    ],
                    'body': {
                        'data': base64.urlsafe_b64encode(
                            b'Click here to verify your account immediately'
                        ).decode()
                    }
                }
            }
            
            result = await gmail_integration.scan_email_for_fraud(message_data)
            
            assert result.message_id == 'msg1'
            assert result.is_fraud == True
            assert result.confidence == 0.95
            assert result.fraud_type == 'phishing'
    
    @pytest.mark.asyncio
    async def test_scan_inbox(self, gmail_integration):
        """Test scanning entire inbox"""
        with patch.object(gmail_integration, 'scan_email_for_fraud') as mock_scan:
            mock_scan.return_value = EmailScanResult(
                message_id='msg1',
                is_fraud=False,
                confidence=0.1,
                fraud_type=None,
                risk_score=1.0
            )
            
            results = await gmail_integration.scan_inbox(max_emails=10)
            
            assert len(results) == 2
            assert all(isinstance(r, EmailScanResult) for r in results)
    
    def test_mark_as_spam(self, gmail_integration):
        """Test marking email as spam"""
        mock_modify = Mock()
        mock_modify.execute.return_value = {'id': 'msg1', 'labelIds': ['SPAM']}
        gmail_integration.service.users().messages().modify.return_value = mock_modify
        
        result = gmail_integration.mark_as_spam('msg1')
        
        assert result == True
        mock_modify.execute.assert_called_once()
    
    def test_move_to_trash(self, gmail_integration):
        """Test moving email to trash"""
        mock_trash = Mock()
        mock_trash.execute.return_value = {'id': 'msg1', 'labelIds': ['TRASH']}
        gmail_integration.service.users().messages().trash.return_value = mock_trash
        
        result = gmail_integration.move_to_trash('msg1')
        
        assert result == True
        mock_trash.execute.assert_called_once()
    
    def test_create_filter(self, gmail_integration):
        """Test creating email filter"""
        email_filter = EmailFilter(
            from_address="spammer@example.com",
            action="delete"
        )
        
        mock_create = Mock()
        mock_create.execute.return_value = {'id': 'filter1'}
        gmail_integration.service.users().settings().filters().create.return_value = mock_create
        
        result = gmail_integration.create_filter(email_filter)
        
        assert result == 'filter1'
        mock_create.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_monitor_inbox_realtime(self, gmail_integration):
        """Test real-time inbox monitoring"""
        scan_count = 0
        
        async def mock_scan_inbox(*args, **kwargs):
            nonlocal scan_count
            scan_count += 1
            if scan_count > 2:
                # Stop after 2 iterations
                gmail_integration.monitoring = False
            return []
        
        with patch.object(gmail_integration, 'scan_inbox', side_effect=mock_scan_inbox):
            # Run monitor for a short time
            monitor_task = asyncio.create_task(
                gmail_integration.monitor_inbox_realtime(check_interval=0.1)
            )
            
            await asyncio.sleep(0.5)
            gmail_integration.stop_monitoring()
            
            try:
                await asyncio.wait_for(monitor_task, timeout=1.0)
            except asyncio.TimeoutError:
                pass
            
            assert scan_count >= 2


class TestEmailMessage:
    """Test EmailMessage data class"""
    
    def test_email_message_creation(self):
        """Test creating EmailMessage instance"""
        message = EmailMessage(
            id='msg123',
            thread_id='thread123',
            from_address='sender@example.com',
            to_addresses=['recipient@example.com'],
            subject='Test Subject',
            body='Test body',
            date=datetime.now(),
            labels=['INBOX'],
            attachments=[]
        )
        
        assert message.id == 'msg123'
        assert message.from_address == 'sender@example.com'
        assert message.subject == 'Test Subject'
        assert len(message.to_addresses) == 1
    
    def test_email_message_to_dict(self):
        """Test converting EmailMessage to dictionary"""
        message = EmailMessage(
            id='msg123',
            thread_id='thread123',
            from_address='sender@example.com',
            to_addresses=['recipient@example.com'],
            subject='Test Subject',
            body='Test body',
            date=datetime.now(),
            labels=['INBOX'],
            attachments=[]
        )
        
        message_dict = message.to_dict()
        
        assert message_dict['id'] == 'msg123'
        assert message_dict['from_address'] == 'sender@example.com'
        assert 'date' in message_dict
        assert isinstance(message_dict['date'], str)


class TestBatchEmailProcessor:
    """Test BatchEmailProcessor"""
    
    @pytest.fixture
    def batch_processor(self):
        """Create BatchEmailProcessor instance"""
        return BatchEmailProcessor(batch_size=5, max_workers=2)
    
    @pytest.mark.asyncio
    async def test_process_batch(self, batch_processor):
        """Test processing email batch"""
        emails = [
            EmailMessage(
                id=f'msg{i}',
                thread_id=f'thread{i}',
                from_address=f'sender{i}@example.com',
                to_addresses=[f'recipient{i}@example.com'],
                subject=f'Subject {i}',
                body=f'Body {i}',
                date=datetime.now(),
                labels=['INBOX'],
                attachments=[]
            )
            for i in range(10)
        ]
        
        async def mock_process(email):
            return EmailScanResult(
                message_id=email.id,
                is_fraud=False,
                confidence=0.1,
                fraud_type=None,
                risk_score=1.0
            )
        
        results = await batch_processor.process_batch(emails, mock_process)
        
        assert len(results) == 10
        assert all(isinstance(r, EmailScanResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_process_with_errors(self, batch_processor):
        """Test batch processing with errors"""
        emails = [
            EmailMessage(
                id=f'msg{i}',
                thread_id=f'thread{i}',
                from_address=f'sender{i}@example.com',
                to_addresses=[f'recipient{i}@example.com'],
                subject=f'Subject {i}',
                body=f'Body {i}',
                date=datetime.now(),
                labels=['INBOX'],
                attachments=[]
            )
            for i in range(3)
        ]
        
        async def mock_process(email):
            if email.id == 'msg1':
                raise ValueError("Processing error")
            return EmailScanResult(
                message_id=email.id,
                is_fraud=False,
                confidence=0.1,
                fraud_type=None,
                risk_score=1.0
            )
        
        with patch('fraudlens.api.gmail_integration.logger') as mock_logger:
            results = await batch_processor.process_batch(emails, mock_process)
            
            # Should still process other emails despite error
            assert len(results) == 2
            mock_logger.error.assert_called()


class TestEmailFilter:
    """Test EmailFilter functionality"""
    
    def test_email_filter_creation(self):
        """Test creating EmailFilter"""
        filter = EmailFilter(
            from_address="spammer@example.com",
            subject_contains="URGENT",
            action="delete"
        )
        
        assert filter.from_address == "spammer@example.com"
        assert filter.subject_contains == "URGENT"
        assert filter.action == "delete"
    
    def test_email_filter_to_gmail_format(self):
        """Test converting filter to Gmail API format"""
        filter = EmailFilter(
            from_address="spammer@example.com",
            has_attachment=True,
            action="trash"
        )
        
        gmail_filter = filter.to_gmail_format()
        
        assert 'criteria' in gmail_filter
        assert 'action' in gmail_filter
        assert gmail_filter['criteria']['from'] == "spammer@example.com"
        assert gmail_filter['criteria']['hasAttachment'] == True
        assert gmail_filter['action']['addLabelIds'] == ['TRASH']


class TestGmailIntegrationEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def gmail_integration_with_errors(self):
        """Create GmailIntegration that raises errors"""
        integration = Mock(spec=GmailIntegration)
        integration.service = Mock()
        return integration
    
    def test_handle_api_errors(self, gmail_integration_with_errors):
        """Test handling Gmail API errors"""
        from googleapiclient.errors import HttpError
        
        # Simulate API error
        error_resp = Mock()
        error_resp.status = 429
        error_resp.reason = "Too Many Requests"
        
        gmail_integration_with_errors.service.users().messages().list().execute.side_effect = \
            HttpError(error_resp, b'Rate limit exceeded')
        
        with pytest.raises(HttpError):
            real_integration = GmailIntegration(credentials=Mock())
            real_integration.service = gmail_integration_with_errors.service
            real_integration.list_messages()
    
    def test_handle_invalid_message_format(self):
        """Test handling invalid message format"""
        integration = GmailIntegration(credentials=Mock())
        
        # Invalid message without payload
        invalid_message = {'id': 'msg1'}
        
        body = integration._extract_body(invalid_message.get('payload', {}))
        assert body == ""
    
    def test_handle_missing_headers(self):
        """Test handling missing headers"""
        integration = GmailIntegration(credentials=Mock())
        
        headers = integration._parse_headers([])
        assert headers == {}
        
        headers = integration._parse_headers(None)
        assert headers == {}
    
    @pytest.mark.asyncio
    async def test_concurrent_scanning_limit(self):
        """Test concurrent scanning with semaphore limit"""
        integration = GmailIntegration(credentials=Mock())
        integration.concurrent_limit = 2
        
        scan_times = []
        
        async def mock_scan(msg):
            scan_times.append(datetime.now())
            await asyncio.sleep(0.1)
            return EmailScanResult(
                message_id=msg['id'],
                is_fraud=False,
                confidence=0.1,
                fraud_type=None,
                risk_score=1.0
            )
        
        messages = [{'id': f'msg{i}'} for i in range(5)]
        
        with patch.object(integration, 'scan_email_for_fraud', side_effect=mock_scan):
            results = await integration._scan_messages_async(messages)
            
            # Check that no more than 2 scans happened simultaneously
            for i in range(len(scan_times) - 2):
                time_diff = (scan_times[i + 2] - scan_times[i]).total_seconds()
                assert time_diff >= 0.09  # Allow small timing variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])