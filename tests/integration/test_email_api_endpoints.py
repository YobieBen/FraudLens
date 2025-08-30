"""
Integration tests for email API endpoints
Tests the complete email fraud detection API flow
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
import base64
from datetime import datetime
from typing import Dict, List, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fraudlens.api.secured_api import app
from fraudlens.api.auth import create_tokens, UserInDB, UserRole
from fraudlens.api.gmail_integration import EmailMessage, EmailScanResult


class TestEmailAPIEndpoints:
    """Integration tests for email API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authenticated headers"""
        # Create test user
        test_user = UserInDB(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
            role=UserRole.USER,
            is_active=True,
            created_at=datetime.now()
        )
        
        # Generate token
        tokens = create_tokens(test_user)
        
        return {
            "Authorization": f"Bearer {tokens.access_token}"
        }
    
    @pytest.fixture
    def admin_headers(self):
        """Create admin authenticated headers"""
        admin_user = UserInDB(
            id=2,
            username="admin",
            email="admin@example.com",
            hashed_password="hashed",
            role=UserRole.ADMIN,
            is_active=True,
            created_at=datetime.now()
        )
        
        tokens = create_tokens(admin_user)
        
        return {
            "Authorization": f"Bearer {tokens.access_token}"
        }
    
    @pytest.fixture
    def mock_gmail_service(self):
        """Mock Gmail service"""
        with patch('fraudlens.api.gmail_integration.build') as mock_build:
            mock_service = Mock()
            
            # Mock list messages
            mock_service.users().messages().list().execute.return_value = {
                'messages': [
                    {'id': 'msg1', 'threadId': 'thread1'},
                    {'id': 'msg2', 'threadId': 'thread2'}
                ],
                'nextPageToken': None
            }
            
            # Mock get message
            mock_service.users().messages().get().execute.return_value = {
                'id': 'msg1',
                'threadId': 'thread1',
                'payload': {
                    'headers': [
                        {'name': 'From', 'value': 'sender@example.com'},
                        {'name': 'Subject', 'value': 'Test Email'}
                    ],
                    'body': {
                        'data': base64.urlsafe_b64encode(b'Test email content').decode()
                    }
                }
            }
            
            mock_build.return_value = mock_service
            yield mock_service
    
    def test_scan_inbox_endpoint(self, client, auth_headers, mock_gmail_service):
        """Test /api/email/scan-inbox endpoint"""
        with patch('fraudlens.api.secured_api.GmailIntegration') as MockGmail:
            mock_integration = Mock()
            MockGmail.return_value = mock_integration
            
            # Mock scan results
            mock_integration.scan_inbox = AsyncMock(return_value=[
                EmailScanResult(
                    message_id='msg1',
                    is_fraud=True,
                    confidence=0.95,
                    fraud_type='phishing',
                    risk_score=9.0,
                    subject='Urgent: Verify Account',
                    from_address='scammer@fake.com'
                ),
                EmailScanResult(
                    message_id='msg2',
                    is_fraud=False,
                    confidence=0.1,
                    fraud_type=None,
                    risk_score=1.0,
                    subject='Newsletter',
                    from_address='news@legitimate.com'
                )
            ])
            
            response = client.post(
                "/api/email/scan-inbox",
                headers=auth_headers,
                json={"max_emails": 10}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert 'results' in data
            assert len(data['results']) == 2
            assert data['results'][0]['is_fraud'] == True
            assert data['results'][0]['fraud_type'] == 'phishing'
    
    def test_scan_specific_email(self, client, auth_headers, mock_gmail_service):
        """Test /api/email/scan/{message_id} endpoint"""
        with patch('fraudlens.api.secured_api.GmailIntegration') as MockGmail:
            mock_integration = Mock()
            MockGmail.return_value = mock_integration
            
            mock_integration.get_message.return_value = {
                'id': 'msg1',
                'payload': {
                    'headers': [
                        {'name': 'Subject', 'value': 'Win $1000000!!!'}
                    ],
                    'body': {
                        'data': base64.urlsafe_b64encode(
                            b'Click here to claim your prize!'
                        ).decode()
                    }
                }
            }
            
            mock_integration.scan_email_for_fraud = AsyncMock(return_value=
                EmailScanResult(
                    message_id='msg1',
                    is_fraud=True,
                    confidence=0.99,
                    fraud_type='scam',
                    risk_score=10.0,
                    subject='Win $1000000!!!',
                    from_address='scammer@fake.com'
                )
            )
            
            response = client.get(
                "/api/email/scan/msg1",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['message_id'] == 'msg1'
            assert data['is_fraud'] == True
            assert data['confidence'] == 0.99
            assert data['fraud_type'] == 'scam'
    
    def test_batch_scan_emails(self, client, auth_headers):
        """Test /api/email/batch-scan endpoint"""
        with patch('fraudlens.api.secured_api.BatchEmailProcessor') as MockBatch:
            mock_processor = Mock()
            MockBatch.return_value = mock_processor
            
            mock_processor.process_batch = AsyncMock(return_value=[
                EmailScanResult(
                    message_id='msg1',
                    is_fraud=True,
                    confidence=0.9,
                    fraud_type='phishing',
                    risk_score=8.5
                ),
                EmailScanResult(
                    message_id='msg2',
                    is_fraud=False,
                    confidence=0.2,
                    fraud_type=None,
                    risk_score=2.0
                )
            ])
            
            response = client.post(
                "/api/email/batch-scan",
                headers=auth_headers,
                json={
                    "message_ids": ["msg1", "msg2"],
                    "batch_size": 10
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert 'results' in data
            assert len(data['results']) == 2
            assert data['total_scanned'] == 2
            assert data['fraud_detected'] == 1
    
    def test_mark_email_as_spam(self, client, auth_headers):
        """Test /api/email/mark-spam endpoint"""
        with patch('fraudlens.api.secured_api.GmailIntegration') as MockGmail:
            mock_integration = Mock()
            MockGmail.return_value = mock_integration
            
            mock_integration.mark_as_spam.return_value = True
            
            response = client.post(
                "/api/email/mark-spam",
                headers=auth_headers,
                json={
                    "message_ids": ["msg1", "msg2"]
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] == True
            assert data['marked_count'] == 2
    
    def test_move_to_trash(self, client, auth_headers):
        """Test /api/email/trash endpoint"""
        with patch('fraudlens.api.secured_api.GmailIntegration') as MockGmail:
            mock_integration = Mock()
            MockGmail.return_value = mock_integration
            
            mock_integration.move_to_trash.return_value = True
            
            response = client.post(
                "/api/email/trash",
                headers=auth_headers,
                json={
                    "message_ids": ["msg1"]
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] == True
            assert data['trashed_count'] == 1
    
    def test_create_email_filter(self, client, admin_headers):
        """Test /api/email/create-filter endpoint (admin only)"""
        with patch('fraudlens.api.secured_api.GmailIntegration') as MockGmail:
            mock_integration = Mock()
            MockGmail.return_value = mock_integration
            
            mock_integration.create_filter.return_value = 'filter123'
            
            response = client.post(
                "/api/email/create-filter",
                headers=admin_headers,
                json={
                    "from_address": "spammer@example.com",
                    "subject_contains": "URGENT",
                    "action": "delete"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['filter_id'] == 'filter123'
            assert data['success'] == True
    
    def test_get_email_stats(self, client, auth_headers):
        """Test /api/email/stats endpoint"""
        with patch('fraudlens.api.secured_api.get_email_stats') as mock_stats:
            mock_stats.return_value = {
                'total_scanned': 1000,
                'fraud_detected': 150,
                'fraud_rate': 0.15,
                'common_fraud_types': {
                    'phishing': 80,
                    'scam': 50,
                    'malware': 20
                },
                'high_risk_senders': [
                    'scammer1@fake.com',
                    'phisher@malicious.com'
                ]
            }
            
            response = client.get(
                "/api/email/stats",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['total_scanned'] == 1000
            assert data['fraud_detected'] == 150
            assert data['fraud_rate'] == 0.15
            assert 'phishing' in data['common_fraud_types']
    
    def test_realtime_monitoring_start(self, client, admin_headers):
        """Test /api/email/monitor/start endpoint"""
        with patch('fraudlens.api.secured_api.GmailIntegration') as MockGmail:
            mock_integration = Mock()
            MockGmail.return_value = mock_integration
            
            mock_integration.monitor_inbox_realtime = AsyncMock()
            
            response = client.post(
                "/api/email/monitor/start",
                headers=admin_headers,
                json={
                    "check_interval": 60,
                    "callback_url": "https://webhook.example.com/fraud"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['monitoring'] == True
            assert 'monitor_id' in data
    
    def test_realtime_monitoring_stop(self, client, admin_headers):
        """Test /api/email/monitor/stop endpoint"""
        with patch('fraudlens.api.secured_api.stop_monitoring') as mock_stop:
            mock_stop.return_value = True
            
            response = client.post(
                "/api/email/monitor/stop",
                headers=admin_headers,
                json={
                    "monitor_id": "monitor123"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['monitoring'] == False
    
    def test_export_fraud_report(self, client, auth_headers):
        """Test /api/email/export-report endpoint"""
        with patch('fraudlens.api.secured_api.generate_fraud_report') as mock_report:
            mock_report.return_value = {
                'report_id': 'report123',
                'generated_at': datetime.now().isoformat(),
                'total_emails': 500,
                'fraud_emails': 75,
                'report_url': 'https://reports.example.com/report123.pdf'
            }
            
            response = client.post(
                "/api/email/export-report",
                headers=auth_headers,
                json={
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "format": "pdf"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['report_id'] == 'report123'
            assert 'report_url' in data


class TestEmailAPIAuthentication:
    """Test authentication and authorization for email endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_unauthenticated_access(self, client):
        """Test accessing endpoints without authentication"""
        response = client.post("/api/email/scan-inbox")
        assert response.status_code == 401
        
        response = client.get("/api/email/scan/msg1")
        assert response.status_code == 401
    
    def test_insufficient_permissions(self, client):
        """Test accessing admin endpoints with user role"""
        # Create user with limited permissions
        user = UserInDB(
            id=3,
            username="viewer",
            email="viewer@example.com",
            hashed_password="hashed",
            role=UserRole.VIEWER,
            is_active=True,
            created_at=datetime.now()
        )
        
        tokens = create_tokens(user)
        headers = {"Authorization": f"Bearer {tokens.access_token}"}
        
        # Try to access admin-only endpoint
        response = client.post(
            "/api/email/create-filter",
            headers=headers,
            json={"from_address": "spam@example.com"}
        )
        
        assert response.status_code == 403
    
    def test_rate_limiting(self, client):
        """Test rate limiting on email endpoints"""
        user = UserInDB(
            id=4,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
            role=UserRole.USER,
            is_active=True,
            created_at=datetime.now()
        )
        
        tokens = create_tokens(user)
        headers = {"Authorization": f"Bearer {tokens.access_token}"}
        
        # Make many requests quickly
        responses = []
        for _ in range(150):  # Exceed rate limit
            response = client.get(
                "/api/email/stats",
                headers=headers
            )
            responses.append(response.status_code)
        
        # Should eventually get rate limited
        assert 429 in responses


class TestEmailAPIErrorHandling:
    """Test error handling in email API"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        user = UserInDB(
            id=5,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
            role=UserRole.USER,
            is_active=True,
            created_at=datetime.now()
        )
        tokens = create_tokens(user)
        return {"Authorization": f"Bearer {tokens.access_token}"}
    
    def test_gmail_api_error(self, client, auth_headers):
        """Test handling Gmail API errors"""
        with patch('fraudlens.api.secured_api.GmailIntegration') as MockGmail:
            mock_integration = Mock()
            MockGmail.return_value = mock_integration
            
            # Simulate Gmail API error
            from googleapiclient.errors import HttpError
            error_resp = Mock()
            error_resp.status = 500
            mock_integration.scan_inbox.side_effect = HttpError(
                error_resp, b'Internal Server Error'
            )
            
            response = client.post(
                "/api/email/scan-inbox",
                headers=auth_headers,
                json={"max_emails": 10}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert 'error' in data
    
    def test_invalid_message_id(self, client, auth_headers):
        """Test scanning non-existent message"""
        with patch('fraudlens.api.secured_api.GmailIntegration') as MockGmail:
            mock_integration = Mock()
            MockGmail.return_value = mock_integration
            
            mock_integration.get_message.return_value = None
            
            response = client.get(
                "/api/email/scan/invalid_id",
                headers=auth_headers
            )
            
            assert response.status_code == 404
            data = response.json()
            assert 'error' in data
    
    def test_batch_size_limit(self, client, auth_headers):
        """Test batch size limit validation"""
        message_ids = [f"msg{i}" for i in range(1001)]  # Exceed limit
        
        response = client.post(
            "/api/email/batch-scan",
            headers=auth_headers,
            json={
                "message_ids": message_ids,
                "batch_size": 1000
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
        assert 'batch size' in data['error'].lower()


class TestEmailAPIWebhooks:
    """Test webhook functionality for email fraud detection"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def admin_headers(self):
        admin = UserInDB(
            id=6,
            username="admin",
            email="admin@example.com",
            hashed_password="hashed",
            role=UserRole.ADMIN,
            is_active=True,
            created_at=datetime.now()
        )
        tokens = create_tokens(admin)
        return {"Authorization": f"Bearer {tokens.access_token}"}
    
    def test_webhook_registration(self, client, admin_headers):
        """Test registering a webhook for fraud alerts"""
        response = client.post(
            "/api/email/webhook/register",
            headers=admin_headers,
            json={
                "url": "https://webhook.example.com/fraud",
                "events": ["fraud_detected", "high_risk_sender"],
                "secret": "webhook_secret_key"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'webhook_id' in data
        assert data['active'] == True
    
    def test_webhook_test(self, client, admin_headers):
        """Test webhook with test payload"""
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            
            response = client.post(
                "/api/email/webhook/test",
                headers=admin_headers,
                json={
                    "webhook_id": "webhook123",
                    "test_payload": {
                        "event": "fraud_detected",
                        "message_id": "test_msg",
                        "fraud_type": "phishing"
                    }
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['test_successful'] == True
            mock_post.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])