"""
FraudLens Python Client SDK
Async-enabled client for FraudLens API
"""

import os
import json
import asyncio
import hashlib
import hmac
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, BinaryIO
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import requests
from urllib.parse import urljoin


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalysisType(Enum):
    """Analysis type enumeration"""
    TEXT = "text"
    DOCUMENT = "document"
    TRANSACTION = "transaction"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class TextAnalysisRequest:
    """Text analysis request"""
    text: str
    language: str = "en"
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None


@dataclass
class TransactionRequest:
    """Transaction analysis request"""
    amount: float
    currency: str
    timestamp: datetime
    sender: Optional[Dict[str, Any]] = None
    recipient: Optional[Dict[str, Any]] = None
    payment_method: Optional[str] = None
    merchant_info: Optional[Dict[str, Any]] = None
    device_info: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisResult:
    """Analysis result"""
    analysis_id: str
    fraud_score: float
    risk_level: str
    fraud_types: List[Dict[str, Any]]
    indicators: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class RiskScore:
    """Transaction risk score"""
    score: float
    risk_level: str
    factors: List[Dict[str, Any]]
    recommendation: str
    requires_manual_review: bool


class FraudLensError(Exception):
    """Base exception for FraudLens SDK"""
    pass


class AuthenticationError(FraudLensError):
    """Authentication failed"""
    pass


class RateLimitError(FraudLensError):
    """Rate limit exceeded"""
    pass


class ValidationError(FraudLensError):
    """Request validation failed"""
    pass


class FraudLensClient:
    """
    FraudLens API client
    
    Usage:
        client = FraudLensClient(api_key="your-api-key")
        
        # Analyze text
        result = await client.analyze_text("Suspicious email content")
        
        # Analyze document
        with open("document.pdf", "rb") as f:
            result = await client.analyze_document(f)
        
        # Analyze transaction
        result = await client.analyze_transaction(
            amount=1000.00,
            currency="USD"
        )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.fraudlens.com/v1",
        timeout: int = 30,
        max_retries: int = 3,
        verify_ssl: bool = True
    ):
        """
        Initialize FraudLens client
        
        Args:
            api_key: API key for authentication
            api_url: Base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            verify_ssl: Verify SSL certificates
        """
        self.api_key = api_key or os.getenv("FRAUDLENS_API_KEY")
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        
        if not self.api_key:
            raise AuthenticationError("API key is required")
        
        # Session for connection pooling
        self.session = None
        self._sync_session = requests.Session()
        self._sync_session.headers.update({
            "X-API-Key": self.api_key,
            "User-Agent": "FraudLens-Python-SDK/1.0.0"
        })
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                "X-API-Key": self.api_key,
                "User-Agent": "FraudLens-Python-SDK/1.0.0"
            },
            connector=aiohttp.TCPConnector(ssl=self.verify_ssl),
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _make_url(self, endpoint: str) -> str:
        """Construct full URL for endpoint"""
        return urljoin(self.api_url, endpoint.lstrip("/"))
    
    def _sign_request(self, method: str, url: str, body: Optional[str] = None) -> str:
        """Generate HMAC signature for request"""
        timestamp = str(int(datetime.utcnow().timestamp()))
        message = f"{method.upper()}\n{url}\n{timestamp}"
        
        if body:
            message += f"\n{body}"
        
        signature = hmac.new(
            self.api_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{timestamp}:{signature}"
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make async HTTP request"""
        
        if not self.session:
            raise RuntimeError("Client must be used within async context manager")
        
        url = self._make_url(endpoint)
        
        for attempt in range(self.max_retries):
            try:
                if files:
                    # Multipart form data
                    form_data = aiohttp.FormData()
                    for key, value in (data or {}).items():
                        form_data.add_field(key, json.dumps(value) if isinstance(value, dict) else str(value))
                    for key, file_data in files.items():
                        form_data.add_field(key, file_data["content"], 
                                          filename=file_data["filename"],
                                          content_type=file_data.get("content_type", "application/octet-stream"))
                    
                    async with self.session.request(method, url, data=form_data, params=params) as response:
                        return await self._handle_response(response)
                else:
                    # JSON request
                    async with self.session.request(method, url, json=data, params=params) as response:
                        return await self._handle_response(response)
                        
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    raise FraudLensError(f"Request failed after {self.max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _request_sync(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make synchronous HTTP request"""
        
        url = self._make_url(endpoint)
        
        for attempt in range(self.max_retries):
            try:
                if files:
                    # Prepare files for upload
                    files_dict = {}
                    for key, file_data in files.items():
                        files_dict[key] = (
                            file_data["filename"],
                            file_data["content"],
                            file_data.get("content_type", "application/octet-stream")
                        )
                    
                    response = self._sync_session.request(
                        method, url, 
                        data=data,
                        files=files_dict,
                        params=params,
                        timeout=self.timeout
                    )
                else:
                    response = self._sync_session.request(
                        method, url,
                        json=data,
                        params=params,
                        timeout=self.timeout
                    )
                
                return self._handle_response_sync(response)
                
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise FraudLensError(f"Request failed after {self.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle async response"""
        
        try:
            data = await response.json()
        except:
            data = {"error": await response.text()}
        
        if response.status == 200:
            return data
        elif response.status == 201:
            return data
        elif response.status == 202:
            return data
        elif response.status == 401:
            raise AuthenticationError(data.get("error", {}).get("message", "Authentication failed"))
        elif response.status == 429:
            raise RateLimitError(data.get("error", {}).get("message", "Rate limit exceeded"))
        elif response.status >= 400:
            raise FraudLensError(data.get("error", {}).get("message", f"Request failed with status {response.status}"))
        
        return data
    
    def _handle_response_sync(self, response: requests.Response) -> Dict[str, Any]:
        """Handle synchronous response"""
        
        try:
            data = response.json()
        except:
            data = {"error": response.text}
        
        if response.status_code == 200:
            return data
        elif response.status_code == 201:
            return data
        elif response.status_code == 202:
            return data
        elif response.status_code == 401:
            raise AuthenticationError(data.get("error", {}).get("message", "Authentication failed"))
        elif response.status_code == 429:
            raise RateLimitError(data.get("error", {}).get("message", "Rate limit exceeded"))
        elif response.status_code >= 400:
            raise FraudLensError(data.get("error", {}).get("message", f"Request failed with status {response.status_code}"))
        
        return data
    
    # Async methods
    async def analyze_text(
        self,
        text: str,
        language: str = "en",
        context: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Analyze text for fraud indicators
        
        Args:
            text: Text content to analyze
            language: Language code (ISO 639-1)
            context: Additional context
            options: Analysis options
            
        Returns:
            AnalysisResult object
        """
        
        request_data = {
            "text": text,
            "language": language,
            "context": context,
            "options": options
        }
        
        response = await self._request("POST", "/analyze/text", data=request_data)
        
        return AnalysisResult(**response)
    
    async def analyze_document(
        self,
        file: Union[str, Path, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None,
        async_processing: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze document for fraud or forgery
        
        Args:
            file: File path or file-like object
            metadata: Additional metadata
            async_processing: Process asynchronously
            
        Returns:
            Analysis result or job response
        """
        
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            filename = file_path.name
            with open(file_path, "rb") as f:
                content = f.read()
        else:
            filename = getattr(file, "name", "document")
            content = file.read()
        
        files = {
            "file": {
                "filename": filename,
                "content": content,
                "content_type": "application/octet-stream"
            }
        }
        
        data = {}
        if metadata:
            data["metadata"] = json.dumps(metadata)
        if async_processing:
            data["async_processing"] = "true"
        
        response = await self._request("POST", "/analyze/document", data=data, files=files)
        
        return response
    
    async def analyze_transaction(
        self,
        amount: float,
        currency: str,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> RiskScore:
        """
        Analyze transaction for fraud risk
        
        Args:
            amount: Transaction amount
            currency: Currency code
            timestamp: Transaction timestamp
            **kwargs: Additional transaction fields
            
        Returns:
            RiskScore object
        """
        
        request_data = {
            "amount": amount,
            "currency": currency,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            **kwargs
        }
        
        response = await self._request("POST", "/analyze/transaction", data=request_data)
        
        return RiskScore(**response)
    
    async def analyze_batch(
        self,
        items: List[Union[TextAnalysisRequest, TransactionRequest]],
        callback_url: Optional[str] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Submit batch analysis job
        
        Args:
            items: List of items to analyze
            callback_url: URL for completion callback
            priority: Job priority (low/normal/high)
            
        Returns:
            Batch job response
        """
        
        request_data = {
            "items": [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in items],
            "callback_url": callback_url,
            "priority": priority
        }
        
        response = await self._request("POST", "/analyze/batch", data=request_data)
        
        return response
    
    async def get_report(
        self,
        analysis_id: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Get analysis report
        
        Args:
            analysis_id: Analysis ID
            format: Report format (json/pdf/html)
            
        Returns:
            Report data
        """
        
        params = {"format": format}
        response = await self._request("GET", f"/reports/{analysis_id}", params=params)
        
        return response
    
    async def subscribe_webhook(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Subscribe to webhook events
        
        Args:
            url: Webhook URL
            events: List of event types
            secret: Secret for signature validation
            
        Returns:
            Webhook subscription
        """
        
        request_data = {
            "url": url,
            "events": events,
            "secret": secret
        }
        
        response = await self._request("POST", "/webhooks", data=request_data)
        
        return response
    
    async def list_webhooks(self) -> List[Dict[str, Any]]:
        """List webhook subscriptions"""
        
        response = await self._request("GET", "/webhooks")
        
        return response
    
    async def delete_webhook(self, webhook_id: str) -> None:
        """Delete webhook subscription"""
        
        await self._request("DELETE", f"/webhooks/{webhook_id}")
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        
        response = await self._request("GET", f"/jobs/{job_id}/status")
        
        return response
    
    async def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Get job results"""
        
        response = await self._request("GET", f"/jobs/{job_id}/results")
        
        return response
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        
        response = await self._request("GET", "/admin/health")
        
        return response
    
    # Synchronous methods
    def analyze_text_sync(
        self,
        text: str,
        language: str = "en",
        context: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """Synchronous version of analyze_text"""
        
        request_data = {
            "text": text,
            "language": language,
            "context": context,
            "options": options
        }
        
        response = self._request_sync("POST", "/analyze/text", data=request_data)
        
        return AnalysisResult(**response)
    
    def analyze_document_sync(
        self,
        file: Union[str, Path, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronous version of analyze_document"""
        
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            filename = file_path.name
            with open(file_path, "rb") as f:
                content = f.read()
        else:
            filename = getattr(file, "name", "document")
            content = file.read()
        
        files = {
            "file": {
                "filename": filename,
                "content": content,
                "content_type": "application/octet-stream"
            }
        }
        
        data = {}
        if metadata:
            data["metadata"] = json.dumps(metadata)
        
        response = self._request_sync("POST", "/analyze/document", data=data, files=files)
        
        return response
    
    def analyze_transaction_sync(
        self,
        amount: float,
        currency: str,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> RiskScore:
        """Synchronous version of analyze_transaction"""
        
        request_data = {
            "amount": amount,
            "currency": currency,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            **kwargs
        }
        
        response = self._request_sync("POST", "/analyze/transaction", data=request_data)
        
        return RiskScore(**response)


# Convenience functions
async def analyze_text(text: str, api_key: Optional[str] = None, **kwargs) -> AnalysisResult:
    """Quick text analysis"""
    async with FraudLensClient(api_key=api_key) as client:
        return await client.analyze_text(text, **kwargs)


async def analyze_document(file: Union[str, Path, BinaryIO], api_key: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Quick document analysis"""
    async with FraudLensClient(api_key=api_key) as client:
        return await client.analyze_document(file, **kwargs)


async def analyze_transaction(amount: float, currency: str, api_key: Optional[str] = None, **kwargs) -> RiskScore:
    """Quick transaction analysis"""
    async with FraudLensClient(api_key=api_key) as client:
        return await client.analyze_transaction(amount, currency, **kwargs)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Initialize client
        async with FraudLensClient(api_key="your-api-key") as client:
            
            # Analyze text
            result = await client.analyze_text(
                text="Congratulations! You've won $1,000,000! Click here to claim.",
                context={"source": "email"}
            )
            print(f"Fraud score: {result.fraud_score}")
            print(f"Risk level: {result.risk_level}")
            
            # Analyze document
            with open("document.pdf", "rb") as f:
                doc_result = await client.analyze_document(f)
                print(f"Document fraud score: {doc_result['fraud_score']}")
            
            # Analyze transaction
            risk = await client.analyze_transaction(
                amount=5000.00,
                currency="USD",
                payment_method="credit_card"
            )
            print(f"Transaction risk: {risk.risk_level}")
    
    asyncio.run(main())