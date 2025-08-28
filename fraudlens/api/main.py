"""
FraudLens RESTful API Implementation
Comprehensive fraud detection API with integrations and compliance features
"""

import os
import json
import uuid
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from io import BytesIO
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security, BackgroundTasks, Request, Query, Header
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr, HttpUrl
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import aiofiles
from loguru import logger

# Import FraudLens components
from fraudlens.core.pipeline import FraudDetectionPipeline
from fraudlens.core.config import Config
from fraudlens.monitoring.monitor import FraudLensMonitor
from fraudlens.integration.manager import IntegrationManager
from fraudlens.compliance.manager import ComplianceManager


# Request/Response Models
class TextAnalysisRequest(BaseModel):
    text: str = Field(..., max_length=50000)
    language: str = Field(default="en", max_length=2)
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None


class DocumentAnalysisRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = None
    async_processing: bool = Field(default=False)
    callback_url: Optional[HttpUrl] = None


class Transaction(BaseModel):
    transaction_id: Optional[str] = None
    amount: float
    currency: str = Field(..., min_length=3, max_length=3)
    timestamp: datetime
    sender: Optional[Dict[str, Any]] = None
    recipient: Optional[Dict[str, Any]] = None
    payment_method: Optional[str] = None
    merchant_info: Optional[Dict[str, Any]] = None
    device_info: Optional[Dict[str, Any]] = None


class BatchAnalysisRequest(BaseModel):
    items: List[Union[TextAnalysisRequest, Transaction]] = Field(..., min_items=1, max_items=1000)
    callback_url: Optional[HttpUrl] = None
    priority: str = Field(default="normal", regex="^(low|normal|high)$")


class WebhookRequest(BaseModel):
    url: HttpUrl
    events: List[str] = Field(..., min_items=1)
    secret: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    retry_policy: Optional[Dict[str, Any]] = None


class AnonymizationRequest(BaseModel):
    data: Dict[str, Any]
    fields: List[str]
    method: str = Field(default="redact", regex="^(redact|hash|mask|synthetic)$")


# Response Models
class AnalysisResult(BaseModel):
    analysis_id: str
    fraud_score: float = Field(..., ge=0, le=1)
    risk_level: str
    fraud_types: List[Dict[str, Any]]
    indicators: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]


class RiskScore(BaseModel):
    score: float = Field(..., ge=0, le=100)
    risk_level: str
    factors: List[Dict[str, Any]]
    recommendation: str
    requires_manual_review: bool


class AsyncJobResponse(BaseModel):
    job_id: str
    status: str
    status_url: str
    results_url: Optional[str] = None


class BatchJobResponse(BaseModel):
    job_id: str
    status: str
    item_count: int
    estimated_completion_time: datetime
    results_url: Optional[str] = None


class WebhookSubscription(BaseModel):
    id: str
    url: str
    events: List[str]
    created_at: datetime
    status: str
    last_delivery: Optional[datetime] = None
    delivery_count: int = 0


class DeletionResult(BaseModel):
    user_id: str
    records_deleted: int
    services: List[Dict[str, Any]]
    completed_at: datetime


class UserDataExport(BaseModel):
    user_id: str
    export_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]


class FraudLensAPI:
    """
    Main API class for FraudLens
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.app = FastAPI(
            title="FraudLens API",
            description="Comprehensive fraud detection and analysis API",
            version="1.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc",
            openapi_url="/api/openapi.json"
        )
        
        # Initialize components
        self.pipeline = FraudDetectionPipeline(self.config)
        self.monitor = FraudLensMonitor()
        self.integration_manager = IntegrationManager()
        self.compliance_manager = ComplianceManager()
        
        # Rate limiter
        self.limiter = Limiter(key_func=get_remote_address)
        
        # Security
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
        
        # Storage for async jobs and webhooks
        self.jobs = {}
        self.webhooks = {}
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Configure middleware"""
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure based on environment
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Rate limiting
        self.app.state.limiter = self.limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # Request logging
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            request_id = str(uuid.uuid4())
            logger.info(f"Request {request_id}: {request.method} {request.url.path}")
            
            response = await call_next(request)
            
            logger.info(f"Response {request_id}: {response.status_code}")
            response.headers["X-Request-ID"] = request_id
            
            return response
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Text Analysis
        @self.app.post("/api/v1/analyze/text", response_model=AnalysisResult)
        @self.limiter.limit("100/minute")
        async def analyze_text(
            request: Request,
            analysis_request: TextAnalysisRequest,
            background_tasks: BackgroundTasks
        ):
            """Analyze text for fraud indicators"""
            
            analysis_id = str(uuid.uuid4())
            
            # Run analysis
            result = await self.pipeline.analyze_text(
                text=analysis_request.text,
                language=analysis_request.language,
                context=analysis_request.context,
                options=analysis_request.options
            )
            
            # Log analysis
            background_tasks.add_task(
                self.monitor.record_detection,
                detector_id="text",
                fraud_score=result["fraud_score"],
                latency_ms=result.get("processing_time", 0) * 1000,
                success=True,
                metadata={
                    "analysis_id": analysis_id,
                    "text_length": len(analysis_request.text)
                }
            )
            
            return AnalysisResult(
                analysis_id=analysis_id,
                fraud_score=result["fraud_score"],
                risk_level=result["risk_level"],
                fraud_types=result.get("fraud_types", []),
                indicators=result.get("indicators", []),
                recommendations=result.get("recommendations", []),
                metadata={
                    "processing_time": result.get("processing_time", 0),
                    "models_used": result.get("models_used", []),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        # Document Analysis
        @self.app.post("/api/v1/analyze/document")
        @self.limiter.limit("50/minute")
        async def analyze_document(
            request: Request,
            file: UploadFile = File(...),
            metadata: Optional[str] = None,
            async_processing: bool = False,
            callback_url: Optional[str] = None,
            background_tasks: BackgroundTasks = None
        ):
            """Analyze document for fraud or forgery"""
            
            analysis_id = str(uuid.uuid4())
            
            # Save uploaded file
            content = await file.read()
            file_hash = hashlib.sha256(content).hexdigest()
            
            if async_processing:
                # Queue for async processing
                job_id = str(uuid.uuid4())
                self.jobs[job_id] = {
                    "type": "document",
                    "status": "pending",
                    "file": content,
                    "filename": file.filename,
                    "metadata": json.loads(metadata) if metadata else {},
                    "callback_url": callback_url,
                    "created_at": datetime.utcnow()
                }
                
                # Process in background
                background_tasks.add_task(
                    self._process_document_async,
                    job_id,
                    content,
                    file.filename
                )
                
                return AsyncJobResponse(
                    job_id=job_id,
                    status="pending",
                    status_url=f"/api/v1/jobs/{job_id}/status",
                    results_url=f"/api/v1/jobs/{job_id}/results"
                )
            else:
                # Process synchronously
                result = await self.pipeline.analyze_document(
                    content=content,
                    filename=file.filename,
                    metadata=json.loads(metadata) if metadata else {}
                )
                
                return {
                    "analysis_id": analysis_id,
                    "document_type": result.get("document_type"),
                    "fraud_score": result["fraud_score"],
                    "forgery_detected": result.get("forgery_detected", False),
                    "manipulations": result.get("manipulations", []),
                    "extracted_text": result.get("extracted_text", ""),
                    "metadata": {
                        "file_hash": file_hash,
                        "file_size": len(content),
                        "mime_type": file.content_type,
                        "processing_time": result.get("processing_time", 0)
                    }
                }
        
        # Transaction Analysis
        @self.app.post("/api/v1/analyze/transaction", response_model=RiskScore)
        @self.limiter.limit("200/minute")
        async def analyze_transaction(
            request: Request,
            transaction: Transaction,
            background_tasks: BackgroundTasks
        ):
            """Analyze transaction for fraud risk"""
            
            # Run risk assessment
            result = await self.pipeline.analyze_transaction(transaction.dict())
            
            # Log transaction analysis
            background_tasks.add_task(
                self.monitor.record_detection,
                detector_id="transaction",
                fraud_score=result["score"] / 100,
                latency_ms=result.get("processing_time", 0) * 1000,
                success=True,
                metadata={
                    "transaction_id": transaction.transaction_id,
                    "amount": transaction.amount,
                    "currency": transaction.currency
                }
            )
            
            return RiskScore(
                score=result["score"],
                risk_level=result["risk_level"],
                factors=result.get("factors", []),
                recommendation=result.get("recommendation", "review"),
                requires_manual_review=result.get("requires_manual_review", False)
            )
        
        # Batch Processing
        @self.app.post("/api/v1/analyze/batch", response_model=BatchJobResponse)
        @self.limiter.limit("10/minute")
        async def analyze_batch(
            request: Request,
            batch_request: BatchAnalysisRequest,
            background_tasks: BackgroundTasks
        ):
            """Submit batch analysis job"""
            
            job_id = str(uuid.uuid4())
            
            # Create batch job
            self.jobs[job_id] = {
                "type": "batch",
                "status": "queued",
                "items": [item.dict() for item in batch_request.items],
                "callback_url": batch_request.callback_url,
                "priority": batch_request.priority,
                "created_at": datetime.utcnow(),
                "progress": 0,
                "results": []
            }
            
            # Process in background
            background_tasks.add_task(
                self._process_batch_async,
                job_id
            )
            
            # Estimate completion time
            items_per_second = 10  # Adjust based on actual performance
            estimated_seconds = len(batch_request.items) / items_per_second
            estimated_completion = datetime.utcnow() + timedelta(seconds=estimated_seconds)
            
            return BatchJobResponse(
                job_id=job_id,
                status="queued",
                item_count=len(batch_request.items),
                estimated_completion_time=estimated_completion,
                results_url=f"/api/v1/jobs/{job_id}/results"
            )
        
        # Streaming Analysis
        @self.app.get("/api/v1/analyze/stream")
        @self.limiter.limit("10/minute")
        async def stream_analysis(
            request: Request,
            types: List[str] = Query(default=["all"])
        ):
            """Open streaming analysis connection"""
            
            async def event_generator():
                """Generate server-sent events"""
                
                client_id = str(uuid.uuid4())
                logger.info(f"Streaming client connected: {client_id}")
                
                try:
                    while True:
                        # Get latest analysis events
                        events = await self._get_stream_events(types)
                        
                        for event in events:
                            yield f"data: {json.dumps(event)}\n\n"
                        
                        await asyncio.sleep(1)  # Poll interval
                        
                except asyncio.CancelledError:
                    logger.info(f"Streaming client disconnected: {client_id}")
                    raise
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # Reports
        @self.app.get("/api/v1/reports/{analysis_id}")
        @self.limiter.limit("100/minute")
        async def get_report(
            request: Request,
            analysis_id: str,
            format: str = Query(default="json", regex="^(json|pdf|html)$")
        ):
            """Get detailed analysis report"""
            
            # Generate report
            report = await self.pipeline.generate_report(analysis_id, format)
            
            if format == "json":
                return report
            elif format == "pdf":
                # Generate PDF report
                pdf_content = await self._generate_pdf_report(report)
                return StreamingResponse(
                    BytesIO(pdf_content),
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f"attachment; filename=report_{analysis_id}.pdf"
                    }
                )
            elif format == "html":
                # Generate HTML report
                html_content = await self._generate_html_report(report)
                return StreamingResponse(
                    BytesIO(html_content.encode()),
                    media_type="text/html"
                )
        
        # Webhooks
        @self.app.get("/api/v1/webhooks", response_model=List[WebhookSubscription])
        async def list_webhooks(request: Request):
            """List webhook subscriptions"""
            return list(self.webhooks.values())
        
        @self.app.post("/api/v1/webhooks", response_model=WebhookSubscription)
        @self.limiter.limit("10/minute")
        async def subscribe_webhook(
            request: Request,
            webhook_request: WebhookRequest
        ):
            """Create webhook subscription"""
            
            webhook_id = str(uuid.uuid4())
            
            subscription = WebhookSubscription(
                id=webhook_id,
                url=str(webhook_request.url),
                events=webhook_request.events,
                created_at=datetime.utcnow(),
                status="active",
                last_delivery=None,
                delivery_count=0
            )
            
            self.webhooks[webhook_id] = {
                **subscription.dict(),
                "secret": webhook_request.secret,
                "headers": webhook_request.headers,
                "retry_policy": webhook_request.retry_policy
            }
            
            logger.info(f"Webhook subscription created: {webhook_id}")
            
            return subscription
        
        @self.app.delete("/api/v1/webhooks/{webhook_id}", status_code=204)
        async def delete_webhook(request: Request, webhook_id: str):
            """Delete webhook subscription"""
            
            if webhook_id not in self.webhooks:
                raise HTTPException(status_code=404, detail="Webhook not found")
            
            del self.webhooks[webhook_id]
            logger.info(f"Webhook subscription deleted: {webhook_id}")
        
        # Compliance
        @self.app.post("/api/v1/compliance/anonymize")
        @self.limiter.limit("50/minute")
        async def anonymize_data(
            request: Request,
            anonymization_request: AnonymizationRequest
        ):
            """Anonymize sensitive data"""
            
            result = await self.compliance_manager.anonymize_data(
                data=anonymization_request.data,
                fields=anonymization_request.fields,
                method=anonymization_request.method
            )
            
            return {
                "data": result["data"],
                "anonymization_report": {
                    "fields_anonymized": anonymization_request.fields,
                    "method": anonymization_request.method,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        @self.app.delete("/api/v1/compliance/delete-user/{user_id}")
        @self.limiter.limit("10/minute")
        async def delete_user_data(
            request: Request,
            user_id: str,
            background_tasks: BackgroundTasks
        ):
            """Delete all user data (GDPR right to deletion)"""
            
            # Perform deletion
            result = await self.compliance_manager.delete_user_data(user_id)
            
            # Log deletion for audit
            background_tasks.add_task(
                self.compliance_manager.log_deletion,
                user_id=user_id,
                records_deleted=result["records_deleted"],
                timestamp=datetime.utcnow()
            )
            
            return DeletionResult(
                user_id=user_id,
                records_deleted=result["records_deleted"],
                services=result["services"],
                completed_at=datetime.utcnow()
            )
        
        @self.app.get("/api/v1/compliance/export-user/{user_id}")
        @self.limiter.limit("10/minute")
        async def export_user_data(
            request: Request,
            user_id: str,
            format: str = Query(default="json", regex="^(json|csv|xml)$")
        ):
            """Export all user data (GDPR data portability)"""
            
            export_id = str(uuid.uuid4())
            
            # Export data
            data = await self.compliance_manager.export_user_data(user_id, format)
            
            return UserDataExport(
                user_id=user_id,
                export_id=export_id,
                data=data["content"],
                metadata={
                    "exported_at": datetime.utcnow().isoformat(),
                    "format": format,
                    "size_bytes": len(json.dumps(data["content"]))
                }
            )
        
        @self.app.get("/api/v1/compliance/audit-log")
        @self.limiter.limit("100/minute")
        async def get_audit_log(
            request: Request,
            start_date: datetime,
            end_date: datetime,
            user_id: Optional[str] = None,
            page: int = Query(default=1, ge=1),
            limit: int = Query(default=100, le=1000)
        ):
            """Retrieve audit log"""
            
            entries = await self.compliance_manager.get_audit_log(
                start_date=start_date,
                end_date=end_date,
                user_id=user_id,
                page=page,
                limit=limit
            )
            
            total_records = len(entries)
            total_pages = (total_records + limit - 1) // limit
            
            return {
                "entries": entries[(page-1)*limit:page*limit],
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total_pages": total_pages,
                    "total_records": total_records
                }
            }
        
        # Job Status
        @self.app.get("/api/v1/jobs/{job_id}/status")
        async def get_job_status(request: Request, job_id: str):
            """Get job status"""
            
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.jobs[job_id]
            return {
                "job_id": job_id,
                "status": job["status"],
                "progress": job.get("progress", 0),
                "created_at": job["created_at"],
                "updated_at": job.get("updated_at")
            }
        
        @self.app.get("/api/v1/jobs/{job_id}/results")
        async def get_job_results(request: Request, job_id: str):
            """Get job results"""
            
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.jobs[job_id]
            
            if job["status"] != "completed":
                raise HTTPException(status_code=400, detail="Job not completed")
            
            return job.get("results", [])
        
        # Health and Metrics
        @self.app.get("/api/v1/admin/health")
        async def health_check():
            """Health check endpoint"""
            
            services_status = {
                "pipeline": self.pipeline.is_initialized,
                "monitor": True,
                "integrations": self.integration_manager.is_healthy(),
                "compliance": self.compliance_manager.is_healthy()
            }
            
            overall_status = "healthy" if all(services_status.values()) else "degraded"
            
            return {
                "status": overall_status,
                "version": "1.0.0",
                "uptime": self.monitor.get_uptime(),
                "services": services_status
            }
        
        @self.app.get("/api/v1/admin/metrics")
        async def get_metrics():
            """Get service metrics in Prometheus format"""
            
            metrics = self.monitor.get_prometheus_metrics()
            return StreamingResponse(
                BytesIO(metrics.encode()),
                media_type="text/plain"
            )
    
    async def _process_document_async(self, job_id: str, content: bytes, filename: str):
        """Process document asynchronously"""
        
        try:
            self.jobs[job_id]["status"] = "processing"
            self.jobs[job_id]["updated_at"] = datetime.utcnow()
            
            # Process document
            result = await self.pipeline.analyze_document(
                content=content,
                filename=filename
            )
            
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["results"] = result
            self.jobs[job_id]["updated_at"] = datetime.utcnow()
            
            # Send webhook if configured
            if self.jobs[job_id].get("callback_url"):
                await self._send_webhook_notification(
                    url=self.jobs[job_id]["callback_url"],
                    event="analysis.completed",
                    data={
                        "job_id": job_id,
                        "results": result
                    }
                )
            
        except Exception as e:
            logger.error(f"Document processing failed for job {job_id}: {e}")
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["updated_at"] = datetime.utcnow()
    
    async def _process_batch_async(self, job_id: str):
        """Process batch job asynchronously"""
        
        try:
            job = self.jobs[job_id]
            job["status"] = "processing"
            job["updated_at"] = datetime.utcnow()
            
            results = []
            total_items = len(job["items"])
            
            for i, item in enumerate(job["items"]):
                # Process each item
                if "text" in item:
                    result = await self.pipeline.analyze_text(**item)
                elif "amount" in item:
                    result = await self.pipeline.analyze_transaction(item)
                else:
                    result = {"error": "Unknown item type"}
                
                results.append(result)
                
                # Update progress
                job["progress"] = (i + 1) / total_items
                job["updated_at"] = datetime.utcnow()
            
            job["status"] = "completed"
            job["results"] = results
            job["updated_at"] = datetime.utcnow()
            
            # Send webhook if configured
            if job.get("callback_url"):
                await self._send_webhook_notification(
                    url=job["callback_url"],
                    event="batch.completed",
                    data={
                        "job_id": job_id,
                        "results": results
                    }
                )
            
        except Exception as e:
            logger.error(f"Batch processing failed for job {job_id}: {e}")
            job["status"] = "failed"
            job["error"] = str(e)
            job["updated_at"] = datetime.utcnow()
    
    async def _get_stream_events(self, types: List[str]) -> List[Dict[str, Any]]:
        """Get events for streaming"""
        
        events = []
        
        # Get recent analysis events
        if "all" in types or "analysis" in types:
            # Implement event collection logic
            pass
        
        return events
    
    async def _send_webhook_notification(self, url: str, event: str, data: Dict[str, Any]):
        """Send webhook notification"""
        
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json={
                        "event": event,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": data
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status >= 200 and response.status < 300:
                        logger.info(f"Webhook delivered to {url}")
                    else:
                        logger.error(f"Webhook delivery failed: {response.status}")
        except Exception as e:
            logger.error(f"Webhook delivery error: {e}")
    
    async def _generate_pdf_report(self, report: Dict[str, Any]) -> bytes:
        """Generate PDF report"""
        
        # Implement PDF generation using reportlab or similar
        # This is a placeholder
        return b"PDF Report Content"
    
    async def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FraudLens Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .summary {{ background: #f0f0f0; padding: 20px; border-radius: 8px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <h1>FraudLens Analysis Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">Risk Level: {report.get('risk_level', 'Unknown')}</div>
                <div class="metric">Fraud Score: {report.get('fraud_score', 0):.2%}</div>
            </div>
            <h2>Details</h2>
            <pre>{json.dumps(report, indent=2)}</pre>
        </body>
        </html>
        """
        
        return html
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server"""
        
        import uvicorn
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )


if __name__ == "__main__":
    api = FraudLensAPI()
    api.run()