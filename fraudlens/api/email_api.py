"""
REST API for Email Fraud Detection
FastAPI endpoints for Gmail integration and bulk processing
"""

import asyncio
import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from .gmail_integration import EmailAction, EmailAnalysisResult, GmailFraudScanner

# FastAPI app
app = FastAPI(
    title="FraudLens Email API",
    description="Email fraud detection and management API",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global scanner instance
scanner: Optional[GmailFraudScanner] = None
monitoring_task: Optional[asyncio.Task] = None


# Pydantic models
class EmailQuery(BaseModel):
    """Email query parameters."""

    query: str = Field(default="is:unread", description="Gmail search query")
    max_results: int = Field(default=100, ge=1, le=500, description="Maximum emails to process")
    process_attachments: bool = Field(default=True, description="Process email attachments")
    since_days: int = Field(default=7, ge=1, le=365, description="Process emails from last N days")


class BulkProcessRequest(BaseModel):
    """Bulk processing request."""

    queries: List[str] = Field(description="List of Gmail queries to process")
    parallel: bool = Field(default=True, description="Process queries in parallel")
    max_workers: int = Field(default=5, ge=1, le=20, description="Maximum parallel workers")


class MonitoringConfig(BaseModel):
    """Email monitoring configuration."""

    enabled: bool = Field(description="Enable/disable monitoring")
    interval_seconds: int = Field(
        default=60, ge=10, le=3600, description="Check interval in seconds"
    )
    query: str = Field(default="is:unread", description="Gmail query for monitoring")
    auto_action: bool = Field(default=False, description="Automatically take action on fraud")


class ActionConfig(BaseModel):
    """Action configuration."""

    fraud_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Fraud detection threshold"
    )
    auto_action: bool = Field(default=False, description="Enable automatic actions")
    action_thresholds: Dict[str, float] = Field(
        default={
            "flag": 0.5,
            "spam": 0.7,
            "trash": 0.95,
            "quarantine": 0.8,
        },
        description="Action thresholds",
    )


class EmailActionRequest(BaseModel):
    """Manual action request."""

    message_ids: List[str] = Field(description="Email message IDs")
    action: EmailAction = Field(description="Action to take")


# API Endpoints


@app.on_event("startup")
async def startup_event():
    """Initialize scanner on startup."""
    global scanner
    scanner = GmailFraudScanner()
    await scanner.initialize()
    logger.info("Email API started successfully")


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "FraudLens Email API",
        "version": "1.0.0",
        "status": "running",
        "scanner_initialized": scanner is not None,
        "monitoring_active": monitoring_task is not None and not monitoring_task.done(),
    }


@app.post("/api/v1/scan/stream")
async def stream_and_scan(
    query: EmailQuery = Body(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Stream and scan emails from Gmail.

    Process emails matching the query and return fraud analysis results.
    """
    if not scanner:
        raise HTTPException(status_code=503, detail="Scanner not initialized")

    try:
        results = await scanner.stream_emails(
            query=query.query,
            max_results=query.max_results,
            process_attachments=query.process_attachments,
            since_days=query.since_days,
        )

        # Convert results to dict
        results_dict = [
            {
                **result.__dict__,
                "date": result.date.isoformat(),
                "action_taken": result.action_taken.value,
                "fraud_types": result.fraud_types,
            }
            for result in results
        ]

        return {
            "success": True,
            "count": len(results),
            "results": results_dict,
            "statistics": scanner.get_statistics(),
        }

    except Exception as e:
        logger.error(f"Error scanning emails: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/scan/bulk")
async def bulk_scan(request: BulkProcessRequest):
    """
    Bulk process multiple email queries.

    Process multiple Gmail queries and return aggregated results.
    """
    if not scanner:
        raise HTTPException(status_code=503, detail="Scanner not initialized")

    try:
        results = await scanner.bulk_process(
            queries=request.queries,
            parallel=request.parallel,
            max_workers=request.max_workers,
        )

        # Convert results
        formatted_results = {}
        total_count = 0
        total_fraud = 0

        for query, query_results in results.items():
            formatted_results[query] = {
                "count": len(query_results),
                "fraud_count": sum(
                    1 for r in query_results if r.fraud_score > scanner.fraud_threshold
                ),
                "results": [
                    {
                        **r.__dict__,
                        "date": r.date.isoformat(),
                        "action_taken": r.action_taken.value,
                    }
                    for r in query_results
                ],
            }
            total_count += len(query_results)
            total_fraud += formatted_results[query]["fraud_count"]

        return {
            "success": True,
            "total_processed": total_count,
            "total_fraud_detected": total_fraud,
            "queries": formatted_results,
            "statistics": scanner.get_statistics(),
        }

    except Exception as e:
        logger.error(f"Error in bulk scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scan/email/{message_id}")
async def scan_single_email(
    message_id: str,
    process_attachments: bool = Query(default=True),
):
    """
    Scan a single email by message ID.

    Process a specific email and return fraud analysis.
    """
    if not scanner:
        raise HTTPException(status_code=503, detail="Scanner not initialized")

    try:
        result = await scanner.process_email(
            message_id=message_id,
            process_attachments=process_attachments,
        )

        return {
            "success": True,
            "result": {
                **result.__dict__,
                "date": result.date.isoformat(),
                "action_taken": result.action_taken.value,
            },
        }

    except Exception as e:
        logger.error(f"Error scanning email {message_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/monitor/start")
async def start_monitoring(config: MonitoringConfig):
    """
    Start continuous email monitoring.

    Begin monitoring inbox for new emails and automatically process them.
    """
    global monitoring_task

    if not scanner:
        raise HTTPException(status_code=503, detail="Scanner not initialized")

    if monitoring_task and not monitoring_task.done():
        raise HTTPException(status_code=400, detail="Monitoring already active")

    if config.enabled:
        # Update scanner settings
        scanner.auto_action = config.auto_action

        # Start monitoring task
        monitoring_task = asyncio.create_task(
            scanner.monitor_inbox(
                interval_seconds=config.interval_seconds,
                query=config.query,
            )
        )

        return {
            "success": True,
            "message": "Monitoring started",
            "config": config.dict(),
        }
    else:
        return {
            "success": False,
            "message": "Monitoring not enabled",
        }


@app.post("/api/v1/monitor/stop")
async def stop_monitoring():
    """
    Stop email monitoring.

    Stop the continuous monitoring task.
    """
    global monitoring_task

    if monitoring_task and not monitoring_task.done():
        monitoring_task.cancel()
        monitoring_task = None
        return {
            "success": True,
            "message": "Monitoring stopped",
        }
    else:
        return {
            "success": False,
            "message": "No active monitoring",
        }


@app.get("/api/v1/monitor/status")
async def monitoring_status():
    """
    Get monitoring status.

    Check if monitoring is active and get current statistics.
    """
    return {
        "monitoring_active": monitoring_task is not None and not monitoring_task.done(),
        "statistics": scanner.get_statistics() if scanner else {},
    }


@app.post("/api/v1/action/execute")
async def execute_action(request: EmailActionRequest):
    """
    Execute action on emails.

    Manually execute an action on specified emails.
    """
    if not scanner:
        raise HTTPException(status_code=503, detail="Scanner not initialized")

    results = []

    for message_id in request.message_ids:
        try:
            # Create a dummy result for action
            result = EmailAnalysisResult(
                message_id=message_id,
                subject="",
                sender="",
                recipient="",
                date=datetime.now(),
                fraud_score=1.0,  # Assume fraud for manual action
                fraud_types=[],
                confidence=1.0,
                explanation="Manual action",
                attachments_analyzed=[],
                action_taken=request.action,
                processing_time_ms=0,
                raw_content_score=1.0,
                attachment_scores=[],
                combined_score=1.0,
                flagged=True,
            )

            await scanner._take_action(result)
            results.append({"message_id": message_id, "success": True})

        except Exception as e:
            results.append({"message_id": message_id, "success": False, "error": str(e)})

    return {
        "success": True,
        "action": request.action.value,
        "results": results,
    }


@app.put("/api/v1/config/update")
async def update_configuration(config: ActionConfig):
    """
    Update scanner configuration.

    Update fraud thresholds and action settings.
    """
    if not scanner:
        raise HTTPException(status_code=503, detail="Scanner not initialized")

    scanner.fraud_threshold = config.fraud_threshold
    scanner.auto_action = config.auto_action

    # Update action thresholds
    for action_name, threshold in config.action_thresholds.items():
        action = EmailAction(action_name)
        scanner.action_threshold[action] = threshold

    return {
        "success": True,
        "config": {
            "fraud_threshold": scanner.fraud_threshold,
            "auto_action": scanner.auto_action,
            "action_thresholds": {
                action.value: threshold for action, threshold in scanner.action_threshold.items()
            },
        },
    }


@app.get("/api/v1/statistics")
async def get_statistics():
    """
    Get processing statistics.

    Return detailed statistics about email processing.
    """
    if not scanner:
        raise HTTPException(status_code=503, detail="Scanner not initialized")

    return scanner.get_statistics()


@app.post("/api/v1/webhook/gmail")
async def gmail_webhook(
    background_tasks: BackgroundTasks,
    data: Dict[str, Any] = Body(...),
):
    """
    Gmail push notification webhook.

    Receive push notifications from Gmail and process new emails.
    """
    if not scanner:
        raise HTTPException(status_code=503, detail="Scanner not initialized")

    # Extract message data from webhook
    message_data = data.get("message", {})

    if message_data:
        # Process in background
        background_tasks.add_task(
            scanner.process_email,
            message_data.get("id"),
            process_attachments=True,
        )

    return {"success": True, "message": "Processing initiated"}


@app.get("/api/v1/export/results")
async def export_results(
    format: str = Query(default="json", regex="^(json|csv)$"),
    since_days: int = Query(default=7, ge=1, le=365),
):
    """
    Export scan results.

    Export email scan results in JSON or CSV format.
    """
    if not scanner:
        raise HTTPException(status_code=503, detail="Scanner not initialized")

    # Get recent results
    results = await scanner.stream_emails(
        query="label:FraudLens/Analyzed",
        max_results=500,
        since_days=since_days,
    )

    if format == "json":
        # Return JSON
        return JSONResponse(
            content={
                "export_date": datetime.now().isoformat(),
                "count": len(results),
                "results": [
                    {
                        **r.__dict__,
                        "date": r.date.isoformat(),
                        "action_taken": r.action_taken.value,
                    }
                    for r in results
                ],
            }
        )
    else:
        # Return CSV
        import csv

        output = io.StringIO()

        if results:
            fieldnames = [
                "message_id",
                "subject",
                "sender",
                "date",
                "fraud_score",
                "fraud_types",
                "action_taken",
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                writer.writerow(
                    {
                        "message_id": r.message_id,
                        "subject": r.subject,
                        "sender": r.sender,
                        "date": r.date.isoformat(),
                        "fraud_score": r.fraud_score,
                        "fraud_types": ",".join(r.fraud_types),
                        "action_taken": r.action_taken.value,
                    }
                )

        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=fraud_results.csv"},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
