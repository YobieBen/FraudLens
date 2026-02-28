"""
Local transport for in-process FraudLens execution.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import asyncio
from datetime import datetime
from typing import Any, AsyncIterator
from uuid import uuid4

from fraudlens.client.exceptions import AnalysisError, TimeoutError, TransportError
from fraudlens.client.models import AnalysisResult, AnalysisStatus, StreamEvent


class LocalTransport:
    """
    Transport for direct in-process calls to FraudLens engine.
    
    This transport executes fraud analysis locally without any network calls.
    Ideal for self-hosted deployments and embedded use cases.
    """
    
    def __init__(self, engine: Any = None):
        """
        Initialize local transport.
        
        Args:
            engine: FraudLens engine instance (optional, will be lazy-loaded)
        """
        self._engine = engine
        self._closed = False
    
    async def _get_engine(self) -> Any:
        """Get or initialize the FraudLens engine."""
        if self._engine is None:
            # Lazy import to avoid circular dependencies
            from fraudlens.core.container import get_container
            
            container = get_container()
            self._engine = await container.get_engine()
        
        return self._engine
    
    async def send_request(
        self,
        endpoint: str,
        data: dict[str, Any],
        timeout: float | None = None
    ) -> dict[str, Any]:
        """
        Send a request to local engine.
        
        Args:
            endpoint: API endpoint (e.g., "/analyze/email")
            data: Request payload
            timeout: Request timeout in seconds
        
        Returns:
            Response data
        
        Raises:
            TransportError: If execution fails
            TimeoutError: If request times out
        """
        if self._closed:
            raise TransportError("Transport is closed")
        
        try:
            # Route to appropriate handler based on endpoint
            if "/analyze/text" in endpoint:
                result = await self._analyze_text(data, timeout)
            elif "/analyze/email" in endpoint:
                result = await self._analyze_email(data, timeout)
            elif "/analyze/document" in endpoint:
                result = await self._analyze_document(data, timeout)
            elif "/analyze/image" in endpoint:
                result = await self._analyze_image(data, timeout)
            elif "/analyze/transaction" in endpoint:
                result = await self._analyze_transaction(data, timeout)
            elif "/analyze/batch" in endpoint:
                result = await self._analyze_batch(data, timeout)
            else:
                raise TransportError(f"Unknown endpoint: {endpoint}")
            
            return result
        
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timed out after {timeout}s")
        except Exception as e:
            raise TransportError(f"Local execution failed: {str(e)}") from e
    
    async def stream_request(
        self,
        endpoint: str,
        data: dict[str, Any],
        timeout: float | None = None
    ) -> AsyncIterator[StreamEvent]:
        """
        Send a streaming request to local engine.
        
        Args:
            endpoint: API endpoint
            data: Request payload
            timeout: Request timeout
        
        Yields:
            Stream events as they arrive
        
        Raises:
            TransportError: If execution fails
            TimeoutError: If request times out
        """
        if self._closed:
            raise TransportError("Transport is closed")
        
        try:
            # Emit start event
            yield StreamEvent(
                type="started",
                timestamp=datetime.utcnow(),
                data={"endpoint": endpoint}
            )
            
            # Get engine and execute with streaming
            engine = await self._get_engine()
            
            # Route to appropriate streaming handler
            if "/analyze" in endpoint:
                async for event in self._stream_analysis(engine, data, timeout):
                    yield event
            else:
                raise TransportError(f"Streaming not supported for: {endpoint}")
        
        except asyncio.TimeoutError:
            yield StreamEvent(
                type="error",
                timestamp=datetime.utcnow(),
                error=f"Request timed out after {timeout}s"
            )
        except Exception as e:
            yield StreamEvent(
                type="error",
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def _analyze_text(
        self,
        data: dict[str, Any],
        timeout: float | None
    ) -> dict[str, Any]:
        """Analyze text content."""
        engine = await self._get_engine()
        
        # Execute with timeout if specified
        if timeout:
            result = await asyncio.wait_for(
                engine.analyze_text(
                    content=data.get("content", ""),
                    context=data.get("context"),
                    metadata=data.get("metadata", {})
                ),
                timeout=timeout
            )
        else:
            result = await engine.analyze_text(
                content=data.get("content", ""),
                context=data.get("context"),
                metadata=data.get("metadata", {})
            )
        
        # Convert to response format
        return self._format_analysis_result(result)
    
    async def _analyze_email(
        self,
        data: dict[str, Any],
        timeout: float | None
    ) -> dict[str, Any]:
        """Analyze email."""
        engine = await self._get_engine()
        
        if timeout:
            result = await asyncio.wait_for(
                engine.analyze_email(
                    sender=data.get("sender"),
                    recipient=data.get("recipient"),
                    subject=data.get("subject", ""),
                    body=data.get("body", ""),
                    headers=data.get("headers", {}),
                    metadata=data.get("metadata", {})
                ),
                timeout=timeout
            )
        else:
            result = await engine.analyze_email(
                sender=data.get("sender"),
                recipient=data.get("recipient"),
                subject=data.get("subject", ""),
                body=data.get("body", ""),
                headers=data.get("headers", {}),
                metadata=data.get("metadata", {})
            )
        
        return self._format_analysis_result(result)
    
    async def _analyze_document(
        self,
        data: dict[str, Any],
        timeout: float | None
    ) -> dict[str, Any]:
        """Analyze document."""
        engine = await self._get_engine()
        
        if timeout:
            result = await asyncio.wait_for(
                engine.analyze_document(
                    document_url=data.get("document_url"),
                    document_content=data.get("document_content"),
                    document_type=data.get("document_type"),
                    metadata=data.get("metadata", {})
                ),
                timeout=timeout
            )
        else:
            result = await engine.analyze_document(
                document_url=data.get("document_url"),
                document_content=data.get("document_content"),
                document_type=data.get("document_type"),
                metadata=data.get("metadata", {})
            )
        
        return self._format_analysis_result(result)
    
    async def _analyze_image(
        self,
        data: dict[str, Any],
        timeout: float | None
    ) -> dict[str, Any]:
        """Analyze image."""
        engine = await self._get_engine()
        
        if timeout:
            result = await asyncio.wait_for(
                engine.analyze_image(
                    image_url=data.get("image_url"),
                    image_data=data.get("image_data"),
                    metadata=data.get("metadata", {})
                ),
                timeout=timeout
            )
        else:
            result = await engine.analyze_image(
                image_url=data.get("image_url"),
                image_data=data.get("image_data"),
                metadata=data.get("metadata", {})
            )
        
        return self._format_analysis_result(result)
    
    async def _analyze_transaction(
        self,
        data: dict[str, Any],
        timeout: float | None
    ) -> dict[str, Any]:
        """Analyze transaction."""
        engine = await self._get_engine()
        
        if timeout:
            result = await asyncio.wait_for(
                engine.analyze_transaction(
                    user_id=data.get("user_id"),
                    transaction_id=data.get("transaction_id"),
                    amount=data.get("amount"),
                    currency=data.get("currency", "USD"),
                    merchant=data.get("merchant"),
                    location=data.get("location", {}),
                    device_info=data.get("device_info", {}),
                    metadata=data.get("metadata", {})
                ),
                timeout=timeout
            )
        else:
            result = await engine.analyze_transaction(
                user_id=data.get("user_id"),
                transaction_id=data.get("transaction_id"),
                amount=data.get("amount"),
                currency=data.get("currency", "USD"),
                merchant=data.get("merchant"),
                location=data.get("location", {}),
                device_info=data.get("device_info", {}),
                metadata=data.get("metadata", {})
            )
        
        return self._format_analysis_result(result)
    
    async def _analyze_batch(
        self,
        data: dict[str, Any],
        timeout: float | None
    ) -> dict[str, Any]:
        """Analyze batch of items."""
        engine = await self._get_engine()
        
        items = data.get("items", [])
        analysis_type = data.get("analysis_type")
        concurrency = data.get("concurrency", 5)
        
        results = []
        errors = []
        
        # Process in batches with concurrency limit
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_item(item: dict[str, Any], index: int):
            async with semaphore:
                try:
                    # Route based on analysis type
                    if analysis_type == "text":
                        result = await engine.analyze_text(**item)
                    elif analysis_type == "email":
                        result = await engine.analyze_email(**item)
                    elif analysis_type == "document":
                        result = await engine.analyze_document(**item)
                    elif analysis_type == "image":
                        result = await engine.analyze_image(**item)
                    elif analysis_type == "transaction":
                        result = await engine.analyze_transaction(**item)
                    else:
                        raise ValueError(f"Unknown analysis type: {analysis_type}")
                    
                    return (index, self._format_analysis_result(result), None)
                except Exception as e:
                    return (index, None, {"index": index, "error": str(e)})
        
        # Execute all tasks
        if timeout:
            tasks = [process_item(item, i) for i, item in enumerate(items)]
            completed = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        else:
            tasks = [process_item(item, i) for i, item in enumerate(items)]
            completed = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sort results by index
        for index, result, error in sorted(completed, key=lambda x: x[0]):
            results.append(result)
            if error:
                errors.append(error)
        
        return {
            "id": str(uuid4()),
            "total_items": len(items),
            "completed": len([r for r in results if r is not None]),
            "failed": len(errors),
            "results": results,
            "errors": errors,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def _stream_analysis(
        self,
        engine: Any,
        data: dict[str, Any],
        timeout: float | None
    ) -> AsyncIterator[StreamEvent]:
        """Stream analysis events from engine."""
        # Subscribe to engine events
        async for event in engine.analyze_stream(**data):
            # Convert engine event to StreamEvent
            yield StreamEvent(
                type=event.get("type", "progress"),
                timestamp=datetime.utcnow(),
                data=event.get("data", {}),
                agent_id=event.get("agent_id"),
                progress=event.get("progress"),
                result=event.get("result"),
                error=event.get("error")
            )
    
    def _format_analysis_result(self, result: Any) -> dict[str, Any]:
        """Format engine result to API response format."""
        # Convert AnalysisResult to dict
        if hasattr(result, "model_dump"):
            return result.model_dump()
        elif hasattr(result, "dict"):
            return result.dict()
        
        # Manual conversion for legacy result format
        return {
            "id": getattr(result, "id", str(uuid4())),
            "status": getattr(result, "status", AnalysisStatus.COMPLETED.value),
            "fraud_detected": getattr(result, "fraud_detected", False),
            "fraud_score": getattr(result, "fraud_score", 0.0),
            "fraud_types": getattr(result, "fraud_types", []),
            "confidence": getattr(result, "confidence", 0.0),
            "reasoning": getattr(result, "reasoning", ""),
            "evidence": [
                e.model_dump() if hasattr(e, "model_dump") else e
                for e in getattr(result, "evidence", [])
            ],
            "recommended_actions": getattr(result, "recommended_actions", []),
            "processing_time_ms": getattr(result, "processing_time_ms", None),
            "created_at": getattr(result, "created_at", datetime.utcnow()).isoformat(),
            "metadata": getattr(result, "metadata", {})
        }
    
    async def close(self) -> None:
        """Close transport and cleanup resources."""
        self._closed = True
        
        # Cleanup engine if needed
        if self._engine and hasattr(self._engine, "close"):
            await self._engine.close()
