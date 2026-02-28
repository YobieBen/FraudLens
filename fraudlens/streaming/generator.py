"""
Async generators for streaming fraud detection results.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

from fraudlens.llm.schemas import FraudAnalysisOutput


class ChunkType(str, Enum):
    """Types of stream chunks."""
    
    STARTED = "started"
    PROGRESS = "progress"
    AGENT_RESULT = "agent_result"
    EVIDENCE_FOUND = "evidence_found"
    REASONING_STEP = "reasoning_step"
    TOOL_CALL = "tool_call"
    PARTIAL_RESULT = "partial_result"
    COMPLETE = "complete"
    ERROR = "error"


class StreamChunk(BaseModel):
    """
    A chunk of streaming data.
    
    Represents incremental updates during fraud detection analysis.
    """
    
    type: ChunkType = Field(..., description="Type of chunk")
    data: dict[str, Any] = Field(..., description="Chunk data")
    progress: float = Field(0.0, ge=0, le=1, description="Overall progress (0-1)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "agent_result",
                "data": {
                    "agent_id": "phishing_agent",
                    "fraud_detected": True,
                    "confidence": 0.85
                },
                "progress": 0.33,
                "metadata": {"processing_time_ms": 125}
            }
        }


class AnalysisStream:
    """
    Stream wrapper for fraud analysis.
    
    Provides utilities for creating and managing analysis streams.
    """
    
    @staticmethod
    async def from_analysis(
        analysis_func: callable,
        *args,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Create stream from analysis function.
        
        Args:
            analysis_func: Async function that performs analysis
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
        
        Yields:
            Stream chunks
        """
        # Emit started chunk
        yield StreamChunk(
            type=ChunkType.STARTED,
            data={"status": "started"},
            progress=0.0
        )
        
        try:
            # Run analysis
            result = await analysis_func(*args, **kwargs)
            
            # Emit complete chunk
            yield StreamChunk(
                type=ChunkType.COMPLETE,
                data=result.model_dump() if hasattr(result, "model_dump") else result,
                progress=1.0
            )
        
        except Exception as e:
            # Emit error chunk
            yield StreamChunk(
                type=ChunkType.ERROR,
                data={"error": str(e), "error_type": type(e).__name__},
                progress=0.0
            )
    
    @staticmethod
    async def merge_streams(
        *streams: AsyncIterator[StreamChunk]
    ) -> AsyncIterator[StreamChunk]:
        """
        Merge multiple streams into one.
        
        Args:
            *streams: Streams to merge
        
        Yields:
            Merged stream chunks
        """
        import asyncio
        
        # Convert streams to list for concurrent processing
        stream_list = list(streams)
        
        # Process all streams concurrently
        async def process_stream(stream, stream_id):
            async for chunk in stream:
                chunk.metadata["stream_id"] = stream_id
                yield chunk
        
        # This is a simplified implementation
        # A full implementation would use asyncio.as_completed
        for i, stream in enumerate(stream_list):
            async for chunk in process_stream(stream, i):
                yield chunk
    
    @staticmethod
    def create_progress_chunk(
        message: str,
        progress: float,
        **metadata
    ) -> StreamChunk:
        """
        Create a progress update chunk.
        
        Args:
            message: Progress message
            progress: Progress value (0-1)
            **metadata: Additional metadata
        
        Returns:
            Progress chunk
        """
        return StreamChunk(
            type=ChunkType.PROGRESS,
            data={"message": message},
            progress=progress,
            metadata=metadata
        )
    
    @staticmethod
    def create_agent_result_chunk(
        agent_id: str,
        result: Any,
        progress: float = 0.0
    ) -> StreamChunk:
        """
        Create agent result chunk.
        
        Args:
            agent_id: ID of agent that produced result
            result: Agent result
            progress: Overall progress
        
        Returns:
            Agent result chunk
        """
        return StreamChunk(
            type=ChunkType.AGENT_RESULT,
            data={
                "agent_id": agent_id,
                "result": result.model_dump() if hasattr(result, "model_dump") else result
            },
            progress=progress,
            metadata={"agent_id": agent_id}
        )
    
    @staticmethod
    def create_complete_chunk(
        result: FraudAnalysisOutput
    ) -> StreamChunk:
        """
        Create completion chunk.
        
        Args:
            result: Final analysis result
        
        Returns:
            Complete chunk
        """
        return StreamChunk(
            type=ChunkType.COMPLETE,
            data=result.model_dump(),
            progress=1.0
        )
