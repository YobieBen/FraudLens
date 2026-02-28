"""
LLM integration for FraudLens.

Modern LLM patterns including structured outputs, tool calling,
and reasoning loops for fraud detection.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from fraudlens.llm.schemas import (
    Action,
    ActionType,
    AgentMessage,
    Evidence,
    EvidenceType,
    FraudAnalysisOutput,
    FraudType,
    ReasoningStep,
    Severity,
    ToolCall,
)
from fraudlens.llm.orchestrator import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    LLMClient,
    LLMConfig,
    LLMOrchestrator,
    LLMProvider,
    LLMResponse,
    RateLimitConfig,
    RateLimiter,
    RetryConfig,
)
from fraudlens.llm.reasoning import ReasoningLoop, create_reasoning_chain
from fraudlens.llm.tools import Tool, ToolRegistry, register_tool

__all__ = [
    # Schemas
    "FraudAnalysisOutput",
    "Evidence",
    "EvidenceType",
    "Action",
    "ActionType",
    "AgentMessage",
    "FraudType",
    "Severity",
    "ReasoningStep",
    "ToolCall",
    # Orchestrator
    "LLMOrchestrator",
    "LLMClient",
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "RetryConfig",
    "RateLimitConfig",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitState",
    "RateLimiter",
    # Tools
    "Tool",
    "ToolRegistry",
    "register_tool",
    # Reasoning
    "ReasoningLoop",
    "create_reasoning_chain",
]
