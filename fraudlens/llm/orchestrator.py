"""
LLM Orchestrator for managing interactions with multiple LLM providers.

Features:
- Retry logic with exponential backoff
- Fallback to alternative providers
- Rate limiting and circuit breaker
- Streaming and non-streaming support
- Cost tracking and token usage monitoring
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol, Type

from pydantic import BaseModel

from fraudlens.llm.schemas import FraudAnalysisOutput, ToolCall
from fraudlens.llm.tools import ToolRegistry


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    LOCAL = "local"


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    concurrent_requests: int = 10


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""

    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 30.0
    retry: RetryConfig = field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)


class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    async def generate(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """Generate a response from the LLM."""
        ...

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM."""
        ...


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: Any
    provider: LLMProvider
    model: str
    tokens_used: int
    latency_ms: float
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """State for a circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_tokens = config.requests_per_minute
        self.token_tokens = config.tokens_per_minute
        self.last_refill = time.time()
        self.active_requests = 0
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 1000) -> None:
        """Acquire rate limit tokens."""
        async with self._lock:
            # Refill tokens based on time passed
            now = time.time()
            time_passed = now - self.last_refill
            if time_passed >= 60.0:  # Refill every minute
                self.request_tokens = self.config.requests_per_minute
                self.token_tokens = self.config.tokens_per_minute
                self.last_refill = now

            # Wait for concurrent request slot
            while self.active_requests >= self.config.concurrent_requests:
                await asyncio.sleep(0.1)

            # Wait for request tokens
            while self.request_tokens <= 0:
                await asyncio.sleep(1.0)
                # Check for refill
                now = time.time()
                if now - self.last_refill >= 60.0:
                    self.request_tokens = self.config.requests_per_minute
                    self.token_tokens = self.config.tokens_per_minute
                    self.last_refill = now

            # Wait for token budget
            while self.token_tokens < estimated_tokens:
                await asyncio.sleep(1.0)
                # Check for refill
                now = time.time()
                if now - self.last_refill >= 60.0:
                    self.request_tokens = self.config.requests_per_minute
                    self.token_tokens = self.config.tokens_per_minute
                    self.last_refill = now

            # Consume tokens
            self.request_tokens -= 1
            self.token_tokens -= estimated_tokens
            self.active_requests += 1

    def release(self) -> None:
        """Release concurrent request slot."""
        self.active_requests = max(0, self.active_requests - 1)


class CircuitBreaker:
    """Circuit breaker for LLM providers."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.states: Dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)

    def get_state(self, provider: str) -> CircuitState:
        """Get current circuit state for a provider."""
        state = self.states[provider]

        # Check if we should transition from OPEN to HALF_OPEN
        if state.state == CircuitState.OPEN:
            if state.last_failure_time:
                time_since_failure = datetime.now() - state.last_failure_time
                if time_since_failure.total_seconds() >= self.config.timeout:
                    state.state = CircuitState.HALF_OPEN
                    state.success_count = 0

        return state.state

    def record_success(self, provider: str) -> None:
        """Record a successful request."""
        state = self.states[provider]

        if state.state == CircuitState.HALF_OPEN:
            state.success_count += 1
            if state.success_count >= self.config.success_threshold:
                state.state = CircuitState.CLOSED
                state.failure_count = 0
                state.success_count = 0
        elif state.state == CircuitState.CLOSED:
            state.failure_count = max(0, state.failure_count - 1)

    def record_failure(self, provider: str) -> None:
        """Record a failed request."""
        state = self.states[provider]
        state.failure_count += 1
        state.last_failure_time = datetime.now()

        if state.state == CircuitState.HALF_OPEN:
            # Failed during recovery, go back to OPEN
            state.state = CircuitState.OPEN
            state.success_count = 0
        elif state.failure_count >= self.config.failure_threshold:
            state.state = CircuitState.OPEN

    def is_available(self, provider: str) -> bool:
        """Check if provider is available."""
        return self.get_state(provider) != CircuitState.OPEN


class LLMOrchestrator:
    """
    Orchestrates LLM interactions with retry, fallback, and rate limiting.
    
    Example:
        ```python
        orchestrator = LLMOrchestrator()
        
        # Register providers with fallback chain
        orchestrator.register_provider(
            LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-5-sonnet-20241022",
            ),
            client=anthropic_client,
        )
        orchestrator.register_provider(
            LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4o",
            ),
            client=openai_client,
        )
        
        # Generate with automatic fallback
        response = await orchestrator.generate(
            messages=[{"role": "user", "content": "Analyze this..."}],
            response_model=FraudAnalysisOutput,
        )
        ```
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.providers: List[tuple[LLMConfig, LLMClient]] = []
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig())
        self.tool_registry = tool_registry or ToolRegistry()
        self.stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "total_latency": 0.0,
            }
        )

    def register_provider(
        self,
        config: LLMConfig,
        client: LLMClient,
        priority: int = 0,
    ) -> None:
        """Register an LLM provider."""
        self.providers.append((config, client))
        # Sort by priority (higher first)
        self.providers.sort(key=lambda x: priority, reverse=True)

        # Create rate limiter
        provider_key = f"{config.provider}:{config.model}"
        self.rate_limiters[provider_key] = RateLimiter(config.rate_limit)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[List[str]] = None,
        response_model: Optional[Type[BaseModel]] = None,
        max_fallbacks: int = 3,
    ) -> LLMResponse:
        """
        Generate a response with automatic retry and fallback.
        
        Args:
            messages: Conversation messages
            system: System prompt
            tools: List of tool names to make available
            response_model: Pydantic model for structured output
            max_fallbacks: Maximum number of providers to try
            
        Returns:
            LLMResponse with the generated content
        """
        # Convert tools to LLM format
        tool_specs = None
        if tools:
            tool_specs = [
                self.tool_registry.get_tool_spec(name) for name in tools
            ]

        last_error = None
        attempts = 0

        for config, client in self.providers[:max_fallbacks]:
            provider_key = f"{config.provider}:{config.model}"

            # Check circuit breaker
            if not self.circuit_breaker.is_available(provider_key):
                continue

            # Try with retries
            for attempt in range(config.retry.max_attempts):
                attempts += 1
                try:
                    # Rate limiting
                    rate_limiter = self.rate_limiters[provider_key]
                    await rate_limiter.acquire()

                    # Track stats
                    self.stats[provider_key]["requests"] += 1
                    start_time = time.time()

                    # Generate
                    result = await asyncio.wait_for(
                        client.generate(
                            messages=messages,
                            system=system,
                            tools=tool_specs,
                            response_model=response_model,
                        ),
                        timeout=config.timeout,
                    )

                    # Calculate metrics
                    latency_ms = (time.time() - start_time) * 1000
                    tokens_used = self._estimate_tokens(result)
                    cost = self._calculate_cost(
                        config.provider, config.model, tokens_used
                    )

                    # Update stats
                    self.stats[provider_key]["successes"] += 1
                    self.stats[provider_key]["total_tokens"] += tokens_used
                    self.stats[provider_key]["total_cost"] += cost
                    self.stats[provider_key]["total_latency"] += latency_ms

                    # Record success
                    self.circuit_breaker.record_success(provider_key)
                    rate_limiter.release()

                    return LLMResponse(
                        content=result,
                        provider=config.provider,
                        model=config.model,
                        tokens_used=tokens_used,
                        latency_ms=latency_ms,
                        cost_usd=cost,
                        metadata={"attempts": attempts},
                    )

                except asyncio.TimeoutError:
                    last_error = f"Timeout after {config.timeout}s"
                    rate_limiter.release()
                except Exception as e:
                    last_error = str(e)
                    rate_limiter.release()
                    self.stats[provider_key]["failures"] += 1
                    self.circuit_breaker.record_failure(provider_key)

                # Exponential backoff
                if attempt < config.retry.max_attempts - 1:
                    delay = min(
                        config.retry.initial_delay
                        * (config.retry.exponential_base ** attempt),
                        config.retry.max_delay,
                    )
                    if config.retry.jitter:
                        delay *= 0.5 + (asyncio.get_event_loop().time() % 1) * 0.5
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"All LLM providers failed after {attempts} attempts. Last error: {last_error}"
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        # Use first available provider
        for config, client in self.providers:
            provider_key = f"{config.provider}:{config.model}"

            if not self.circuit_breaker.is_available(provider_key):
                continue

            try:
                # Rate limiting
                rate_limiter = self.rate_limiters[provider_key]
                await rate_limiter.acquire()

                # Convert tools to LLM format
                tool_specs = None
                if tools:
                    tool_specs = [
                        self.tool_registry.get_tool_spec(name) for name in tools
                    ]

                # Stream
                async for chunk in client.generate_stream(
                    messages=messages,
                    system=system,
                    tools=tool_specs,
                ):
                    yield chunk

                rate_limiter.release()
                self.circuit_breaker.record_success(provider_key)
                return

            except Exception as e:
                rate_limiter.release()
                self.circuit_breaker.record_failure(provider_key)
                continue

        raise RuntimeError("No LLM providers available for streaming")

    def _estimate_tokens(self, result: Any) -> int:
        """Estimate token count from result."""
        if isinstance(result, BaseModel):
            text = result.model_dump_json()
        else:
            text = str(result)
        # Rough estimate: 1 token ~= 4 characters
        return len(text) // 4

    def _calculate_cost(
        self, provider: LLMProvider, model: str, tokens: int
    ) -> float:
        """Calculate cost in USD."""
        # Simplified cost calculation (would need real pricing data)
        cost_per_1k_tokens = {
            LLMProvider.ANTHROPIC: 0.015,  # Claude 3.5 Sonnet
            LLMProvider.OPENAI: 0.01,  # GPT-4o
            LLMProvider.GOOGLE: 0.0125,  # Gemini 1.5 Pro
            LLMProvider.AZURE: 0.01,
            LLMProvider.LOCAL: 0.0,
        }
        rate = cost_per_1k_tokens.get(provider, 0.01)
        return (tokens / 1000.0) * rate

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics."""
        return dict(self.stats)

    def reset_circuit_breaker(self, provider: Optional[str] = None) -> None:
        """Reset circuit breaker for a provider or all providers."""
        if provider:
            self.circuit_breaker.states[provider] = CircuitBreakerState()
        else:
            self.circuit_breaker.states.clear()
