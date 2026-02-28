"""Tests for LLM orchestrator."""

import asyncio
import pytest
from typing import Any, AsyncIterator, Dict, List, Optional, Type
from pydantic import BaseModel

from fraudlens.llm.orchestrator import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    LLMClient,
    LLMConfig,
    LLMOrchestrator,
    LLMProvider,
    RateLimiter,
    RateLimitConfig,
    RetryConfig,
)
from fraudlens.llm.schemas import FraudAnalysisOutput, FraudType, Severity


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, should_fail: bool = False, delay: float = 0.0):
        self.should_fail = should_fail
        self.delay = delay
        self.call_count = 0

    async def generate(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """Mock generate method."""
        self.call_count += 1

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise Exception("Mock failure")

        if response_model:
            # Return mock structured output
            if response_model == FraudAnalysisOutput:
                return FraudAnalysisOutput(
                    fraud_detected=True,
                    confidence=0.8,
                    fraud_types=[FraudType.PHISHING],
                    severity=Severity.HIGH,
                    fraud_score=0.8,
                    reasoning="Mock analysis",
                )

        return "Mock response"

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        """Mock streaming method."""
        for i in range(3):
            await asyncio.sleep(0.01)
            yield f"chunk_{i}"


class TestRateLimiter:
    """Test RateLimiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests(self):
        """Test that rate limiter allows requests within limits."""
        config = RateLimitConfig(
            requests_per_minute=10, tokens_per_minute=1000, concurrent_requests=5
        )
        limiter = RateLimiter(config)

        # Should allow request
        await limiter.acquire(estimated_tokens=100)
        limiter.release()

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_limit(self):
        """Test concurrent request limiting."""
        config = RateLimitConfig(
            requests_per_minute=100, tokens_per_minute=10000, concurrent_requests=2
        )
        limiter = RateLimiter(config)

        # Acquire 2 slots (max)
        await limiter.acquire(estimated_tokens=100)
        await limiter.acquire(estimated_tokens=100)

        # Third acquire should block briefly
        acquired = False

        async def try_acquire():
            nonlocal acquired
            await limiter.acquire(estimated_tokens=100)
            acquired = True
            limiter.release()

        task = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.05)
        assert not acquired  # Should still be blocked

        # Release one slot
        limiter.release()
        await asyncio.sleep(0.15)
        assert acquired  # Should now be acquired

        limiter.release()  # Release second slot
        await task


class TestCircuitBreaker:
    """Test CircuitBreaker."""

    def test_circuit_breaker_initial_state(self):
        """Test initial circuit breaker state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        assert cb.get_state("test_provider") == CircuitState.CLOSED
        assert cb.is_available("test_provider")

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        # Record failures
        for _ in range(3):
            cb.record_failure("test_provider")

        assert cb.get_state("test_provider") == CircuitState.OPEN
        assert not cb.is_available("test_provider")

    def test_circuit_breaker_half_open_after_timeout(self):
        """Test circuit breaker transitions to half-open."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        cb = CircuitBreaker(config)

        # Trigger open
        cb.record_failure("test_provider")
        cb.record_failure("test_provider")
        assert cb.get_state("test_provider") == CircuitState.OPEN

        # Wait for timeout
        import time

        time.sleep(0.15)

        # Should now be half-open
        assert cb.get_state("test_provider") == CircuitState.HALF_OPEN

    def test_circuit_breaker_closes_after_successes(self):
        """Test circuit breaker closes after successful requests."""
        config = CircuitBreakerConfig(
            failure_threshold=2, success_threshold=2, timeout=0.1
        )
        cb = CircuitBreaker(config)

        # Open the circuit
        cb.record_failure("test_provider")
        cb.record_failure("test_provider")
        assert cb.get_state("test_provider") == CircuitState.OPEN

        # Wait for half-open
        import time

        time.sleep(0.15)
        assert cb.get_state("test_provider") == CircuitState.HALF_OPEN

        # Record successes to close
        cb.record_success("test_provider")
        cb.record_success("test_provider")
        assert cb.get_state("test_provider") == CircuitState.CLOSED


class TestLLMOrchestrator:
    """Test LLMOrchestrator."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = LLMOrchestrator()
        assert len(orchestrator.providers) == 0

    def test_register_provider(self):
        """Test registering a provider."""
        orchestrator = LLMOrchestrator()
        client = MockLLMClient()
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o")

        orchestrator.register_provider(config, client)
        assert len(orchestrator.providers) == 1

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful generation."""
        orchestrator = LLMOrchestrator()
        client = MockLLMClient()
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            retry=RetryConfig(max_attempts=1),
        )

        orchestrator.register_provider(config, client)

        response = await orchestrator.generate(
            messages=[{"role": "user", "content": "Test"}]
        )

        assert response.content == "Mock response"
        assert response.provider == LLMProvider.OPENAI
        assert response.model == "gpt-4o"
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_with_retry(self):
        """Test generation with retry on failure."""
        orchestrator = LLMOrchestrator()

        # First client fails
        failing_client = MockLLMClient(should_fail=True)
        config1 = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            retry=RetryConfig(max_attempts=2, initial_delay=0.01),
        )
        orchestrator.register_provider(config1, failing_client, priority=100)

        # Second client succeeds
        success_client = MockLLMClient()
        config2 = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-5-sonnet",
            retry=RetryConfig(max_attempts=1),
        )
        orchestrator.register_provider(config2, success_client, priority=50)

        response = await orchestrator.generate(
            messages=[{"role": "user", "content": "Test"}]
        )

        # Should fall back to second provider
        assert response.provider == LLMProvider.ANTHROPIC
        assert success_client.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_with_structured_output(self):
        """Test generation with structured output."""
        orchestrator = LLMOrchestrator()
        client = MockLLMClient()
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o")

        orchestrator.register_provider(config, client)

        response = await orchestrator.generate(
            messages=[{"role": "user", "content": "Analyze fraud"}],
            response_model=FraudAnalysisOutput,
        )

        assert isinstance(response.content, FraudAnalysisOutput)
        assert response.content.fraud_detected is True

    @pytest.mark.asyncio
    async def test_generate_timeout(self):
        """Test generation timeout."""
        orchestrator = LLMOrchestrator()
        # Client that takes too long
        slow_client = MockLLMClient(delay=1.0)
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            timeout=0.1,  # Very short timeout
            retry=RetryConfig(max_attempts=1),
        )

        orchestrator.register_provider(config, slow_client)

        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            await orchestrator.generate(messages=[{"role": "user", "content": "Test"}])

    @pytest.mark.asyncio
    async def test_generate_all_providers_fail(self):
        """Test when all providers fail."""
        orchestrator = LLMOrchestrator()

        # Both fail
        client1 = MockLLMClient(should_fail=True)
        config1 = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            retry=RetryConfig(max_attempts=1, initial_delay=0.01),
        )
        orchestrator.register_provider(config1, client1)

        client2 = MockLLMClient(should_fail=True)
        config2 = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude",
            retry=RetryConfig(max_attempts=1, initial_delay=0.01),
        )
        orchestrator.register_provider(config2, client2)

        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            await orchestrator.generate(messages=[{"role": "user", "content": "Test"}])

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test streaming generation."""
        orchestrator = LLMOrchestrator()
        client = MockLLMClient()
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o")

        orchestrator.register_provider(config, client)

        chunks = []
        async for chunk in orchestrator.generate_stream(
            messages=[{"role": "user", "content": "Test"}]
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0] == "chunk_0"
        assert chunks[2] == "chunk_2"

    def test_get_stats(self):
        """Test getting usage statistics."""
        orchestrator = LLMOrchestrator()
        stats = orchestrator.get_stats()
        assert isinstance(stats, dict)

    def test_reset_circuit_breaker(self):
        """Test resetting circuit breaker."""
        orchestrator = LLMOrchestrator()
        orchestrator.circuit_breaker.record_failure("test")
        orchestrator.reset_circuit_breaker("test")
        assert orchestrator.circuit_breaker.get_state("test") == CircuitState.CLOSED
