"""Integration tests for Phase 2 components."""

import pytest
from typing import Dict, Any

from fraudlens.agents.base import BaseAgent, AgentCapabilities
from fraudlens.agents.coordinator import AgentCoordinator
from fraudlens.events.bus import EventBus
from fraudlens.llm.schemas import (
    FraudAnalysisOutput,
    FraudType,
    Severity,
    Evidence,
    EvidenceType,
    Action,
    ActionType,
)
from fraudlens.llm.tools import ToolRegistry
from fraudlens.streaming.generator import StreamChunk, ChunkType


class MockFraudAgent(BaseAgent):
    """Mock fraud detection agent for testing."""

    def __init__(
        self,
        name: str,
        event_bus: EventBus,
        tool_registry: ToolRegistry,
        fraud_score: float = 0.8,
    ):
        super().__init__(
            agent_id=name,
            agent_type="mock_agent",
            fraud_types=[FraudType.PHISHING.value],
            event_bus=event_bus,
            tool_registry=tool_registry,
        )
        self.fraud_score = fraud_score

    async def _setup(self) -> None:
        """Agent-specific setup logic."""
        # Mock setup - nothing to do
        pass

    async def analyze(self, data: Dict[str, Any]) -> FraudAnalysisOutput:
        """Analyze for fraud."""
        return FraudAnalysisOutput(
            fraud_detected=self.fraud_score > 0.5,
            confidence=self.fraud_score,
            fraud_types=[FraudType.PHISHING],
            severity=Severity.HIGH if self.fraud_score > 0.7 else Severity.MEDIUM,
            fraud_score=self.fraud_score,
            reasoning=f"{self.name} detected fraud",
            evidence=[
                Evidence(
                    type=EvidenceType.PATTERN_MATCH,
                    description=f"Evidence from {self.name}",
                    confidence=self.fraud_score,
                    source=self.name,
                )
            ],
            recommended_actions=[
                Action(
                    type=ActionType.FLAG_FOR_REVIEW,
                    priority=1,
                    description="Review required",
                )
            ],
        )

    async def cleanup(self) -> None:
        """Cleanup agent."""
        self._initialized = False


class TestAgentIntegration:
    """Test agent integration."""

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self):
        """Test complete agent lifecycle."""
        event_bus = EventBus()
        tool_registry = ToolRegistry()
        agent = MockFraudAgent("test_agent", event_bus, tool_registry)

        # Initialize
        await agent.initialize()
        assert agent._initialized

        # Analyze
        result = await agent.analyze({"text": "test"})
        assert isinstance(result, FraudAnalysisOutput)
        assert result.fraud_detected is True

        # Cleanup
        await agent.cleanup()
        assert not agent._initialized

    @pytest.mark.asyncio
    async def test_agent_streaming(self):
        """Test agent streaming analysis."""
        event_bus = EventBus()
        tool_registry = ToolRegistry()
        agent = MockFraudAgent("test_agent", event_bus, tool_registry)

        await agent.initialize()

        chunks = []
        async for chunk in agent.analyze_stream({"text": "test"}):
            chunks.append(chunk)

        # Should get started, progress, and complete chunks
        assert len(chunks) > 0
        assert any(c.type == ChunkType.STARTED for c in chunks)
        assert any(c.type == ChunkType.COMPLETE for c in chunks)

        await agent.cleanup()


class TestCoordinatorIntegration:
    """Test coordinator integration."""

    @pytest.mark.asyncio
    async def test_coordinator_with_multiple_agents(self):
        """Test coordinator with multiple agents."""
        event_bus = EventBus()
        tool_registry = ToolRegistry()

        # Create agents with different fraud scores
        agent1 = MockFraudAgent("agent1", event_bus, tool_registry, fraud_score=0.8)
        agent2 = MockFraudAgent("agent2", event_bus, tool_registry, fraud_score=0.9)
        agent3 = MockFraudAgent("agent3", event_bus, tool_registry, fraud_score=0.7)

        coordinator = AgentCoordinator(agents=[agent1, agent2, agent3])
        coordinator.consensus_threshold = 0.5

        await coordinator.initialize()

        # Coordinate analysis
        result = await coordinator.analyze({"text": "test"})

        assert isinstance(result, FraudAnalysisOutput)
        assert result.fraud_detected is True
        # Consensus should be around average
        assert 0.7 <= result.fraud_score <= 0.9
        # Should have evidence from all agents
        assert len(result.evidence) == 3

        await coordinator.cleanup()

    @pytest.mark.asyncio
    async def test_coordinator_streaming(self):
        """Test coordinator streaming."""
        event_bus = EventBus()
        tool_registry = ToolRegistry()

        agent1 = MockFraudAgent("agent1", event_bus, tool_registry, fraud_score=0.8)
        agent2 = MockFraudAgent("agent2", event_bus, tool_registry, fraud_score=0.7)

        coordinator = AgentCoordinator(agents=[agent1, agent2])

        await coordinator.initialize()

        chunks = []
        async for chunk in coordinator.analyze_stream({"text": "test"}):
            chunks.append(chunk)

        # Should get chunks from both agents plus final result
        assert len(chunks) > 0
        assert any(c.type == ChunkType.STARTED for c in chunks)
        assert any(c.type == ChunkType.COMPLETE for c in chunks)

        await coordinator.cleanup()

    @pytest.mark.asyncio
    async def test_coordinator_consensus_building(self):
        """Test coordinator consensus building."""
        event_bus = EventBus()
        tool_registry = ToolRegistry()

        # One agent says fraud, two say no fraud
        agent1 = MockFraudAgent("agent1", event_bus, tool_registry, fraud_score=0.9)
        agent2 = MockFraudAgent("agent2", event_bus, tool_registry, fraud_score=0.3)
        agent3 = MockFraudAgent("agent3", event_bus, tool_registry, fraud_score=0.2)

        coordinator = AgentCoordinator(agents=[agent1, agent2, agent3])
        coordinator.consensus_threshold = 0.5

        await coordinator.initialize()
        result = await coordinator.analyze({"text": "test"})

        # Consensus should reflect mixed signals
        assert 0.2 <= result.fraud_score <= 0.9
        # At high consensus threshold, might not detect fraud
        # with only 1/3 agents detecting it

        await coordinator.cleanup()


class TestFullWorkflow:
    """Test complete workflow integration."""

    @pytest.mark.asyncio
    async def test_end_to_end_fraud_detection(self):
        """Test end-to-end fraud detection workflow."""
        # Setup
        event_bus = EventBus()
        tool_registry = ToolRegistry()

        # Track events
        events_received = []

        async def track_event(event):
            events_received.append(event)

        event_bus.subscribe("fraud.*", track_event)

        # Create agents
        phishing_agent = MockFraudAgent(
            "phishing_agent", event_bus, tool_registry, fraud_score=0.85
        )
        social_eng_agent = MockFraudAgent(
            "social_eng_agent", event_bus, tool_registry, fraud_score=0.75
        )

        # Create coordinator
        coordinator = AgentCoordinator(agents=[phishing_agent, social_eng_agent])

        # Initialize
        await coordinator.initialize()

        # Analyze suspicious email
        email_data = {
            "subject": "URGENT: Verify your account",
            "body": "Click here immediately to avoid account closure",
            "sender": "noreply@suspicious-domain.com",
        }

        result = await coordinator.analyze(email_data)

        # Verify results
        assert result.fraud_detected is True
        assert result.confidence > 0.7
        assert FraudType.PHISHING in result.fraud_types
        assert len(result.evidence) > 0
        assert len(result.recommended_actions) > 0

        # Verify events were emitted
        assert len(events_received) > 0

        # Cleanup
        await coordinator.cleanup()

    @pytest.mark.asyncio
    async def test_streaming_workflow(self):
        """Test streaming workflow."""
        event_bus = EventBus()
        tool_registry = ToolRegistry()

        agent1 = MockFraudAgent("agent1", event_bus, tool_registry, fraud_score=0.8)
        agent2 = MockFraudAgent("agent2", event_bus, tool_registry, fraud_score=0.7)

        coordinator = AgentCoordinator(agents=[agent1, agent2])
        await coordinator.initialize()

        # Stream analysis
        all_chunks = []
        final_result = None

        async for chunk in coordinator.analyze_stream({"text": "test"}):
            all_chunks.append(chunk)
            if chunk.type == ChunkType.COMPLETE:
                final_result = chunk.data

        # Verify streaming progression
        assert len(all_chunks) > 0
        assert any(c.type == ChunkType.STARTED for c in all_chunks)
        assert any(c.type == ChunkType.PROGRESS for c in all_chunks)
        assert any(c.type == ChunkType.COMPLETE for c in all_chunks)

        # Verify final result
        assert final_result is not None
        assert isinstance(final_result, FraudAnalysisOutput)

        await coordinator.cleanup()
