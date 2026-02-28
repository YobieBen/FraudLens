"""
Phase 2 Integration Demo: Multi-Agent Fraud Detection with Streaming

This demo showcases the complete Phase 2 modernization including:
- Multi-agent architecture with coordination
- Structured LLM outputs
- Tool calling framework
- Real-time streaming results
- Reasoning loops
- Event-driven architecture

Run with: python -m examples.phase2_demo
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Type

from pydantic import BaseModel

from fraudlens.agents.base import BaseAgent, AgentCapabilities
from fraudlens.agents.coordinator import AgentCoordinator
from fraudlens.agents.specialists.phishing_agent import PhishingAgent
from fraudlens.events.bus import EventBus
from fraudlens.llm.schemas import (
    Action,
    ActionType,
    Evidence,
    EvidenceType,
    FraudAnalysisOutput,
    FraudType,
    Severity,
)
from fraudlens.llm.tools import ToolRegistry, get_default_registry
from fraudlens.llm.orchestrator import (
    LLMConfig,
    LLMOrchestrator,
    LLMProvider,
    LLMClient,
)
from fraudlens.streaming.generator import ChunkType


class MockLLMClient:
    """Mock LLM client for demo purposes."""

    async def generate(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """Mock LLM generation."""
        # In real use, this would call actual LLM APIs
        if response_model == FraudAnalysisOutput:
            return FraudAnalysisOutput(
                fraud_detected=True,
                confidence=0.85,
                fraud_types=[FraudType.PHISHING],
                severity=Severity.HIGH,
                fraud_score=0.85,
                reasoning="Suspicious email with urgency and URL indicators",
                evidence=[
                    Evidence(
                        type=EvidenceType.URL_ANALYSIS,
                        description="Suspicious domain detected",
                        confidence=0.9,
                        source="url_analyzer",
                    )
                ],
                recommended_actions=[
                    Action(
                        type=ActionType.BLOCK_USER,
                        priority=1,
                        description="Block sender immediately",
                    )
                ],
            )
        return "Mock response"

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        """Mock streaming."""
        for chunk in ["Analyzing", " email", " content", "..."]:
            await asyncio.sleep(0.1)
            yield chunk


class SimpleFraudAgent(BaseAgent):
    """Simple fraud detection agent for demo."""

    def __init__(
        self,
        name: str,
        event_bus: EventBus,
        tool_registry: ToolRegistry,
        fraud_type: FraudType,
        base_confidence: float = 0.75,
    ):
        super().__init__(
            agent_id=name,
            agent_type="simple_fraud_agent",
            fraud_types=[fraud_type.value],
            event_bus=event_bus,
            tool_registry=tool_registry,
        )
        self.fraud_type = fraud_type
        self.base_confidence = base_confidence

    async def _setup(self) -> None:
        """Agent-specific setup logic."""
        print(f"✓ {self.agent_id} initialized")

    async def analyze(
        self, input_data: Dict[str, Any], context: Dict[str, Any] | None = None
    ) -> FraudAnalysisOutput:
        """Analyze for fraud."""
        # Simulate analysis
        await asyncio.sleep(0.2)

        text = str(input_data)
        is_fraud = any(
            keyword in text.lower()
            for keyword in ["urgent", "verify", "suspend", "click here", "limited time"]
        )

        confidence = self.base_confidence if is_fraud else 0.3

        return FraudAnalysisOutput(
            fraud_detected=is_fraud,
            confidence=confidence,
            fraud_types=[self.fraud_type] if is_fraud else [],
            severity=Severity.HIGH if confidence > 0.7 else Severity.MEDIUM,
            fraud_score=confidence,
            reasoning=f"{self.agent_id} analysis: {'Fraud indicators detected' if is_fraud else 'No fraud detected'}",
            evidence=[
                Evidence(
                    type=EvidenceType.PATTERN_MATCH,
                    description=f"Analysis by {self.agent_id}",
                    confidence=confidence,
                    source=self.agent_id,
                )
            ],
            recommended_actions=[
                Action(
                    type=ActionType.FLAG_FOR_REVIEW if is_fraud else ActionType.MONITOR,
                    priority=1 if is_fraud else 3,
                    description="Review for potential fraud" if is_fraud else "Monitor",
                )
            ],
        )


async def demo_basic_analysis():
    """Demo 1: Basic fraud analysis."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Fraud Analysis")
    print("=" * 70)

    event_bus = EventBus()
    tool_registry = get_default_registry()

    # Create agent
    agent = SimpleFraudAgent(
        name="phishing_detector",
        event_bus=event_bus,
        tool_registry=tool_registry,
        fraud_type=FraudType.PHISHING,
    )

    await agent.initialize()

    # Analyze suspicious email
    email = {
        "subject": "URGENT: Verify your account",
        "body": "Your account will be suspended. Click here immediately!",
        "sender": "noreply@suspicious-site.com",
    }

    print("\n📧 Analyzing Email:")
    print(f"  Subject: {email['subject']}")
    print(f"  Sender: {email['sender']}")

    result = await agent.analyze(email)

    print("\n🔍 Analysis Results:")
    print(f"  Fraud Detected: {result.fraud_detected}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Fraud Score: {result.fraud_score:.2f}")
    print(f"  Severity: {result.severity}")
    print(f"  Reasoning: {result.reasoning}")

    await agent.cleanup()


async def demo_multi_agent_coordination():
    """Demo 2: Multi-agent coordination with consensus."""
    print("\n" + "=" * 70)
    print("DEMO 2: Multi-Agent Coordination")
    print("=" * 70)

    event_bus = EventBus()
    tool_registry = get_default_registry()

    # Create multiple specialist agents
    agents = [
        SimpleFraudAgent(
            "phishing_agent",
            event_bus,
            tool_registry,
            FraudType.PHISHING,
            base_confidence=0.85,
        ),
        SimpleFraudAgent(
            "social_eng_agent",
            event_bus,
            tool_registry,
            FraudType.SOCIAL_ENGINEERING,
            base_confidence=0.80,
        ),
        SimpleFraudAgent(
            "urgency_agent",
            event_bus,
            tool_registry,
            FraudType.PHISHING,
            base_confidence=0.75,
        ),
    ]

    # Create coordinator
    coordinator = AgentCoordinator(agents=agents)
    coordinator.consensus_threshold = 0.6

    await coordinator.initialize()

    # Analyze suspicious message
    message = {
        "text": "URGENT! Your account will be suspended in 24 hours. Verify now at http://suspicious-link.com"
    }

    print("\n📨 Analyzing Message:")
    print(f"  Text: {message['text']}")
    print(f"\n🤖 Coordinating {len(agents)} agents...")

    result = await coordinator.analyze(message)

    print("\n📊 Consensus Results:")
    print(f"  Fraud Detected: {result.fraud_detected}")
    print(f"  Consensus Score: {result.fraud_score:.2f}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Severity: {result.severity}")
    print(f"\n📋 Evidence ({len(result.evidence)} sources):")
    for i, evidence in enumerate(result.evidence, 1):
        print(f"    {i}. [{evidence.source}] {evidence.description} ({evidence.confidence:.2%})")

    print(f"\n⚡ Recommended Actions ({len(result.recommended_actions)}):")
    for i, action in enumerate(result.recommended_actions, 1):
        print(f"    {i}. {action.type} (Priority: {action.priority})")
        print(f"       {action.description}")

    await coordinator.cleanup()


async def demo_streaming_analysis():
    """Demo 3: Real-time streaming analysis."""
    print("\n" + "=" * 70)
    print("DEMO 3: Streaming Analysis")
    print("=" * 70)

    event_bus = EventBus()
    tool_registry = get_default_registry()

    agents = [
        SimpleFraudAgent(
            "agent_1", event_bus, tool_registry, FraudType.PHISHING, 0.8
        ),
        SimpleFraudAgent(
            "agent_2", event_bus, tool_registry, FraudType.SOCIAL_ENGINEERING, 0.75
        ),
    ]

    coordinator = AgentCoordinator(agents=agents)
    await coordinator.initialize()

    message = {
        "text": "Limited time offer! Click now to claim your prize. Account verification required."
    }

    print("\n📨 Analyzing Message (Streaming):")
    print(f"  Text: {message['text']}")
    print("\n🔄 Real-time Updates:")

    async for chunk in coordinator.analyze_stream(message):
        if chunk.type == ChunkType.STARTED:
            print(f"  ▶ Started: {chunk.data.get('message', 'Analysis beginning')}")
        elif chunk.type == ChunkType.PROGRESS:
            progress = chunk.data.get("progress", 0)
            agent = chunk.data.get("agent", "unknown")
            print(f"  ⋯ Progress: {agent} - {progress:.0%}")
        elif chunk.type == ChunkType.AGENT_RESULT:
            agent = chunk.data.get("agent", "unknown")
            score = chunk.data.get("fraud_score", 0)
            print(f"  ✓ {agent}: fraud_score={score:.2f}")
        elif chunk.type == ChunkType.COMPLETE:
            result = chunk.data
            print(f"\n✅ Complete!")
            # Handle dict or FraudAnalysisOutput
            if isinstance(result, dict):
                print(f"  Final Score: {result.get('fraud_score', 0.0):.2f}")
                print(f"  Fraud Detected: {result.get('fraud_detected', False)}")
            else:
                print(f"  Final Score: {result.fraud_score:.2f}")
                print(f"  Fraud Detected: {result.fraud_detected}")

    await coordinator.cleanup()


async def demo_llm_orchestrator():
    """Demo 4: LLM orchestrator with retry and fallback."""
    print("\n" + "=" * 70)
    print("DEMO 4: LLM Orchestrator")
    print("=" * 70)

    tool_registry = get_default_registry()
    orchestrator = LLMOrchestrator(tool_registry=tool_registry)

    # Register mock LLM providers (in real use, these would be actual LLM clients)
    print("\n🔧 Registering LLM Providers:")
    
    orchestrator.register_provider(
        LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-5-sonnet-20241022",
        ),
        MockLLMClient(),
        priority=100,
    )
    print("  ✓ Anthropic Claude (Priority: 100)")

    orchestrator.register_provider(
        LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o"),
        MockLLMClient(),
        priority=50,
    )
    print("  ✓ OpenAI GPT-4 (Priority: 50)")

    print("\n🤖 Generating Analysis...")
    response = await orchestrator.generate(
        messages=[
            {
                "role": "user",
                "content": "Analyze this email for phishing: URGENT account verification required",
            }
        ],
        response_model=FraudAnalysisOutput,
    )

    print("\n📊 Results:")
    print(f"  Provider Used: {response.provider}")
    print(f"  Model: {response.model}")
    print(f"  Tokens: {response.tokens_used}")
    print(f"  Latency: {response.latency_ms:.2f}ms")
    print(f"  Cost: ${response.cost_usd:.4f}")
    print(f"\n  Fraud Detected: {response.content.fraud_detected}")
    print(f"  Confidence: {response.content.confidence:.2%}")

    # Show stats
    stats = orchestrator.get_stats()
    print("\n📈 Orchestrator Statistics:")
    for provider, provider_stats in stats.items():
        if provider_stats["requests"] > 0:
            print(f"  {provider}:")
            print(f"    Requests: {provider_stats['requests']}")
            print(f"    Success Rate: {provider_stats['successes']}/{provider_stats['requests']}")
            print(f"    Total Tokens: {provider_stats['total_tokens']}")


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("FraudLens Phase 2 Integration Demo")
    print("Modern LLM & Agentic Architecture")
    print("=" * 70)

    try:
        await demo_basic_analysis()
        await asyncio.sleep(1)

        await demo_multi_agent_coordination()
        await asyncio.sleep(1)

        await demo_streaming_analysis()
        await asyncio.sleep(1)

        await demo_llm_orchestrator()

        print("\n" + "=" * 70)
        print("✅ All Demos Complete!")
        print("=" * 70)
        print("\nPhase 2 demonstrates:")
        print("  ✓ Multi-agent architecture with coordination")
        print("  ✓ Structured LLM outputs (Pydantic models)")
        print("  ✓ Tool calling framework")
        print("  ✓ Real-time streaming results")
        print("  ✓ Consensus building from multiple agents")
        print("  ✓ LLM orchestration with retry/fallback")
        print("  ✓ Event-driven architecture")
        print("  ✓ Circuit breaker and rate limiting")
        print("\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
