"""Tests for LLM schemas and structured outputs."""

import pytest
from datetime import datetime

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


class TestEvidence:
    """Test Evidence model."""

    def test_evidence_creation(self):
        """Test creating evidence."""
        evidence = Evidence(
            type=EvidenceType.URL_ANALYSIS,
            description="Suspicious URL detected",
            confidence=0.85,
            source="url_analyzer",
            metadata={"domain": "evil.com"},
        )

        assert evidence.type == EvidenceType.URL_ANALYSIS
        assert evidence.confidence == 0.85
        assert evidence.source == "url_analyzer"
        assert evidence.metadata["domain"] == "evil.com"

    def test_evidence_validation(self):
        """Test evidence confidence validation."""
        with pytest.raises(ValueError):
            Evidence(
                type=EvidenceType.URL_ANALYSIS,
                description="Test",
                confidence=1.5,  # Invalid: > 1.0
                source="test",
            )


class TestAction:
    """Test Action model."""

    def test_action_creation(self):
        """Test creating action."""
        action = Action(
            type=ActionType.BLOCK_USER,
            priority=1,
            description="Block suspicious user",
            rationale="Multiple fraud indicators detected",
            automated=True,
        )

        assert action.type == ActionType.BLOCK_USER
        assert action.priority == 1
        assert action.automated is True

    def test_action_priority_validation(self):
        """Test action priority validation."""
        with pytest.raises(ValueError):
            Action(
                type=ActionType.FLAG_FOR_REVIEW,
                priority=0,  # Invalid: must be >= 1
                description="Test",
            )


class TestReasoningStep:
    """Test ReasoningStep model."""

    def test_reasoning_step_creation(self):
        """Test creating reasoning step."""
        step = ReasoningStep(
            step_number=1,
            observation="Email contains urgency language",
            thought="High urgency is common in phishing attacks",
            action_taken="Analyzed language patterns",
            result="Urgency score: 0.8",
            confidence_change=0.2,
        )

        assert step.step_number == 1
        assert step.confidence_change == 0.2
        assert "urgency" in step.observation.lower()

    def test_reasoning_step_validation(self):
        """Test reasoning step confidence validation."""
        with pytest.raises(ValueError):
            ReasoningStep(
                step_number=1,
                observation="Test",
                thought="Test",
                action_taken="Test",
                result="Test",
                confidence_change=2.0,  # Invalid: > 1.0
            )


class TestToolCall:
    """Test ToolCall model."""

    def test_tool_call_creation(self):
        """Test creating tool call."""
        tool_call = ToolCall(
            tool_name="analyze_url",
            arguments={"url": "https://evil.com"},
            reasoning="Need to check URL reputation",
        )

        assert tool_call.tool_name == "analyze_url"
        assert tool_call.arguments["url"] == "https://evil.com"


class TestFraudAnalysisOutput:
    """Test FraudAnalysisOutput model."""

    def test_full_analysis_creation(self):
        """Test creating complete fraud analysis."""
        analysis = FraudAnalysisOutput(
            fraud_detected=True,
            confidence=0.85,
            fraud_types=[FraudType.PHISHING, FraudType.SOCIAL_ENGINEERING],
            severity=Severity.HIGH,
            fraud_score=0.85,
            reasoning="Multiple fraud indicators detected",
            evidence=[
                Evidence(
                    type=EvidenceType.URL_ANALYSIS,
                    description="Malicious URL",
                    confidence=0.9,
                    source="url_analyzer",
                )
            ],
            recommended_actions=[
                Action(
                    type=ActionType.BLOCK_USER,
                    priority=1,
                    description="Block immediately",
                )
            ],
            reasoning_steps=[],
            tool_calls=[],
            metadata={"agent": "phishing_agent"},
        )

        assert analysis.fraud_detected is True
        assert analysis.confidence == 0.85
        assert FraudType.PHISHING in analysis.fraud_types
        assert analysis.severity == Severity.HIGH
        assert len(analysis.evidence) == 1
        assert len(analysis.recommended_actions) == 1

    def test_fraud_score_validation(self):
        """Test fraud score validation."""
        with pytest.raises(ValueError):
            FraudAnalysisOutput(
                fraud_detected=True,
                confidence=0.8,
                fraud_types=[FraudType.PHISHING],
                severity=Severity.HIGH,
                fraud_score=1.5,  # Invalid: > 1.0
                reasoning="Test",
            )

    def test_json_serialization(self):
        """Test that analysis can be serialized to JSON."""
        analysis = FraudAnalysisOutput(
            fraud_detected=True,
            confidence=0.75,
            fraud_types=[FraudType.PHISHING],
            severity=Severity.MEDIUM,
            fraud_score=0.75,
            reasoning="Test reasoning",
        )

        json_str = analysis.model_dump_json()
        assert "fraud_detected" in json_str
        assert "phishing" in json_str.lower()


class TestAgentMessage:
    """Test AgentMessage model."""

    def test_agent_message_creation(self):
        """Test creating agent message."""
        msg = AgentMessage(
            sender="phishing_agent",
            recipient="coordinator",
            message_type="analysis_result",
            content={"fraud_score": 0.8},
            timestamp=datetime.now(),
        )

        assert msg.sender == "phishing_agent"
        assert msg.recipient == "coordinator"
        assert msg.content["fraud_score"] == 0.8
