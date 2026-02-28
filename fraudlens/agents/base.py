"""
Base classes for fraud detection agents.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Protocol

from pydantic import BaseModel, Field

from fraudlens.events import EventBus, get_event_bus
from fraudlens.llm.schemas import AgentMessage, FraudAnalysisOutput
from fraudlens.llm.tools import ToolRegistry, get_tool_registry


class AgentCapabilities(BaseModel):
    """Agent capabilities and metadata."""
    
    agent_id: str
    agent_type: str
    fraud_types: list[str] = Field(default_factory=list)
    supports_streaming: bool = False
    max_concurrent_analyses: int = 1
    confidence_threshold: float = 0.5


class Agent(Protocol):
    """
    Protocol for fraud detection agents.
    
    All agents must implement this interface to participate in the
    multi-agent fraud detection system.
    """
    
    agent_id: str
    """Unique agent identifier."""
    
    async def initialize(self) -> None:
        """Initialize the agent and load resources."""
        ...
    
    async def analyze(
        self,
        input_data: Any,
        context: dict[str, Any] | None = None,
    ) -> FraudAnalysisOutput:
        """
        Analyze input for fraud.
        
        Args:
            input_data: Data to analyze
            context: Optional analysis context
        
        Returns:
            Fraud analysis output
        """
        ...
    
    async def analyze_stream(
        self,
        input_data: Any,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[FraudAnalysisOutput]:
        """
        Stream analysis results as they become available.
        
        Args:
            input_data: Data to analyze
            context: Optional context
        
        Yields:
            Incremental analysis results
        """
        ...
    
    def get_capabilities(self) -> AgentCapabilities:
        """Get agent capabilities."""
        ...
    
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        ...


class BaseAgent(ABC):
    """
    Base implementation for fraud detection agents.
    
    Provides common functionality for all agents including
    event emission, tool access, and message handling.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        fraud_types: list[str] | None = None,
        event_bus: EventBus | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (e.g., "phishing_specialist")
            fraud_types: List of fraud types this agent handles
            event_bus: Event bus for communication
            tool_registry: Tool registry for accessing tools
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.fraud_types = fraud_types or []
        self.event_bus = event_bus or get_event_bus()
        self.tool_registry = tool_registry or get_tool_registry()
        
        self._initialized = False
        self._messages: list[AgentMessage] = []
    
    async def initialize(self) -> None:
        """Initialize the agent."""
        if self._initialized:
            return
        
        await self._setup()
        self._initialized = True
    
    @abstractmethod
    async def _setup(self) -> None:
        """Agent-specific setup logic."""
        pass
    
    @abstractmethod
    async def analyze(
        self,
        input_data: Any,
        context: dict[str, Any] | None = None,
    ) -> FraudAnalysisOutput:
        """
        Analyze input for fraud.
        
        Must be implemented by subclasses.
        """
        pass
    
    async def analyze_stream(
        self,
        input_data: Any,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[FraudAnalysisOutput]:
        """
        Default streaming implementation.
        
        Yields single result from analyze(). Subclasses can override
        for true streaming.
        """
        result = await self.analyze(input_data, context)
        yield result
    
    def get_capabilities(self) -> AgentCapabilities:
        """Get agent capabilities."""
        return AgentCapabilities(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            fraud_types=self.fraud_types,
            supports_streaming=False,  # Override in subclasses
            max_concurrent_analyses=1,
        )
    
    async def send_message(self, message: AgentMessage) -> None:
        """
        Send message to another agent.
        
        Args:
            message: Message to send
        """
        self._messages.append(message)
        # In a full implementation, this would use the event bus
        # to route messages to the appropriate agent
    
    async def receive_message(self, message: AgentMessage) -> None:
        """
        Receive message from another agent.
        
        Args:
            message: Received message
        """
        await self._handle_message(message)
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """
        Handle received message.
        
        Can be overridden by subclasses for custom message handling.
        """
        pass
    
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        self._initialized = False
        self._messages.clear()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(id='{self.agent_id}', type='{self.agent_type}')"
