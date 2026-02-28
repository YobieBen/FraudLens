"""
Agent coordinator for orchestrating multiple specialist agents.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import asyncio
from typing import Any, AsyncIterator

from fraudlens.agents.base import BaseAgent
from fraudlens.events import AnalysisCompleteEvent, DetectionStartedEvent
from fraudlens.llm.schemas import (
    Action,
    Evidence,
    FraudAnalysisOutput,
    FraudType,
    Severity,
)
from fraudlens.streaming import AnalysisStream, ChunkType, StreamChunk


class AgentCoordinator(BaseAgent):
    """
    Coordinator agent that orchestrates multiple specialist agents.
    
    Uses a consensus mechanism to combine results from multiple agents
    and produce a final fraud detection decision.
    """
    
    def __init__(
        self,
        agents: list[BaseAgent] | None = None,
        agent_id: str = "coordinator"
    ):
        """
        Initialize coordinator.
        
        Args:
            agents: List of specialist agents to coordinate
            agent_id: Coordinator identifier
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="coordinator",
            fraud_types=[]  # Coordinator handles all types
        )
        
        self.agents = agents or []
        self.consensus_threshold = 0.5  # Minimum agreement threshold
        self.enable_parallel_execution = True
    
    async def _setup(self) -> None:
        """Setup coordinator and all agents."""
        # Initialize all specialist agents
        init_tasks = [agent.initialize() for agent in self.agents]
        await asyncio.gather(*init_tasks)
    
    def add_agent(self, agent: BaseAgent) -> None:
        """
        Add specialist agent to coordinator.
        
        Args:
            agent: Agent to add
        """
        self.agents.append(agent)
    
    async def analyze(
        self,
        input_data: Any,
        context: dict[str, Any] | None = None,
    ) -> FraudAnalysisOutput:
        """
        Analyze input using all specialist agents and build consensus.
        
        Args:
            input_data: Data to analyze
            context: Optional context
        
        Returns:
            Consensus fraud analysis output
        """
        if not self.agents:
            raise ValueError("No agents registered with coordinator")
        
        # Emit detection started event
        await self.event_bus.emit(DetectionStartedEvent(
            source=self.agent_id,
            input_id=context.get("input_id", "unknown") if context else "unknown",
            modality="multi_agent"
        ))
        
        # Execute all agents
        if self.enable_parallel_execution:
            # Parallel execution
            tasks = [
                agent.analyze(input_data, context)
                for agent in self.agents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential execution
            results = []
            for agent in self.agents:
                try:
                    result = await agent.analyze(input_data, context)
                    results.append(result)
                except Exception as e:
                    results.append(e)
        
        # Filter out exceptions
        valid_results = [
            r for r in results
            if isinstance(r, FraudAnalysisOutput)
        ]
        
        if not valid_results:
            raise RuntimeError("All agents failed to produce results")
        
        # Build consensus
        consensus = self._build_consensus(valid_results)
        
        # Emit analysis complete event
        await self.event_bus.emit(AnalysisCompleteEvent(
            source=self.agent_id,
            input_id=context.get("input_id", "unknown") if context else "unknown",
            fraud_score=consensus.fraud_score,
            fraud_types=[ft.value for ft in consensus.fraud_types],
            processing_time_ms=0.0  # Would track actual time
        ))
        
        return consensus
    
    async def analyze_stream(
        self,
        input_data: Any,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream analysis with real-time agent results.
        
        Args:
            input_data: Data to analyze
            context: Optional context
        
        Yields:
            Stream chunks with agent progress
        """
        # Start
        yield StreamChunk(
            type=ChunkType.STARTED,
            data={"agents": len(self.agents)},
            progress=0.0
        )
        
        # Execute agents and stream their results
        results = []
        for i, agent in enumerate(self.agents):
            try:
                result = await agent.analyze(input_data, context)
                results.append(result)
                
                # Emit agent result chunk
                yield AnalysisStream.create_agent_result_chunk(
                    agent_id=agent.agent_id,
                    result=result,
                    progress=(i + 1) / len(self.agents)
                )
            
            except Exception as e:
                yield StreamChunk(
                    type=ChunkType.ERROR,
                    data={
                        "agent_id": agent.agent_id,
                        "error": str(e)
                    },
                    progress=(i + 1) / len(self.agents)
                )
        
        # Build consensus
        if results:
            consensus = self._build_consensus(results)
            yield AnalysisStream.create_complete_chunk(consensus)
        else:
            yield StreamChunk(
                type=ChunkType.ERROR,
                data={"error": "No agents produced valid results"},
                progress=1.0
            )
    
    def _build_consensus(
        self,
        results: list[FraudAnalysisOutput]
    ) -> FraudAnalysisOutput:
        """
        Build consensus from multiple agent results.
        
        Uses weighted voting based on confidence scores.
        
        Args:
            results: Results from all agents
        
        Returns:
            Consensus result
        """
        if not results:
            raise ValueError("No results to build consensus from")
        
        # Calculate weighted fraud score
        total_weight = sum(r.confidence for r in results)
        weighted_fraud_score = sum(
            r.fraud_score * r.confidence for r in results
        ) / total_weight if total_weight > 0 else 0.0
        
        # Aggregate fraud types (those mentioned by multiple agents)
        fraud_type_counts: dict[FraudType, float] = {}
        for result in results:
            for fraud_type in result.fraud_types:
                fraud_type_counts[fraud_type] = (
                    fraud_type_counts.get(fraud_type, 0.0) + result.confidence
                )
        
        # Select fraud types with sufficient consensus
        consensus_fraud_types = [
            ft for ft, weight in fraud_type_counts.items()
            if weight / total_weight >= self.consensus_threshold
        ]
        
        # Aggregate evidence
        all_evidence: list[Evidence] = []
        for result in results:
            all_evidence.extend(result.evidence)
        
        # Sort evidence by confidence
        all_evidence.sort(key=lambda e: e.confidence, reverse=True)
        
        # Aggregate actions (deduplicate and sort by priority)
        all_actions: list[Action] = []
        seen_action_types = set()
        for result in results:
            for action in result.recommended_actions:
                if action.type not in seen_action_types:
                    all_actions.append(action)
                    seen_action_types.add(action.type)
        
        all_actions.sort(key=lambda a: a.priority, reverse=True)
        
        # Determine overall severity
        severity_scores = {
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4
        }
        avg_severity_score = sum(
            severity_scores[r.severity] * r.confidence
            for r in results
        ) / total_weight if total_weight > 0 else 1
        
        if avg_severity_score >= 3.5:
            consensus_severity = Severity.CRITICAL
        elif avg_severity_score >= 2.5:
            consensus_severity = Severity.HIGH
        elif avg_severity_score >= 1.5:
            consensus_severity = Severity.MEDIUM
        else:
            consensus_severity = Severity.LOW
        
        # Build consensus reasoning
        agent_summaries = [
            f"{r.metadata.get('agent_id', 'agent')}: {r.fraud_score:.2f} confidence"
            for r in results
        ]
        
        reasoning = (
            f"Consensus from {len(results)} specialist agents. "
            f"Agent results: {', '.join(agent_summaries)}. "
        )
        
        if consensus_fraud_types:
            reasoning += f"Detected {', '.join(ft.value for ft in consensus_fraud_types)}."
        
        # Calculate overall confidence
        # Higher when agents agree
        fraud_score_variance = sum(
            (r.fraud_score - weighted_fraud_score) ** 2
            for r in results
        ) / len(results)
        
        agreement_factor = max(0.5, 1.0 - fraud_score_variance)
        overall_confidence = min(
            sum(r.confidence for r in results) / len(results) * agreement_factor,
            1.0
        )
        
        # Determine if human review needed
        requires_review = (
            weighted_fraud_score > 0.5 and
            (fraud_score_variance > 0.1 or any(r.requires_human_review for r in results))
        )
        
        return FraudAnalysisOutput(
            fraud_detected=weighted_fraud_score > 0.5,
            confidence=overall_confidence,
            fraud_types=consensus_fraud_types or [FraudType.UNKNOWN],
            severity=consensus_severity,
            fraud_score=weighted_fraud_score,
            reasoning=reasoning,
            evidence=all_evidence[:10],  # Top 10 pieces of evidence
            recommended_actions=all_actions[:5],  # Top 5 actions
            requires_human_review=requires_review,
            tools_used=[],
            metadata={
                "coordinator_id": self.agent_id,
                "num_agents": len(results),
                "consensus_threshold": self.consensus_threshold,
                "agent_results": [
                    {
                        "agent_id": r.metadata.get("agent_id"),
                        "fraud_score": r.fraud_score,
                        "confidence": r.confidence
                    }
                    for r in results
                ]
            }
        )
