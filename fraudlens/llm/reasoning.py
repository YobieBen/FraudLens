"""
Chain-of-thought reasoning for fraud detection.

Implements multi-step reasoning loops where agents can think through
fraud detection problems step by step, using tools as needed.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from typing import Any

from fraudlens.llm.schemas import (
    Evidence,
    FraudAnalysisOutput,
    FraudType,
    ReasoningStep,
    ToolCall,
)
from fraudlens.llm.tools import ToolRegistry, get_tool_registry


class ReasoningLoop:
    """
    Chain-of-thought reasoning loop for fraud detection.
    
    Enables multi-step reasoning where each step can:
    - Make observations
    - Think about the observations
    - Call tools to gather more information
    - Update confidence based on findings
    """
    
    def __init__(
        self,
        max_steps: int = 10,
        confidence_threshold: float = 0.9,
        tool_registry: ToolRegistry | None = None,
    ):
        """
        Initialize reasoning loop.
        
        Args:
            max_steps: Maximum reasoning steps
            confidence_threshold: Stop when confidence exceeds this
            tool_registry: Tool registry for tool access
        """
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self.tool_registry = tool_registry or get_tool_registry()
        
        self.steps: list[ReasoningStep] = []
        self.evidence: list[Evidence] = []
        self.current_confidence = 0.0
    
    async def reason(
        self,
        initial_observation: str,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute reasoning loop.
        
        Args:
            initial_observation: Starting observation
            context: Optional reasoning context
        
        Returns:
            Reasoning results with steps and evidence
        """
        self.steps = []
        self.evidence = []
        self.current_confidence = 0.0
        
        # Step 1: Initial observation
        step = ReasoningStep(
            step_number=1,
            observation=initial_observation,
            thought="Beginning analysis of the input content",
            confidence_change=0.0
        )
        self.steps.append(step)
        
        # Continue reasoning until confidence threshold or max steps
        for step_num in range(2, self.max_steps + 1):
            if self.current_confidence >= self.confidence_threshold:
                break
            
            # Generate next reasoning step
            next_step = await self._generate_next_step(step_num, context)
            
            if next_step:
                self.steps.append(next_step)
                self.current_confidence += next_step.confidence_change
                self.current_confidence = max(0.0, min(1.0, self.current_confidence))
            else:
                # No more reasoning needed
                break
        
        return {
            "steps": self.steps,
            "evidence": self.evidence,
            "final_confidence": self.current_confidence,
            "steps_taken": len(self.steps)
        }
    
    async def _generate_next_step(
        self,
        step_number: int,
        context: dict[str, Any] | None = None
    ) -> ReasoningStep | None:
        """
        Generate the next reasoning step.
        
        This is a simplified implementation. In a full system, this would
        use an LLM to determine what to analyze next.
        
        Args:
            step_number: Current step number
            context: Reasoning context
        
        Returns:
            Next reasoning step or None if done
        """
        previous_step = self.steps[-1] if self.steps else None
        
        if not previous_step:
            return None
        
        # Simplified reasoning logic
        # In reality, this would use an LLM to decide what to analyze
        
        if step_number == 2:
            # Analyze urgency
            return ReasoningStep(
                step_number=step_number,
                observation="Checking for urgency and pressure tactics",
                thought="Phishing often uses urgency to bypass critical thinking",
                action_taken="analyze_urgency_indicators",
                confidence_change=0.2
            )
        
        elif step_number == 3:
            # Check URLs
            return ReasoningStep(
                step_number=step_number,
                observation="Examining any URLs present in content",
                thought="Malicious URLs are strong fraud indicators",
                action_taken="extract_and_analyze_urls",
                confidence_change=0.3
            )
        
        elif step_number == 4:
            # Check brand mentions
            return ReasoningStep(
                step_number=step_number,
                observation="Looking for brand impersonation attempts",
                thought="Fake brands are common in phishing",
                action_taken="check_brand_authenticity",
                confidence_change=0.2
            )
        
        # Done reasoning
        return None
    
    async def reason_with_tools(
        self,
        initial_input: str,
        available_tools: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Reason through a problem using tool calls.
        
        Args:
            initial_input: Input to reason about
            available_tools: List of tool names to use
        
        Returns:
            Reasoning results with tool call history
        """
        self.steps = []
        self.evidence = []
        tool_calls_made: list[dict] = []
        
        # Step 1: Initial assessment
        step = ReasoningStep(
            step_number=1,
            observation=f"Input received: {initial_input[:100]}...",
            thought="Need to analyze this input systematically",
            confidence_change=0.0
        )
        self.steps.append(step)
        
        # Step 2: Extract URLs
        url_tool = self.tool_registry.get("extract_urls")
        if url_tool and (not available_tools or "extract_urls" in available_tools):
            url_result = await url_tool.execute(text=initial_input)
            urls = url_result.get("urls", [])
            
            step = ReasoningStep(
                step_number=2,
                observation=f"Found {len(urls)} URLs in input",
                thought="URLs are key fraud indicators, need to analyze them",
                action_taken="extract_urls tool call",
                result=f"Extracted {len(urls)} URLs",
                confidence_change=0.1 if urls else 0.0
            )
            self.steps.append(step)
            tool_calls_made.append({"tool": "extract_urls", "result": url_result})
            
            # Step 3: Analyze each URL
            for i, url in enumerate(urls[:3], start=3):  # Limit to 3 URLs
                analyze_tool = self.tool_registry.get("analyze_url")
                if analyze_tool:
                    url_analysis = await analyze_tool.execute(url=url)
                    is_suspicious = url_analysis.get("is_suspicious", False)
                    
                    step = ReasoningStep(
                        step_number=i,
                        observation=f"Analyzed URL: {url}",
                        thought="Checking if URL matches known fraud patterns",
                        action_taken="analyze_url tool call",
                        result=f"Suspicious: {is_suspicious}",
                        confidence_change=0.3 if is_suspicious else 0.0
                    )
                    self.steps.append(step)
                    tool_calls_made.append({"tool": "analyze_url", "result": url_analysis})
                    
                    if is_suspicious:
                        self.evidence.append(Evidence(
                            type="suspicious_url",
                            description=f"URL {url} flagged as suspicious",
                            confidence=0.8,
                            source="reasoning_loop",
                            metadata=url_analysis
                        ))
        
        # Step N: Analyze urgency
        urgency_tool = self.tool_registry.get("analyze_urgency")
        if urgency_tool and (not available_tools or "analyze_urgency" in available_tools):
            urgency_result = await urgency_tool.execute(text=initial_input)
            urgency_score = urgency_result.get("urgency_score", 0.0)
            
            step = ReasoningStep(
                step_number=len(self.steps) + 1,
                observation=f"Urgency score: {urgency_score:.2f}",
                thought="High urgency is a social engineering tactic",
                action_taken="analyze_urgency tool call",
                result=f"Urgency detected: {'Yes' if urgency_score > 0.5 else 'No'}",
                confidence_change=urgency_score * 0.2
            )
            self.steps.append(step)
            tool_calls_made.append({"tool": "analyze_urgency", "result": urgency_result})
            
            if urgency_score > 0.5:
                self.evidence.append(Evidence(
                    type="urgency_tactics",
                    description=f"High urgency detected (score: {urgency_score:.2f})",
                    confidence=urgency_score,
                    source="reasoning_loop",
                    metadata=urgency_result
                ))
        
        # Calculate final confidence
        total_confidence_change = sum(s.confidence_change for s in self.steps)
        final_confidence = min(total_confidence_change, 1.0)
        
        return {
            "steps": self.steps,
            "evidence": self.evidence,
            "tool_calls": tool_calls_made,
            "final_confidence": final_confidence,
            "steps_taken": len(self.steps)
        }
    
    def to_reasoning_steps(self) -> list[ReasoningStep]:
        """Get all reasoning steps."""
        return self.steps.copy()
    
    def get_evidence(self) -> list[Evidence]:
        """Get collected evidence."""
        return self.evidence.copy()


def create_reasoning_chain(
    observations: list[str],
    thoughts: list[str],
) -> list[ReasoningStep]:
    """
    Create a reasoning chain from observations and thoughts.
    
    Utility function for building reasoning steps manually.
    
    Args:
        observations: List of observations
        thoughts: List of corresponding thoughts
    
    Returns:
        List of reasoning steps
    """
    steps = []
    
    for i, (obs, thought) in enumerate(zip(observations, thoughts), start=1):
        step = ReasoningStep(
            step_number=i,
            observation=obs,
            thought=thought,
            confidence_change=0.0
        )
        steps.append(step)
    
    return steps
