"""
Phishing detection specialist agent.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from typing import Any

from fraudlens.agents.base import BaseAgent
from fraudlens.llm.schemas import (
    Action,
    ActionType,
    Evidence,
    FraudAnalysisOutput,
    FraudType,
    Severity,
)
from fraudlens.llm.tools import ToolCall


class PhishingAgent(BaseAgent):
    """
    Specialist agent for phishing detection.
    
    Analyzes text, emails, and URLs for phishing attempts using
    multiple detection techniques and tool calling.
    """
    
    def __init__(self, agent_id: str = "phishing_agent"):
        """
        Initialize phishing agent.
        
        Args:
            agent_id: Unique agent identifier
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="phishing_specialist",
            fraud_types=["phishing", "brand_impersonation", "social_engineering"]
        )
        
        self.confidence_threshold = 0.6
    
    async def _setup(self) -> None:
        """Setup phishing agent."""
        # Load any models or resources needed
        pass
    
    async def analyze(
        self,
        input_data: Any,
        context: dict[str, Any] | None = None,
    ) -> FraudAnalysisOutput:
        """
        Analyze input for phishing.
        
        Args:
            input_data: Text or email content to analyze
            context: Optional analysis context
        
        Returns:
            Fraud analysis output
        """
        if not isinstance(input_data, str):
            input_data = str(input_data)
        
        evidence_list = []
        fraud_types_detected = []
        fraud_score = 0.0
        
        # Step 1: Extract URLs
        url_tool = self.tool_registry.get("extract_urls")
        if url_tool:
            url_result = await url_tool.execute(text=input_data)
            urls = url_result.get("urls", [])
            
            if urls:
                evidence_list.append(Evidence(
                    type="urls_found",
                    description=f"Found {len(urls)} URLs in content",
                    confidence=0.5,
                    source=self.agent_id,
                    metadata={"urls": urls}
                ))
        else:
            urls = []
        
        # Step 2: Analyze URLs for phishing
        for url in urls[:3]:  # Analyze up to 3 URLs
            analyze_tool = self.tool_registry.get("analyze_url")
            if analyze_tool:
                url_analysis = await analyze_tool.execute(url=url)
                
                if url_analysis.get("is_suspicious"):
                    evidence_list.append(Evidence(
                        type="suspicious_url",
                        description=f"URL appears suspicious: {url}",
                        confidence=0.8,
                        source=self.agent_id,
                        metadata=url_analysis
                    ))
                    fraud_score += 0.3
                    if "phishing" not in fraud_types_detected:
                        fraud_types_detected.append("phishing")
        
        # Step 3: Analyze urgency
        urgency_tool = self.tool_registry.get("analyze_urgency")
        if urgency_tool:
            urgency_result = await urgency_tool.execute(text=input_data)
            urgency_score = urgency_result.get("urgency_score", 0.0)
            
            if urgency_score > 0.5:
                evidence_list.append(Evidence(
                    type="urgency_tactics",
                    description=f"High urgency detected (score: {urgency_score:.2f})",
                    confidence=urgency_score,
                    source=self.agent_id,
                    metadata=urgency_result
                ))
                fraud_score += urgency_score * 0.3
                if "social_engineering" not in fraud_types_detected:
                    fraud_types_detected.append("social_engineering")
        
        # Step 4: Check for brand impersonation keywords
        brand_keywords = ["paypal", "amazon", "apple", "microsoft", "bank"]
        text_lower = input_data.lower()
        
        for brand in brand_keywords:
            if brand in text_lower and f"{brand}.com" not in text_lower:
                evidence_list.append(Evidence(
                    type="brand_mention",
                    description=f"Mentions {brand} but may not be official",
                    confidence=0.6,
                    source=self.agent_id,
                    metadata={"brand": brand}
                ))
                fraud_score += 0.2
                if "brand_impersonation" not in fraud_types_detected:
                    fraud_types_detected.append("brand_impersonation")
        
        # Normalize fraud score
        fraud_score = min(fraud_score, 1.0)
        
        # Determine severity
        if fraud_score > 0.8:
            severity = Severity.CRITICAL
        elif fraud_score > 0.6:
            severity = Severity.HIGH
        elif fraud_score > 0.3:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW
        
        # Generate reasoning
        reasoning = self._generate_reasoning(fraud_score, evidence_list, fraud_types_detected)
        
        # Recommend actions
        actions = self._recommend_actions(fraud_score, severity, fraud_types_detected)
        
        # Convert string fraud types to FraudType enum
        fraud_type_enums = []
        for ft in fraud_types_detected:
            try:
                fraud_type_enums.append(FraudType(ft))
            except ValueError:
                fraud_type_enums.append(FraudType.UNKNOWN)
        
        return FraudAnalysisOutput(
            fraud_detected=fraud_score > self.confidence_threshold,
            confidence=min(fraud_score + 0.1, 1.0),  # Add base confidence
            fraud_types=fraud_type_enums,
            severity=severity,
            fraud_score=fraud_score,
            reasoning=reasoning,
            evidence=evidence_list,
            recommended_actions=actions,
            requires_human_review=fraud_score > 0.7 and fraud_score < 0.9,
            tools_used=["extract_urls", "analyze_url", "analyze_urgency"],
            metadata={
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "analyzed_length": len(input_data)
            }
        )
    
    def _generate_reasoning(
        self,
        fraud_score: float,
        evidence: list[Evidence],
        fraud_types: list[str]
    ) -> str:
        """Generate human-readable reasoning."""
        if fraud_score < 0.3:
            return "Content appears legitimate with no significant fraud indicators."
        
        reasoning_parts = []
        
        if fraud_types:
            reasoning_parts.append(
                f"Detected potential {', '.join(fraud_types)}."
            )
        
        if evidence:
            reasoning_parts.append(
                f"Found {len(evidence)} pieces of supporting evidence including "
                f"{', '.join(e.type for e in evidence[:3])}."
            )
        
        return " ".join(reasoning_parts)
    
    def _recommend_actions(
        self,
        fraud_score: float,
        severity: Severity,
        fraud_types: list[str]
    ) -> list[Action]:
        """Recommend appropriate actions."""
        actions = []
        
        if fraud_score > 0.8:
            actions.append(Action(
                action_type=ActionType.BLOCK,
                priority=5,
                description="Block and quarantine this content immediately",
                rationale="High confidence phishing attempt detected",
                automated=True
            ))
        elif fraud_score > 0.6:
            actions.append(Action(
                action_type=ActionType.FLAG_FOR_REVIEW,
                priority=4,
                description="Flag for security team review",
                rationale="Moderate confidence phishing indicators present",
                automated=True
            ))
            actions.append(Action(
                action_type=ActionType.NOTIFY_USER,
                priority=3,
                description="Notify user of potential phishing",
                rationale="User should be aware of suspicious content",
                automated=True
            ))
        elif fraud_score > 0.3:
            actions.append(Action(
                action_type=ActionType.LOG_INCIDENT,
                priority=2,
                description="Log incident for monitoring",
                rationale="Low confidence indicators warrant logging",
                automated=True
            ))
        
        return actions
