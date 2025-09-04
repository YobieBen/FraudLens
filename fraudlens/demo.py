"""
FraudLens Demo - Interactive fraud detection demonstration.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from fraudlens.core.base.detector import (
    DetectionResult,
    FraudDetector,
    FraudType,
    Modality,
)
from fraudlens.core.base.processor import ModalityProcessor, ProcessedData
from fraudlens.core.base.scorer import RiskAssessment, RiskLevel, RiskScorer
from fraudlens.core.resource_manager.manager import ResourceManager
from fraudlens.pipelines.async_pipeline import AsyncPipeline
from fraudlens.utils.config import ConfigManager


console = Console()


class DemoTextDetector(FraudDetector):
    """Demo text fraud detector."""

    def __init__(self):
        super().__init__(
            detector_id="demo_text_detector",
            modality=Modality.TEXT,
            config={"threshold": 0.5},
        )

    async def initialize(self) -> None:
        """Initialize the detector."""
        self._initialized = True
        console.print("[green]✓[/green] Text detector initialized")

    async def detect(self, input_data: str, **kwargs) -> DetectionResult:
        """Detect fraud in text."""
        # Simulate processing time
        await asyncio.sleep(0.5)

        # Simple rule-based detection for demo
        suspicious_words = [
            "urgent",
            "verify",
            "suspend",
            "click here",
            "act now",
            "limited time",
            "winner",
            "congratulations",
            "tax refund",
        ]

        text_lower = input_data.lower()
        fraud_score = 0.0
        evidence = []

        for word in suspicious_words:
            if word in text_lower:
                fraud_score += 0.15
                evidence.append(f"Found suspicious term: '{word}'")

        fraud_score = min(fraud_score, 1.0)

        fraud_types = []
        if fraud_score > 0.3:
            fraud_types.append(FraudType.PHISHING)
        if "account" in text_lower or "password" in text_lower:
            fraud_types.append(FraudType.ACCOUNT_TAKEOVER)

        return DetectionResult(
            fraud_score=fraud_score,
            fraud_types=fraud_types or [FraudType.UNKNOWN],
            confidence=0.85 if fraud_score > 0 else 0.95,
            explanation=f"Text analysis detected {len(evidence)} suspicious indicators",
            evidence={"suspicious_terms": evidence},
            timestamp=datetime.now(),
            detector_id=self.detector_id,
            modality=self.modality,
            processing_time_ms=500,
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._initialized = False

    def get_memory_usage(self) -> int:
        """Get memory usage."""
        return 100 * 1024 * 1024  # 100MB

    def validate_input(self, input_data: str) -> bool:
        """Validate input."""
        return isinstance(input_data, str) and len(input_data) > 0


class DemoImageDetector(FraudDetector):
    """Demo image fraud detector."""

    def __init__(self):
        super().__init__(
            detector_id="demo_image_detector",
            modality=Modality.IMAGE,
            config={"threshold": 0.6},
        )

    async def initialize(self) -> None:
        """Initialize the detector."""
        self._initialized = True
        console.print("[green]✓[/green] Image detector initialized")

    async def detect(self, input_data: Any, **kwargs) -> DetectionResult:
        """Detect fraud in images."""
        await asyncio.sleep(0.8)

        # Simulate image analysis
        fraud_score = np.random.uniform(0.1, 0.7)

        return DetectionResult(
            fraud_score=fraud_score,
            fraud_types=[FraudType.DOCUMENT_FORGERY if fraud_score > 0.5 else FraudType.UNKNOWN],
            confidence=0.75,
            explanation="Image analysis for document authenticity",
            evidence={"analysis": "Texture and metadata analysis performed"},
            timestamp=datetime.now(),
            detector_id=self.detector_id,
            modality=self.modality,
            processing_time_ms=800,
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._initialized = False

    def get_memory_usage(self) -> int:
        """Get memory usage."""
        return 200 * 1024 * 1024  # 200MB

    def validate_input(self, input_data: Any) -> bool:
        """Validate input."""
        return True


class DemoScorer(RiskScorer):
    """Demo risk scorer."""

    def __init__(self):
        super().__init__(scorer_id="demo_scorer")

    async def score(self, detection_results: List[DetectionResult], **kwargs) -> RiskAssessment:
        """Calculate risk score."""
        if not detection_results:
            return RiskAssessment(
                overall_score=0.0,
                risk_level=RiskLevel.VERY_LOW,
                confidence=1.0,
                contributing_factors=[],
                recommendations=[],
                timestamp=datetime.now(),
                assessment_id="demo_assessment",
            )

        # Aggregate scores
        scores = [r.fraud_score for r in detection_results]
        overall_score = self.aggregate_scores(scores)

        # Determine risk level
        risk_level = RiskLevel.from_score(overall_score)

        # Generate factors
        factors = []
        for result in detection_results:
            if result.fraud_score > 0.3:
                factors.append(
                    {
                        "detector": result.detector_id,
                        "score": result.fraud_score,
                        "weight": 1.0,
                        "fraud_types": [ft.value for ft in result.fraud_types],
                    }
                )

        # Generate recommendations
        recommendations = self.generate_recommendations_internal(overall_score, risk_level)

        return RiskAssessment(
            overall_score=overall_score,
            risk_level=risk_level,
            confidence=self.calculate_confidence(detection_results),
            contributing_factors=factors,
            recommendations=recommendations,
            timestamp=datetime.now(),
            assessment_id="demo_assessment",
        )

    def aggregate_scores(self, scores: List[float], weights: List[float] = None) -> float:
        """Aggregate scores using weighted average."""
        if not scores:
            return 0.0
        if weights:
            return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return sum(scores) / len(scores)

    def calculate_confidence(self, detection_results: List[DetectionResult]) -> float:
        """Calculate confidence."""
        if not detection_results:
            return 0.0
        confidences = [r.confidence for r in detection_results]
        return sum(confidences) / len(confidences)

    def generate_recommendations(self, risk_assessment: RiskAssessment) -> List[str]:
        """Generate recommendations."""
        return self.generate_recommendations_internal(
            risk_assessment.overall_score, risk_assessment.risk_level
        )

    def generate_recommendations_internal(self, score: float, risk_level: RiskLevel) -> List[str]:
        """Generate recommendations based on risk."""
        recommendations = []

        if risk_level == RiskLevel.VERY_HIGH:
            recommendations.append("🚨 Immediate action required: Block transaction")
            recommendations.append("📞 Contact customer for verification")
            recommendations.append("📋 File suspicious activity report")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("⚠️ Flag for manual review")
            recommendations.append("🔍 Request additional verification")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("👁️ Monitor account activity")
            recommendations.append("📊 Increase monitoring frequency")
        else:
            recommendations.append("✅ Continue normal processing")

        return recommendations


async def run_demo():
    """Run the FraudLens demo."""
    console.print(
        Panel.fit(
            "[bold cyan]FraudLens Demo[/bold cyan]\n" "Multi-modal Fraud Detection System",
            border_style="cyan",
        )
    )

    # Initialize components
    console.print("\n[bold]Initializing FraudLens components...[/bold]\n")

    # Resource manager
    resource_manager = ResourceManager(max_memory_gb=100, enable_monitoring=True)
    await resource_manager.start_monitoring()
    console.print("[green]✓[/green] Resource manager started")

    # Pipeline
    pipeline = AsyncPipeline(max_workers=5)
    console.print("[green]✓[/green] Async pipeline created")

    # Register detectors
    text_detector = DemoTextDetector()
    image_detector = DemoImageDetector()
    pipeline.register_detector("text_detector", text_detector)
    pipeline.register_detector("image_detector", image_detector)
    console.print("[green]✓[/green] Fraud detectors registered")

    # Register scorer
    scorer = DemoScorer()
    pipeline.register_scorer("risk_scorer", scorer)
    console.print("[green]✓[/green] Risk scorer registered")

    # Start pipeline
    await pipeline.start()
    console.print("[green]✓[/green] Pipeline started\n")

    # Demo samples
    samples = [
        {
            "id": "DEMO-001",
            "type": "text",
            "content": "URGENT: Your account will be suspended. Click here to verify immediately!",
            "description": "Phishing attempt",
        },
        {
            "id": "DEMO-002",
            "type": "text",
            "content": "Thank you for your recent purchase. Your order has been shipped.",
            "description": "Legitimate message",
        },
        {
            "id": "DEMO-003",
            "type": "text",
            "content": "Congratulations! You're our winner! Act now to claim your tax refund.",
            "description": "Scam message",
        },
        {
            "id": "DEMO-004",
            "type": "image",
            "content": "fake_document.jpg",
            "description": "Potentially forged document",
        },
    ]

    # Process samples
    console.print("[bold]Processing fraud detection samples...[/bold]\n")

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        for sample in samples:
            task = progress.add_task(
                f"Analyzing {sample['id']}: {sample['description']}...", total=None
            )

            # Process based on type
            if sample["type"] == "text":
                result = await pipeline.process(sample["content"], modality=Modality.TEXT)
            else:
                result = await pipeline.process(sample["content"], modality=Modality.IMAGE)

            results.append((sample, result))
            progress.remove_task(task)

    # Display results
    console.print("\n[bold]Detection Results:[/bold]\n")

    for sample, result in results:
        # Create result table
        table = Table(title=f"Sample {sample['id']}", border_style="blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        # Determine color based on risk
        if result.risk_assessment:
            risk_level = result.risk_assessment.risk_level
            score = result.risk_assessment.overall_score

            if risk_level == RiskLevel.VERY_HIGH:
                risk_color = "red"
            elif risk_level == RiskLevel.HIGH:
                risk_color = "orange1"
            elif risk_level == RiskLevel.MEDIUM:
                risk_color = "yellow"
            else:
                risk_color = "green"

            table.add_row("Description", sample["description"])
            table.add_row("Content Preview", sample["content"][:50] + "...")
            table.add_row("Risk Score", f"[{risk_color}]{score:.2%}[/{risk_color}]")
            table.add_row("Risk Level", f"[{risk_color}]{risk_level.value.upper()}[/{risk_color}]")
            table.add_row("Confidence", f"{result.risk_assessment.confidence:.2%}")

            # Add fraud types
            if result.detection_results:
                fraud_types = set()
                for dr in result.detection_results:
                    fraud_types.update(dr.fraud_types)
                fraud_types_str = ", ".join([ft.value for ft in fraud_types])
                table.add_row("Fraud Types", fraud_types_str or "None")

            # Add recommendations
            if result.risk_assessment.recommendations:
                recs = "\n".join(result.risk_assessment.recommendations[:2])
                table.add_row("Recommendations", recs)

            table.add_row("Processing Time", f"{result.processing_time_ms:.0f}ms")

        console.print(table)
        console.print()

    # Show system statistics
    console.print("[bold]System Statistics:[/bold]\n")

    # Resource stats
    resource_stats = resource_manager.get_statistics()
    stats_table = Table(border_style="green")
    stats_table.add_column("Resource", style="cyan")
    stats_table.add_column("Usage", style="white")

    current = resource_stats["current"]
    stats_table.add_row("Memory Used", f"{current['memory_used_gb']:.2f} GB")
    stats_table.add_row("Memory Available", f"{current['memory_available_gb']:.2f} GB")
    stats_table.add_row("CPU Usage", f"{current['cpu_percent']:.1f}%")
    stats_table.add_row("Active Models", str(current["active_models"]))

    console.print(stats_table)
    console.print()

    # Pipeline stats
    pipeline_stats = pipeline.get_statistics()
    pipeline_table = Table(border_style="yellow")
    pipeline_table.add_column("Pipeline Metric", style="cyan")
    pipeline_table.add_column("Value", style="white")

    pipeline_table.add_row("Total Processed", str(pipeline_stats["total_processed"]))
    pipeline_table.add_row("Cache Hits", str(pipeline_stats["cache_hits"]))
    pipeline_table.add_row("Average Time", f"{pipeline_stats['average_time_ms']:.0f}ms")
    pipeline_table.add_row("Active Workers", str(pipeline_stats["workers"]))

    console.print(pipeline_table)

    # Cleanup
    console.print("\n[bold]Cleaning up...[/bold]")
    await pipeline.stop()
    await resource_manager.stop_monitoring()
    console.print("[green]✓[/green] Demo completed successfully!\n")

    # Final message
    console.print(
        Panel(
            "[bold green]Demo Complete![/bold green]\n\n"
            "FraudLens successfully demonstrated:\n"
            "• Multi-modal fraud detection (text & image)\n"
            "• Async parallel processing\n"
            "• Risk scoring and recommendations\n"
            "• Resource monitoring\n\n"
            "[cyan]Ready for production deployment![/cyan]",
            border_style="green",
        )
    )


def main():
    """Main entry point."""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
