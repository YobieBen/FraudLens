"""
Explanation generator for fraud detection results.

Author: Yobie Benjamin
Date: 2025
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from fraudlens.core.base.detector import FraudType
from fraudlens.fusion.fusion_engine import FusedResult, RiskScore
from fraudlens.fusion.validators import ConsistencyReport


@dataclass
class Explanation:
    """Natural language explanation of risk factors."""

    summary: str
    risk_factors: List[str]
    confidence_explanation: str
    recommendations: List[str]
    visual_highlights: Dict[str, Any]
    audit_trail: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "risk_factors": self.risk_factors,
            "confidence_explanation": self.confidence_explanation,
            "recommendations": self.recommendations,
            "visual_highlights": self.visual_highlights,
            "audit_trail": self.audit_trail,
        }


class ExplanationGenerator:
    """
    Generates natural language explanations for fraud detection results.
    """

    def __init__(self, language: str = "en", detail_level: str = "medium"):
        """
        Initialize explanation generator.

        Args:
            language: Language for explanations
            detail_level: Level of detail (low, medium, high)
        """
        self.language = language
        self.detail_level = detail_level

        # Templates for explanations
        self.templates = self._load_templates()

        # Risk level descriptions
        self.risk_descriptions = {
            "low": "minimal risk indicators detected",
            "medium": "moderate risk factors present",
            "high": "significant fraud indicators identified",
            "critical": "critical fraud patterns detected - immediate action recommended",
        }

        logger.info(f"ExplanationGenerator initialized for {language}")

    def _load_templates(self) -> Dict[str, str]:
        """Load explanation templates."""
        return {
            "summary": "Analysis of {modality_count} modalities revealed {risk_level} risk with {confidence:.1%} confidence. {main_finding}",
            "risk_factor": "• {factor}: {description} (Impact: {impact})",
            "confidence": "Confidence level of {confidence:.1%} is based on {basis}",
            "recommendation": "→ {action}: {reason}",
            "inconsistency": "⚠️ {type}: {description}",
        }

    async def generate_explanation(
        self,
        fused_result: FusedResult,
        risk_score: RiskScore,
        consistency_report: Optional[ConsistencyReport] = None,
    ) -> Explanation:
        """
        Generate comprehensive explanation.

        Args:
            fused_result: Fused detection result
            risk_score: Comprehensive risk score
            consistency_report: Consistency validation report

        Returns:
            Natural language explanation
        """
        # Generate summary
        summary = self._generate_summary(fused_result, risk_score)

        # Extract risk factors
        risk_factors = self._extract_risk_factors(fused_result, risk_score)

        # Explain confidence
        confidence_explanation = self._explain_confidence(fused_result, consistency_report)

        # Generate recommendations
        recommendations = self._generate_recommendations(risk_score, consistency_report)

        # Identify visual highlights
        visual_highlights = self._identify_visual_highlights(fused_result)

        # Create audit trail
        audit_trail = self._create_audit_trail(fused_result, risk_score, consistency_report)

        return Explanation(
            summary=summary,
            risk_factors=risk_factors,
            confidence_explanation=confidence_explanation,
            recommendations=recommendations,
            visual_highlights=visual_highlights,
            audit_trail=audit_trail,
        )

    def _generate_summary(self, fused_result: FusedResult, risk_score: RiskScore) -> str:
        """Generate executive summary."""
        modality_count = len(fused_result.modality_scores)

        # Determine main finding
        if fused_result.fraud_types:
            main_fraud_type = fused_result.fraud_types[0]
            main_finding = f"Primary concern: {main_fraud_type.value.replace('_', ' ')}"
        else:
            main_finding = "No specific fraud patterns identified"

        summary = self.templates["summary"].format(
            modality_count=modality_count,
            risk_level=risk_score.risk_level,
            confidence=fused_result.confidence,
            main_finding=main_finding,
        )

        # Add trend information
        if risk_score.trend != "stable":
            summary += f" Risk trend: {risk_score.trend}."

        return summary

    def _extract_risk_factors(
        self,
        fused_result: FusedResult,
        risk_score: RiskScore,
    ) -> List[str]:
        """Extract and format risk factors."""
        risk_factors = []

        # Process risk score factors
        # Handle both RiskProfile (with 'factors') and RiskScore (with 'risk_factors')
        factors = getattr(risk_score, "factors", None) or getattr(risk_score, "risk_factors", [])
        for factor in factors:
            factor_name = factor["factor"].replace("_", " ").title()

            # Determine impact level
            value = factor["value"]
            if value > 0.8:
                impact = "High"
            elif value > 0.5:
                impact = "Medium"
            else:
                impact = "Low"

            # Generate description
            description = self._describe_risk_factor(factor)

            formatted = self.templates["risk_factor"].format(
                factor=factor_name,
                description=description,
                impact=impact,
            )
            risk_factors.append(formatted)

        # Add fraud type specific factors
        for fraud_type in fused_result.fraud_types:
            risk_factors.append(f"• {fraud_type.value.replace('_', ' ').title()} detected")

        # Add modality-specific high scores
        for modality, score in fused_result.modality_scores.items():
            if score > 0.7:
                risk_factors.append(f"• High risk in {modality} analysis: {score:.2f}")

        return risk_factors[:10] if self.detail_level != "high" else risk_factors

    def _describe_risk_factor(self, factor: Dict[str, Any]) -> str:
        """Generate description for risk factor."""
        factor_type = factor["factor"]
        value = factor["value"]

        descriptions = {
            "base_fraud_score": f"Initial fraud detection score of {value:.2f}",
            "bayesian_aggregation": f"Statistical analysis indicates {value:.2f} probability",
            "anomaly_detection": f"Unusual patterns detected with score {value:.2f}",
            "temporal_pattern": f"Time-based analysis shows {factor.get('trend', 'unusual')} pattern",
            "relationship_risk": f"Network analysis reveals suspicious connections",
        }

        return descriptions.get(factor_type, f"Risk indicator value: {value:.2f}")

    def _explain_confidence(
        self,
        fused_result: FusedResult,
        consistency_report: Optional[ConsistencyReport] = None,
    ) -> str:
        """Explain confidence level."""
        confidence = fused_result.confidence

        # Determine confidence basis
        basis_factors = []

        # Consistency contribution
        if consistency_report:
            if consistency_report.overall_consistency > 0.8:
                basis_factors.append("high cross-modal consistency")
            elif consistency_report.overall_consistency < 0.5:
                basis_factors.append("inconsistencies across modalities")

        # Fusion strategy contribution
        if fused_result.fusion_strategy.value == "hybrid":
            basis_factors.append("comprehensive hybrid analysis")
        elif fused_result.fusion_strategy.value == "hierarchical":
            basis_factors.append("multi-level validation")

        # Modality agreement
        if fused_result.modality_scores:
            scores = list(fused_result.modality_scores.values())
            if np.std(scores) < 0.1:
                basis_factors.append("strong agreement across modalities")
            elif np.std(scores) > 0.3:
                basis_factors.append("varying assessments across modalities")

        basis = ", ".join(basis_factors) if basis_factors else "available evidence"

        explanation = self.templates["confidence"].format(
            confidence=confidence,
            basis=basis,
        )

        return explanation

    def _generate_recommendations(
        self,
        risk_score: RiskScore,
        consistency_report: Optional[ConsistencyReport] = None,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Risk level based recommendations
        if risk_score.risk_level == "critical":
            recommendations.append(
                {
                    "action": "Immediate investigation required",
                    "reason": "Critical fraud indicators detected",
                }
            )
            recommendations.append(
                {
                    "action": "Freeze associated accounts/transactions",
                    "reason": "Prevent potential losses",
                }
            )
        elif risk_score.risk_level == "high":
            recommendations.append(
                {
                    "action": "Manual review recommended",
                    "reason": "Significant risk factors present",
                }
            )
            recommendations.append(
                {
                    "action": "Request additional verification",
                    "reason": "Confirm authenticity",
                }
            )
        elif risk_score.risk_level == "medium":
            recommendations.append(
                {
                    "action": "Monitor for additional signals",
                    "reason": "Moderate risk detected",
                }
            )

        # Inconsistency based recommendations
        if consistency_report and consistency_report.inconsistency_count > 0:
            recommendations.append(
                {
                    "action": "Verify cross-modal inconsistencies",
                    "reason": f"{consistency_report.inconsistency_count} inconsistencies found",
                }
            )

        # Trend based recommendations
        if risk_score.trend == "increasing":
            recommendations.append(
                {
                    "action": "Escalate monitoring frequency",
                    "reason": "Risk trend is increasing",
                }
            )

        # Anomaly based recommendations
        if risk_score.anomaly_score > 0.7:
            recommendations.append(
                {
                    "action": "Investigate unusual patterns",
                    "reason": f"High anomaly score: {risk_score.anomaly_score:.2f}",
                }
            )

        # Format recommendations
        formatted = []
        for rec in recommendations[:5]:  # Limit to top 5
            formatted.append(
                self.templates["recommendation"].format(
                    action=rec["action"],
                    reason=rec["reason"],
                )
            )

        return formatted

    def _identify_visual_highlights(self, fused_result: FusedResult) -> Dict[str, Any]:
        """Identify elements to highlight visually."""
        highlights = {
            "risk_areas": [],
            "suspicious_elements": [],
            "confidence_indicators": [],
        }

        # Identify high-risk areas
        for modality, score in fused_result.modality_scores.items():
            if score > 0.7:
                highlights["risk_areas"].append(
                    {
                        "modality": modality,
                        "score": score,
                        "color": "red" if score > 0.9 else "orange",
                    }
                )

        # Identify suspicious elements from evidence
        if fused_result.evidence:
            for key, value in fused_result.evidence.items():
                if "suspicious" in key.lower() or "anomaly" in key.lower():
                    highlights["suspicious_elements"].append(
                        {
                            "element": key,
                            "details": value,
                        }
                    )

        # Add confidence indicators
        if fused_result.confidence > 0.8:
            highlights["confidence_indicators"].append(
                {
                    "level": "high",
                    "color": "green",
                }
            )
        elif fused_result.confidence < 0.5:
            highlights["confidence_indicators"].append(
                {
                    "level": "low",
                    "color": "yellow",
                }
            )

        return highlights

    def _create_audit_trail(
        self,
        fused_result: FusedResult,
        risk_score: RiskScore,
        consistency_report: Optional[ConsistencyReport] = None,
    ) -> List[Dict[str, Any]]:
        """Create detailed audit trail."""
        audit_trail = []

        # Initial detection
        audit_trail.append(
            {
                "timestamp": fused_result.timestamp.isoformat(),
                "event": "Multi-modal analysis initiated",
                "details": {
                    "modalities": list(fused_result.modality_scores.keys()),
                    "fusion_strategy": fused_result.fusion_strategy.value,
                },
            }
        )

        # Individual modality results
        for modality, score in fused_result.modality_scores.items():
            audit_trail.append(
                {
                    "timestamp": fused_result.timestamp.isoformat(),
                    "event": f"{modality.capitalize()} analysis completed",
                    "details": {
                        "score": score,
                        "processing_time_ms": fused_result.processing_time_ms
                        / len(fused_result.modality_scores),
                    },
                }
            )

        # Fusion result
        audit_trail.append(
            {
                "timestamp": fused_result.timestamp.isoformat(),
                "event": "Fusion completed",
                "details": {
                    "fraud_score": fused_result.fraud_score,
                    "confidence": fused_result.confidence,
                    "fraud_types": [ft.value for ft in fused_result.fraud_types],
                },
            }
        )

        # Risk scoring
        # Handle both RiskProfile and RiskScore objects
        metadata = getattr(risk_score, "metadata", {})
        timestamp = getattr(risk_score, "timestamp", None)
        if timestamp:
            timestamp_str = (
                timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
            )
        else:
            timestamp_str = (
                metadata.get("timestamp", datetime.now()).isoformat()
                if metadata
                else datetime.now().isoformat()
            )

        # Get risk level and overall risk score with fallback
        risk_level = getattr(risk_score, "risk_level", "unknown")
        overall_risk = getattr(risk_score, "overall_risk", None) or getattr(
            risk_score, "risk_score", 0.0
        )
        trend = getattr(risk_score, "trend", "unknown")

        audit_trail.append(
            {
                "timestamp": timestamp_str,
                "event": "Risk scoring completed",
                "details": {
                    "risk_level": risk_level,
                    "overall_risk": overall_risk,
                    "trend": trend,
                },
            }
        )

        # Consistency validation
        if consistency_report:
            audit_trail.append(
                {
                    "timestamp": consistency_report.timestamp.isoformat(),
                    "event": "Consistency validation performed",
                    "details": {
                        "overall_consistency": consistency_report.overall_consistency,
                        "inconsistency_count": consistency_report.inconsistency_count,
                        "high_risk_inconsistencies": len(
                            consistency_report.high_risk_inconsistencies
                        ),
                    },
                }
            )

        return audit_trail


class ReportExporter:
    """
    Exports fraud detection reports in various formats.
    """

    def __init__(self, output_dir: Path = Path("reports")):
        """
        Initialize report exporter.

        Args:
            output_dir: Directory for exported reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        logger.info(f"ReportExporter initialized with output dir: {self.output_dir}")

    async def export_report(
        self,
        case_id: str,
        fused_result: FusedResult,
        risk_score: RiskScore,
        explanation: Explanation,
        consistency_report: Optional[ConsistencyReport] = None,
        format: str = "json",
        include_evidence: bool = True,
    ) -> Path:
        """
        Export comprehensive fraud detection report.

        Args:
            case_id: Unique case identifier
            fused_result: Fused detection result
            risk_score: Risk scoring result
            explanation: Generated explanation
            consistency_report: Consistency validation report
            format: Export format (json, pdf, html)
            include_evidence: Include detailed evidence

        Returns:
            Path to exported report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{case_id}_{timestamp}.{format}"
        output_path = self.output_dir / filename

        if format == "json":
            await self._export_json(
                output_path,
                case_id,
                fused_result,
                risk_score,
                explanation,
                consistency_report,
                include_evidence,
            )
        elif format == "pdf":
            await self._export_pdf(
                output_path,
                case_id,
                fused_result,
                risk_score,
                explanation,
                consistency_report,
                include_evidence,
            )
        elif format == "html":
            await self._export_html(
                output_path,
                case_id,
                fused_result,
                risk_score,
                explanation,
                consistency_report,
                include_evidence,
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Report exported to {output_path}")
        return output_path

    async def _export_json(
        self,
        output_path: Path,
        case_id: str,
        fused_result: FusedResult,
        risk_score: RiskScore,
        explanation: Explanation,
        consistency_report: Optional[ConsistencyReport],
        include_evidence: bool,
    ) -> None:
        """Export as JSON."""
        # Handle both RiskProfile and RiskScore objects
        overall_risk = getattr(risk_score, "overall_risk", None) or getattr(
            risk_score, "risk_score", 0.0
        )
        confidence_interval = getattr(risk_score, "confidence_intervals", None) or getattr(
            risk_score, "confidence_interval", (0.0, 1.0)
        )

        report = {
            "case_id": case_id,
            "timestamp": datetime.now().isoformat(),
            "risk_score": {
                "overall_risk": overall_risk,
                "risk_level": getattr(risk_score, "risk_level", "unknown"),
                "confidence_interval": confidence_interval,
                "trend": getattr(risk_score, "trend", "unknown"),
                "anomaly_score": getattr(risk_score, "anomaly_score", 0.0),
            },
            "fusion_result": {
                "fraud_score": fused_result.fraud_score,
                "confidence": fused_result.confidence,
                "fraud_types": [ft.value for ft in fused_result.fraud_types],
                "modality_scores": fused_result.modality_scores,
                "fusion_strategy": fused_result.fusion_strategy.value,
            },
            "explanation": explanation.to_dict(),
        }

        if consistency_report:
            report["consistency"] = consistency_report.to_dict()

        if include_evidence:
            report["evidence"] = fused_result.evidence
            # Handle both RiskProfile and RiskScore objects
            factors = getattr(risk_score, "factors", None) or getattr(
                risk_score, "risk_factors", []
            )
            report["risk_factors"] = factors

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    async def _export_pdf(
        self,
        output_path: Path,
        case_id: str,
        fused_result: FusedResult,
        risk_score: RiskScore,
        explanation: Explanation,
        consistency_report: Optional[ConsistencyReport],
        include_evidence: bool,
    ) -> None:
        """Export as PDF."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4, letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

            doc = SimpleDocTemplate(str(output_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title = Paragraph(f"Fraud Detection Report - {case_id}", styles["Title"])
            story.append(title)
            story.append(Spacer(1, 0.3 * inch))

            # Executive Summary
            story.append(Paragraph("Executive Summary", styles["Heading1"]))
            story.append(Paragraph(explanation.summary, styles["BodyText"]))
            story.append(Spacer(1, 0.2 * inch))

            # Risk Assessment
            story.append(Paragraph("Risk Assessment", styles["Heading1"]))
            risk_data = [
                ["Metric", "Value"],
                ["Overall Risk", f"{risk_score.overall_risk:.2%}"],
                ["Risk Level", risk_score.risk_level.upper()],
                ["Confidence", f"{fused_result.confidence:.2%}"],
                ["Trend", risk_score.trend.capitalize()],
            ]

            risk_table = Table(risk_data)
            risk_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(risk_table)
            story.append(Spacer(1, 0.2 * inch))

            # Risk Factors
            story.append(Paragraph("Risk Factors", styles["Heading1"]))
            for factor in explanation.risk_factors[:5]:
                story.append(Paragraph(factor, styles["BodyText"]))
            story.append(Spacer(1, 0.2 * inch))

            # Recommendations
            story.append(Paragraph("Recommendations", styles["Heading1"]))
            for rec in explanation.recommendations:
                story.append(Paragraph(rec, styles["BodyText"]))

            # Build PDF
            doc.build(story)

        except ImportError:
            # Fallback to simple text file
            with open(output_path.with_suffix(".txt"), "w") as f:
                f.write(f"Fraud Detection Report - {case_id}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Executive Summary:\n{explanation.summary}\n\n")
                f.write(f"Risk Level: {risk_score.risk_level}\n")
                f.write(f"Overall Risk: {risk_score.overall_risk:.2%}\n\n")
                f.write("Risk Factors:\n")
                for factor in explanation.risk_factors:
                    f.write(f"{factor}\n")
                f.write("\nRecommendations:\n")
                for rec in explanation.recommendations:
                    f.write(f"{rec}\n")

    async def _export_html(
        self,
        output_path: Path,
        case_id: str,
        fused_result: FusedResult,
        risk_score: RiskScore,
        explanation: Explanation,
        consistency_report: Optional[ConsistencyReport],
        include_evidence: bool,
    ) -> None:
        """Export as HTML."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fraud Detection Report - {case_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .risk-critical {{ color: #e74c3c; font-weight: bold; }}
                .risk-high {{ color: #e67e22; font-weight: bold; }}
                .risk-medium {{ color: #f39c12; }}
                .risk-low {{ color: #27ae60; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #34495e; color: white; }}
                .recommendation {{ background-color: #ecf0f1; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Fraud Detection Report</h1>
            <p><strong>Case ID:</strong> {case_id}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Executive Summary</h2>
            <p>{explanation.summary}</p>
            
            <h2>Risk Assessment</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Overall Risk</td>
                    <td class="risk-{risk_score.risk_level}">{risk_score.overall_risk:.2%}</td>
                </tr>
                <tr>
                    <td>Risk Level</td>
                    <td class="risk-{risk_score.risk_level}">{risk_score.risk_level.upper()}</td>
                </tr>
                <tr>
                    <td>Confidence</td>
                    <td>{fused_result.confidence:.2%}</td>
                </tr>
                <tr>
                    <td>Trend</td>
                    <td>{risk_score.trend.capitalize()}</td>
                </tr>
            </table>
            
            <h2>Risk Factors</h2>
            <ul>
                {"".join(f"<li>{factor}</li>" for factor in explanation.risk_factors[:5])}
            </ul>
            
            <h2>Recommendations</h2>
            {"".join(f'<div class="recommendation">{rec}</div>' for rec in explanation.recommendations)}
            
            <h2>Confidence Explanation</h2>
            <p>{explanation.confidence_explanation}</p>
        """

        if consistency_report and consistency_report.inconsistency_count > 0:
            html_content += f"""
            <h2>Consistency Issues</h2>
            <p>Found {consistency_report.inconsistency_count} inconsistencies across modalities.</p>
            <ul>
                {"".join(f"<li>{inc}</li>" for inc in consistency_report.high_risk_inconsistencies[:5])}
            </ul>
            """

        html_content += """
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)
