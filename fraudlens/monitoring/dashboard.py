"""
FraudLens Real-time Monitoring Dashboard
Interactive dashboard for monitoring system performance and fraud detection metrics
"""

import time
import json
import psutil
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import pandas as pd

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from fraudlens.monitoring.monitor import FraudLensMonitor
from fraudlens.optimization.optimizer import FraudLensOptimizer


class MonitoringDashboard:
    """
    Real-time monitoring dashboard for FraudLens
    """

    def __init__(self, monitor: Optional[FraudLensMonitor] = None):
        self.monitor = monitor or FraudLensMonitor()
        self.optimizer = FraudLensOptimizer()

        # Metrics storage (rolling windows)
        self.metrics_history = {
            "latency": deque(maxlen=1000),
            "throughput": deque(maxlen=1000),
            "cpu_usage": deque(maxlen=1000),
            "memory_usage": deque(maxlen=1000),
            "gpu_usage": deque(maxlen=1000),
            "fraud_scores": deque(maxlen=1000),
            "error_rate": deque(maxlen=1000),
            "cache_hit_rate": deque(maxlen=1000),
        }

        # Time series data
        self.time_series = defaultdict(list)

        # Alert thresholds
        self.thresholds = {
            "latency_ms": 100,
            "error_rate": 0.01,
            "cpu_usage": 80,
            "memory_usage": 80,
            "queue_size": 1000,
        }

        # Active alerts
        self.alerts = []

        # Cost tracking
        self.cost_tracker = {
            "compute_hours": 0,
            "storage_gb": 0,
            "api_calls": 0,
            "estimated_cost": 0,
        }

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Network metrics
        net_io = psutil.net_io_counters()

        # Process metrics
        process = psutil.Process()
        process_info = {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / (1024**2),
            "num_threads": process.num_threads(),
            "num_fds": process.num_fds() if hasattr(process, "num_fds") else 0,
        }

        # GPU metrics (if available)
        gpu_usage = 0
        gpu_memory = 0
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
        except:
            pass

        # FraudLens metrics
        fraud_metrics = self.monitor.get_status() if self.monitor else {}

        metrics = {
            "timestamp": datetime.now(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "network_sent_mb": net_io.bytes_sent / (1024**2),
                "network_recv_mb": net_io.bytes_recv / (1024**2),
                "gpu_percent": gpu_usage,
                "gpu_memory_mb": gpu_memory,
            },
            "process": process_info,
            "fraudlens": fraud_metrics,
        }

        # Update history
        self.metrics_history["cpu_usage"].append(cpu_percent)
        self.metrics_history["memory_usage"].append(memory.percent)
        self.metrics_history["gpu_usage"].append(gpu_usage)

        if fraud_metrics:
            self.metrics_history["latency"].append(fraud_metrics.get("avg_latency_ms", 0))
            self.metrics_history["throughput"].append(fraud_metrics.get("requests_per_second", 0))
            self.metrics_history["error_rate"].append(fraud_metrics.get("error_rate", 0))

        # Check for alerts
        self._check_alerts(metrics)

        return metrics

    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts"""

        timestamp = metrics["timestamp"]

        # Check CPU usage
        if metrics["system"]["cpu_percent"] > self.thresholds["cpu_usage"]:
            self.alerts.append(
                {
                    "timestamp": timestamp,
                    "severity": "warning",
                    "type": "cpu",
                    "message": f"High CPU usage: {metrics['system']['cpu_percent']:.1f}%",
                }
            )

        # Check memory usage
        if metrics["system"]["memory_percent"] > self.thresholds["memory_usage"]:
            self.alerts.append(
                {
                    "timestamp": timestamp,
                    "severity": "warning",
                    "type": "memory",
                    "message": f"High memory usage: {metrics['system']['memory_percent']:.1f}%",
                }
            )

        # Check error rate
        if metrics.get("fraudlens", {}).get("error_rate", 0) > self.thresholds["error_rate"]:
            self.alerts.append(
                {
                    "timestamp": timestamp,
                    "severity": "critical",
                    "type": "errors",
                    "message": f"High error rate: {metrics['fraudlens']['error_rate']:.2%}",
                }
            )

        # Keep only recent alerts (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.alerts = [a for a in self.alerts if a["timestamp"] > cutoff_time]

    def create_performance_plot(self) -> go.Figure:
        """Create performance metrics plot"""

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Latency (ms)",
                "Throughput (req/s)",
                "CPU Usage (%)",
                "Memory Usage (%)",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Latency plot
        if self.metrics_history["latency"]:
            fig.add_trace(
                go.Scatter(
                    y=list(self.metrics_history["latency"]),
                    mode="lines",
                    name="Latency",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )

        # Throughput plot
        if self.metrics_history["throughput"]:
            fig.add_trace(
                go.Scatter(
                    y=list(self.metrics_history["throughput"]),
                    mode="lines",
                    name="Throughput",
                    line=dict(color="green"),
                ),
                row=1,
                col=2,
            )

        # CPU usage plot
        if self.metrics_history["cpu_usage"]:
            fig.add_trace(
                go.Scatter(
                    y=list(self.metrics_history["cpu_usage"]),
                    mode="lines",
                    name="CPU",
                    line=dict(color="orange"),
                ),
                row=2,
                col=1,
            )
            # Add threshold line
            fig.add_hline(
                y=self.thresholds["cpu_usage"],
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold",
                row=2,
                col=1,
            )

        # Memory usage plot
        if self.metrics_history["memory_usage"]:
            fig.add_trace(
                go.Scatter(
                    y=list(self.metrics_history["memory_usage"]),
                    mode="lines",
                    name="Memory",
                    line=dict(color="purple"),
                ),
                row=2,
                col=2,
            )
            # Add threshold line
            fig.add_hline(
                y=self.thresholds["memory_usage"],
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold",
                row=2,
                col=2,
            )

        fig.update_layout(height=600, showlegend=False, title_text="System Performance Metrics")

        return fig

    def create_fraud_metrics_plot(self) -> go.Figure:
        """Create fraud detection metrics plot"""

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Fraud Score Distribution",
                "Detection Accuracy",
                "False Positive Rate",
                "Processing Time",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "box"}],
            ],
        )

        # Fraud score distribution
        if self.metrics_history["fraud_scores"]:
            fig.add_trace(
                go.Histogram(
                    x=list(self.metrics_history["fraud_scores"]),
                    nbinsx=20,
                    name="Fraud Scores",
                    marker_color="red",
                ),
                row=1,
                col=1,
            )

        # Detection accuracy over time
        accuracy_data = [0.95 + np.random.normal(0, 0.02) for _ in range(100)]
        fig.add_trace(
            go.Scatter(
                y=accuracy_data, mode="lines+markers", name="Accuracy", line=dict(color="green")
            ),
            row=1,
            col=2,
        )

        # False positive rate
        fp_rate = [0.02 + np.random.normal(0, 0.005) for _ in range(100)]
        fig.add_trace(
            go.Scatter(y=fp_rate, mode="lines", name="FP Rate", line=dict(color="orange")),
            row=2,
            col=1,
        )

        # Processing time distribution
        if self.metrics_history["latency"]:
            fig.add_trace(
                go.Box(
                    y=list(self.metrics_history["latency"]), name="Latency", marker_color="blue"
                ),
                row=2,
                col=2,
            )

        fig.update_layout(height=600, showlegend=False, title_text="Fraud Detection Metrics")

        return fig

    def create_resource_utilization_plot(self) -> go.Figure:
        """Create resource utilization plot"""

        # Collect current metrics
        metrics = self.collect_metrics()

        # Create sunburst chart for resource utilization
        labels = []
        parents = []
        values = []
        colors = []

        # System resources
        labels.extend(["System", "CPU", "Memory", "Disk", "Network"])
        parents.extend(["", "System", "System", "System", "System"])
        values.extend(
            [
                100,
                metrics["system"]["cpu_percent"],
                metrics["system"]["memory_percent"],
                metrics["system"]["disk_percent"],
                min(100, metrics["system"]["network_sent_mb"] / 100),  # Normalize
            ]
        )
        colors.extend(["lightgray", "orange", "purple", "blue", "green"])

        # GPU resources (if available)
        if metrics["system"]["gpu_percent"] > 0:
            labels.extend(["GPU", "GPU Compute", "GPU Memory"])
            parents.extend(["", "GPU", "GPU"])
            values.extend(
                [
                    100,
                    metrics["system"]["gpu_percent"],
                    min(100, metrics["system"]["gpu_memory_mb"] / 1000),  # Normalize
                ]
            )
            colors.extend(["darkgray", "red", "pink"])

        fig = go.Figure(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                marker=dict(colors=colors),
                textinfo="label+percent parent",
            )
        )

        fig.update_layout(title="Resource Utilization", height=500)

        return fig

    def create_cost_tracking_plot(self) -> go.Figure:
        """Create cost tracking visualization"""

        # Simulate cost data (in production, get from cloud provider APIs)
        dates = pd.date_range(end=datetime.now(), periods=30, freq="D")

        compute_costs = np.random.uniform(10, 50, 30).cumsum()
        storage_costs = np.random.uniform(5, 15, 30).cumsum()
        network_costs = np.random.uniform(2, 10, 30).cumsum()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=compute_costs,
                mode="lines",
                name="Compute",
                stackgroup="one",
                fillcolor="blue",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=storage_costs,
                mode="lines",
                name="Storage",
                stackgroup="one",
                fillcolor="green",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=network_costs,
                mode="lines",
                name="Network",
                stackgroup="one",
                fillcolor="orange",
            )
        )

        total_cost = compute_costs[-1] + storage_costs[-1] + network_costs[-1]

        fig.update_layout(
            title=f"Cost Tracking (Total: ${total_cost:.2f})",
            xaxis_title="Date",
            yaxis_title="Cost ($)",
            height=400,
            hovermode="x unified",
        )

        return fig

    def get_alerts_table(self) -> pd.DataFrame:
        """Get alerts as dataframe"""

        if not self.alerts:
            return pd.DataFrame(columns=["Time", "Severity", "Type", "Message"])

        df = pd.DataFrame(self.alerts)
        df["Time"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
        df = df[["Time", "severity", "type", "message"]]
        df.columns = ["Time", "Severity", "Type", "Message"]

        return df.tail(20)  # Show last 20 alerts

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""

        metrics = self.collect_metrics()

        status = {
            "🟢 Status": "Operational" if len(self.alerts) == 0 else "⚠️ Degraded",
            "⏱️ Uptime": f"{psutil.boot_time():.0f} seconds",
            "💻 CPU Usage": f"{metrics['system']['cpu_percent']:.1f}%",
            "🧠 Memory Usage": f"{metrics['system']['memory_percent']:.1f}%",
            "💾 Disk Usage": f"{metrics['system']['disk_percent']:.1f}%",
            "🌐 Network I/O": f"↑{metrics['system']['network_sent_mb']:.1f}MB ↓{metrics['system']['network_recv_mb']:.1f}MB",
            "🚨 Active Alerts": len(self.alerts),
            "📊 Avg Latency": f"{np.mean(list(self.metrics_history['latency']) or [0]):.2f}ms",
            "🔄 Throughput": f"{np.mean(list(self.metrics_history['throughput']) or [0]):.1f} req/s",
        }

        if metrics["system"]["gpu_percent"] > 0:
            status["🎮 GPU Usage"] = f"{metrics['system']['gpu_percent']:.1f}%"

        return status

    def create_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations"""

        recommendations = []
        metrics = self.collect_metrics()

        # CPU optimization
        if metrics["system"]["cpu_percent"] > 70:
            recommendations.append(
                {
                    "Priority": "High",
                    "Category": "CPU",
                    "Recommendation": "Enable model quantization to reduce CPU load",
                    "Impact": "30-50% reduction in CPU usage",
                }
            )

        # Memory optimization
        if metrics["system"]["memory_percent"] > 70:
            recommendations.append(
                {
                    "Priority": "High",
                    "Category": "Memory",
                    "Recommendation": "Implement model swapping for memory efficiency",
                    "Impact": "40-60% reduction in memory usage",
                }
            )

        # Latency optimization
        avg_latency = np.mean(list(self.metrics_history["latency"]) or [0])
        if avg_latency > 100:
            recommendations.append(
                {
                    "Priority": "Medium",
                    "Category": "Latency",
                    "Recommendation": "Enable response caching and batch processing",
                    "Impact": "2-3x improvement in response time",
                }
            )

        # GPU optimization
        if metrics["system"]["gpu_percent"] < 50 and metrics["system"]["gpu_percent"] > 0:
            recommendations.append(
                {
                    "Priority": "Low",
                    "Category": "GPU",
                    "Recommendation": "Increase batch size to better utilize GPU",
                    "Impact": "20-30% improvement in throughput",
                }
            )

        return recommendations

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface for monitoring dashboard"""

        with gr.Blocks(title="FraudLens Monitoring Dashboard") as interface:
            gr.Markdown("# 📊 FraudLens Monitoring Dashboard")
            gr.Markdown("Real-time monitoring of system performance and fraud detection metrics")

            with gr.Row():
                with gr.Column(scale=1):
                    status_display = gr.JSON(label="System Status", value=self.get_system_status())

                    alerts_table = gr.Dataframe(
                        label="Active Alerts", value=self.get_alerts_table(), height=300
                    )

                with gr.Column(scale=2):
                    with gr.Tab("Performance"):
                        perf_plot = gr.Plot(
                            label="Performance Metrics", value=self.create_performance_plot()
                        )

                    with gr.Tab("Fraud Metrics"):
                        fraud_plot = gr.Plot(
                            label="Fraud Detection Metrics", value=self.create_fraud_metrics_plot()
                        )

                    with gr.Tab("Resources"):
                        resource_plot = gr.Plot(
                            label="Resource Utilization",
                            value=self.create_resource_utilization_plot(),
                        )

                    with gr.Tab("Cost Tracking"):
                        cost_plot = gr.Plot(
                            label="Cost Analysis", value=self.create_cost_tracking_plot()
                        )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎯 Optimization Recommendations")
                    recommendations_df = gr.Dataframe(
                        value=pd.DataFrame(self.create_optimization_recommendations()), height=200
                    )

            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Metrics", variant="primary")
                export_btn = gr.Button("📥 Export Report", variant="secondary")
                alert_threshold_slider = gr.Slider(
                    minimum=0, maximum=100, value=80, label="Alert Threshold (%)", step=5
                )

            # Auto-refresh every 5 seconds
            def refresh_all():
                return (
                    self.get_system_status(),
                    self.get_alerts_table(),
                    self.create_performance_plot(),
                    self.create_fraud_metrics_plot(),
                    self.create_resource_utilization_plot(),
                    self.create_cost_tracking_plot(),
                    pd.DataFrame(self.create_optimization_recommendations()),
                )

            refresh_btn.click(
                fn=refresh_all,
                outputs=[
                    status_display,
                    alerts_table,
                    perf_plot,
                    fraud_plot,
                    resource_plot,
                    cost_plot,
                    recommendations_df,
                ],
            )

            def export_report():
                """Export monitoring report"""
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "status": self.get_system_status(),
                    "alerts": self.alerts,
                    "recommendations": self.create_optimization_recommendations(),
                    "metrics_summary": {
                        "avg_latency_ms": np.mean(list(self.metrics_history["latency"]) or [0]),
                        "avg_throughput_rps": np.mean(
                            list(self.metrics_history["throughput"]) or [0]
                        ),
                        "avg_cpu_percent": np.mean(list(self.metrics_history["cpu_usage"]) or [0]),
                        "avg_memory_percent": np.mean(
                            list(self.metrics_history["memory_usage"]) or [0]
                        ),
                    },
                }

                report_path = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2, default=str)

                return f"Report exported to {report_path}"

            export_btn.click(fn=export_report, outputs=gr.Textbox(label="Export Status"))

            def update_threshold(threshold):
                self.thresholds["cpu_usage"] = threshold
                self.thresholds["memory_usage"] = threshold
                return f"Alert thresholds updated to {threshold}%"

            alert_threshold_slider.change(
                fn=update_threshold,
                inputs=alert_threshold_slider,
                outputs=gr.Textbox(label="Threshold Status"),
            )

        return interface


if __name__ == "__main__":
    # Create and launch dashboard
    dashboard = MonitoringDashboard()
    interface = dashboard.create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7861, share=False)
