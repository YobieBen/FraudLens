"""
Multi-agent system for fraud detection.

Implements a coordinator-specialist pattern where specialized agents
focus on specific fraud types and a coordinator orchestrates their work.

Author: Yobie Benjamin
Date: 2026-02-28
"""

from fraudlens.agents.base import Agent, BaseAgent

__all__ = [
    "Agent",
    "BaseAgent",
]
