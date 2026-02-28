"""
Tool calling framework for LLM-based fraud detection.

Enables LLMs to invoke external tools and services during analysis,
following modern function calling patterns.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import inspect
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

from fraudlens.llm.schemas import ToolCall


class Tool(BaseModel):
    """
    Definition of a tool that can be called by an LLM.
    
    Tools extend the LLM's capabilities by providing access to external
    services, databases, and specialized analysis functions.
    """
    
    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="What the tool does")
    parameters: dict[str, Any] = Field(..., description="JSON Schema for parameters")
    function: Callable | None = Field(None, exclude=True, description="The actual function to call")
    
    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "name": "check_url_reputation",
                "description": "Check URL reputation in threat intelligence databases",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to check"}
                    },
                    "required": ["url"]
                }
            }
        }
    
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given arguments.
        
        Args:
            **kwargs: Tool arguments
        
        Returns:
            Tool execution result
        """
        if self.function is None:
            raise ValueError(f"Tool {self.name} has no function implementation")
        
        # Handle both sync and async functions
        if inspect.iscoroutinefunction(self.function):
            return await self.function(**kwargs)
        else:
            return self.function(**kwargs)
    
    def to_llm_format(self) -> dict[str, Any]:
        """
        Convert tool to LLM function calling format.
        
        Returns:
            Tool definition in LLM format
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Provides tool discovery, validation, and execution.
    """
    
    def __init__(self):
        """Initialize tool registry."""
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool to register
        """
        self._tools[tool.name] = tool
    
    def register_function(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> Callable:
        """
        Decorator to register a function as a tool.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: Parameter schema
        
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            tool = Tool(
                name=name,
                description=description,
                parameters=parameters,
                function=func
            )
            self.register(tool)
            return func
        
        return decorator
    
    def get(self, name: str) -> Tool | None:
        """
        Get tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            Tool or None if not found
        """
        return self._tools.get(name)
    
    def get_tool(self, name: str) -> Tool:
        """
        Get tool by name (raises KeyError if not found).
        
        Args:
            name: Tool name
        
        Returns:
            Tool
        
        Raises:
            KeyError: If tool not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]
    
    async def execute(self, name: str, **kwargs) -> Any:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name
            **kwargs: Tool arguments
        
        Returns:
            Tool execution result
        """
        tool = self.get_tool(name)
        return await tool.execute(**kwargs)
    
    def get_tool_spec(self, name: str) -> dict[str, Any]:
        """
        Get tool specification in LLM format.
        
        Args:
            name: Tool name
        
        Returns:
            Tool specification
        """
        tool = self.get_tool(name)
        return tool.to_llm_format()
    
    def list_tools(self) -> list[str]:
        """
        List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def to_llm_format(self) -> list[dict[str, Any]]:
        """
        Convert tools to LLM function calling format.
        
        Returns:
            List of tool definitions in LLM format
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self._tools.values()
        ]
    
    async def execute_tool_call(self, tool_call: ToolCall) -> Any:
        """
        Execute a tool call from an LLM.
        
        Args:
            tool_call: Tool call request
        
        Returns:
            Tool execution result
        
        Raises:
            ValueError: If tool not found
        """
        tool = self.get(tool_call.tool_name)
        if tool is None:
            raise ValueError(f"Tool not found: {tool_call.tool_name}")
        
        return await tool.execute(**tool_call.arguments)


# Global tool registry
_global_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry.
    
    Returns:
        Global ToolRegistry instance
    """
    return _global_registry


def get_default_registry() -> ToolRegistry:
    """
    Get the default/global tool registry with built-in tools.
    
    Returns:
        Global ToolRegistry instance
    """
    return _global_registry


def register_tool(
    description: str,
    parameters: dict[str, Any],
    name: str | None = None,
    registry: ToolRegistry | None = None,
) -> Callable:
    """
    Decorator to register a function as a tool.
    
    Args:
        description: Tool description
        parameters: Parameter schema
        name: Tool name (defaults to function name)
        registry: Registry to register in (defaults to global registry)
    
    Returns:
        Decorator function
    
    Example:
        ```python
        @register_tool(
            description="Check if URL is malicious",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"]
            }
        )
        async def check_url(url: str) -> dict:
            # Implementation
            return {"is_malicious": False}
        ```
    """
    target_registry = registry or _global_registry
    
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool = Tool(
            name=tool_name,
            description=description,
            parameters=parameters,
            function=func
        )
        target_registry.register(tool)
        return func
    
    return decorator


# Built-in fraud detection tools


@register_tool(
    description="Analyze URL for phishing, typosquatting, and malicious patterns",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to analyze"}
        },
        "required": ["url"]
    },
    name="analyze_url"
)
async def analyze_url(url: str) -> dict[str, Any]:
    """
    Analyze URL for fraud indicators.
    
    Args:
        url: URL to analyze
    
    Returns:
        Analysis results
    """
    # Placeholder implementation
    # In real implementation, this would call threat intelligence APIs
    return {
        "url": url,
        "is_suspicious": "paypal" in url.lower() and "paypal.com" not in url.lower(),
        "typosquatting_detected": False,
        "threat_score": 0.0,
        "categories": []
    }


@register_tool(
    description="Check domain registration age (newly registered domains are suspicious)",
    parameters={
        "type": "object",
        "properties": {
            "domain": {"type": "string", "description": "Domain name to check"}
        },
        "required": ["domain"]
    },
    name="check_domain_age"
)
async def check_domain_age(domain: str) -> dict[str, Any]:
    """
    Check domain registration age.
    
    Args:
        domain: Domain name
    
    Returns:
        Domain age information
    """
    # Placeholder implementation
    return {
        "domain": domain,
        "age_days": 365,
        "is_newly_registered": False,
        "registration_date": "2023-01-01"
    }


@register_tool(
    description="Extract all URLs from text content",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to extract URLs from"}
        },
        "required": ["text"]
    },
    name="extract_urls"
)
async def extract_urls(text: str) -> dict[str, Any]:
    """
    Extract URLs from text.
    
    Args:
        text: Text content
    
    Returns:
        Extracted URLs
    """
    import re
    
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    
    return {
        "urls": urls,
        "count": len(urls)
    }


@register_tool(
    description="Analyze text for urgency and pressure tactics (common in phishing)",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to analyze"}
        },
        "required": ["text"]
    },
    name="analyze_urgency"
)
async def analyze_urgency(text: str) -> dict[str, Any]:
    """
    Analyze text for urgency tactics.
    
    Args:
        text: Text to analyze
    
    Returns:
        Urgency analysis
    """
    urgency_keywords = [
        "urgent", "immediately", "act now", "limited time",
        "expires", "suspend", "verify now", "confirm now",
        "24 hours", "asap"
    ]
    
    text_lower = text.lower()
    found_keywords = [kw for kw in urgency_keywords if kw in text_lower]
    
    urgency_score = min(len(found_keywords) / 3.0, 1.0)  # Cap at 1.0
    
    return {
        "urgency_score": urgency_score,
        "found_keywords": found_keywords,
        "is_urgent": urgency_score > 0.5
    }
