"""Tests for tool calling framework."""

import pytest

from fraudlens.llm.tools import Tool, ToolRegistry, register_tool


class TestTool:
    """Test Tool class."""

    @pytest.mark.asyncio
    async def test_sync_tool_execution(self):
        """Test executing synchronous tool."""

        def add_numbers(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = Tool(
            name="add_numbers",
            description="Add two numbers",
            function=add_numbers,
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        )

        result = await tool.execute(a=5, b=3)
        assert result == 8

    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test executing async tool."""

        async def fetch_data(url: str) -> str:
            """Fetch data from URL."""
            return f"Data from {url}"

        tool = Tool(
            name="fetch_data",
            description="Fetch data",
            function=fetch_data,
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        )

        result = await tool.execute(url="https://example.com")
        assert result == "Data from https://example.com"

    def test_tool_to_llm_format(self):
        """Test converting tool to LLM format."""

        def test_func(param: str) -> str:
            return param

        tool = Tool(
            name="test_tool",
            description="Test tool",
            function=test_func,
            parameters={
                "type": "object",
                "properties": {"param": {"type": "string"}},
            },
        )

        llm_format = tool.to_llm_format()
        assert llm_format["name"] == "test_tool"
        assert llm_format["description"] == "Test tool"
        assert "parameters" in llm_format


class TestToolRegistry:
    """Test ToolRegistry class."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()

        def test_func() -> str:
            return "test"

        registry.register(
            Tool(
                name="test_tool",
                description="Test",
                function=test_func,
                parameters={"type": "object", "properties": {}},
            )
        )

        assert "test_tool" in registry.list_tools()

    def test_get_tool(self):
        """Test getting a tool."""
        registry = ToolRegistry()

        def test_func() -> str:
            return "test"

        tool = Tool(
            name="test_tool",
            description="Test",
            function=test_func,
            parameters={"type": "object", "properties": {}},
        )
        registry.register(tool)

        retrieved = registry.get_tool("test_tool")
        assert retrieved.name == "test_tool"

    def test_get_nonexistent_tool(self):
        """Test getting non-existent tool raises error."""
        registry = ToolRegistry()

        with pytest.raises(KeyError):
            registry.get_tool("nonexistent")

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing tool through registry."""
        registry = ToolRegistry()

        def multiply(a: int, b: int) -> int:
            return a * b

        registry.register(
            Tool(
                name="multiply",
                description="Multiply numbers",
                function=multiply,
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                },
            )
        )

        result = await registry.execute("multiply", a=4, b=5)
        assert result == 20

    def test_get_tool_spec(self):
        """Test getting tool specification."""
        registry = ToolRegistry()

        def test_func(x: int) -> int:
            return x

        registry.register(
            Tool(
                name="test_func",
                description="Test function",
                function=test_func,
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                },
            )
        )

        spec = registry.get_tool_spec("test_func")
        assert spec["name"] == "test_func"
        assert "parameters" in spec

    def test_to_llm_format(self):
        """Test converting all tools to LLM format."""
        registry = ToolRegistry()

        def tool1() -> str:
            return "1"

        def tool2() -> str:
            return "2"

        registry.register(
            Tool(
                name="tool1",
                description="Tool 1",
                function=tool1,
                parameters={"type": "object", "properties": {}},
            )
        )
        registry.register(
            Tool(
                name="tool2",
                description="Tool 2",
                function=tool2,
                parameters={"type": "object", "properties": {}},
            )
        )

        llm_format = registry.to_llm_format()
        assert len(llm_format) == 2
        assert any(t["name"] == "tool1" for t in llm_format)
        assert any(t["name"] == "tool2" for t in llm_format)


class TestToolDecorator:
    """Test register_tool decorator."""

    def test_decorator_registration(self):
        """Test that decorator registers tool."""
        registry = ToolRegistry()

        @register_tool(
            registry=registry,
            description="Test decorator",
            parameters={"type": "object", "properties": {}},
        )
        def decorated_func() -> str:
            return "decorated"

        assert "decorated_func" in registry.list_tools()

    @pytest.mark.asyncio
    async def test_decorated_tool_execution(self):
        """Test executing decorated tool."""
        registry = ToolRegistry()

        @register_tool(
            registry=registry,
            description="Add numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
            },
        )
        def add(a: int, b: int) -> int:
            return a + b

        result = await registry.execute("add", a=10, b=20)
        assert result == 30


class TestBuiltInTools:
    """Test built-in tools."""

    @pytest.mark.asyncio
    async def test_extract_urls_tool(self):
        """Test URL extraction tool."""
        from fraudlens.llm.tools import get_default_registry

        registry = get_default_registry()
        text = "Check out https://example.com and http://test.org"

        result = await registry.execute("extract_urls", text=text)
        assert "https://example.com" in result
        assert "http://test.org" in result

    @pytest.mark.asyncio
    async def test_analyze_urgency_tool(self):
        """Test urgency analysis tool."""
        from fraudlens.llm.tools import get_default_registry

        registry = get_default_registry()
        text = "URGENT! Act now or your account will be deleted!"

        result = await registry.execute("analyze_urgency", text=text)
        assert result["is_urgent"] is True
        assert result["urgency_score"] > 0.5
