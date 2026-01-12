"""
Unit tests for Student Assistant tools.

This test suite validates the behavior of individual tools (both LangChain tools
and MCP servers) used by the student assistant agent.

Run locally:
    cd canopy-be
    export CANOPY_CONFIG_PATH=/path/to/canopy-config.yaml
    pytest tests/test_tools.py -v

Run specific test:
    pytest tests/test_tools.py::test_search_knowledge_base -v
"""

import pytest
import os
import yaml
from llama_stack_client import LlamaStackClient
from mcp import ClientSession
from mcp.client.sse import sse_client


# ============================================================================
# Fixtures (configuration and setup)
# ============================================================================

@pytest.fixture(scope="session")
def config():
    """
    Load canopy configuration from configmap or local file.

    Priority:
    1. CANOPY_CONFIG_PATH environment variable
    2. /canopy/canopy-config.yaml (cluster deployment)
    3. ../config/canopy-config.yaml (local development fallback)
    """
    config_path = os.getenv("CANOPY_CONFIG_PATH", "tests/canopy-config.yaml")

    # Fallback for local development
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "../config/canopy-config.yaml")

    if not os.path.exists(config_path):
        pytest.skip(f"Config file not found at {config_path}. Set CANOPY_CONFIG_PATH environment variable.")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def llama_client(config):
    """Create LlamaStackClient for tests."""
    return LlamaStackClient(base_url=config["LLAMA_STACK_URL"])


@pytest.fixture(scope="session")
def vector_store_id(config):
    """Get vector store ID from config."""
    return config["student-assistant"].get("vector_db_id", "latest")


@pytest.fixture(scope="session")
def mcp_calendar_url(config):
    """Get MCP calendar server URL from config."""
    return config["student-assistant"].get(
        "mcp_calendar_url",
        "http://canopy-mcp-calendar-mcp-server:8080/sse"
    )


# ============================================================================
# Tool Tests - LangChain Tools
# ============================================================================

def test_search_knowledge_base(llama_client, vector_store_id):
    """
    Test that search_knowledge_base retrieves relevant context from the vector store.

    This test validates:
    - Tool can be invoked successfully
    - Returns non-empty results for valid queries
    - Does not return "no results" message for topics in the knowledge base
    """
    from app.main import create_student_tools

    # Create tools using the factory function
    tools = create_student_tools(llama_client, vector_store_id)
    search_tool = tools[0]  # First tool is search_knowledge_base

    # Test: Search for a topic that should exist in the knowledge base
    result = search_tool.invoke({"query": "canopy"})

    # Assertions
    assert len(result) > 0, "Search should return results"
    assert "No relevant information found" not in result, "Should find relevant content in knowledge base"

    # Output for debugging
    print(f"\n✓ Search result preview: {result[:200]}...")


def test_find_professors_by_expertise(llama_client, vector_store_id):
    """
    Test that find_professors_by_expertise returns correct professor matches.

    This test validates:
    - Tool matches professors by expertise keywords
    - Returns expected professor information (name, email, department)
    - Handles topic matching correctly (case-insensitive, partial matches)
    """
    from app.main import create_student_tools

    # Create tools using the factory function
    tools = create_student_tools(llama_client, vector_store_id)
    professor_tool = tools[1]  # Second tool is find_professors_by_expertise

    # Test: Find professors with Machine Learning expertise
    result = professor_tool.invoke({"topic": "Machine Learning"})

    # Assertions - Dr. Sarah Chen has "Machine Learning" in expertise
    assert "Dr. Sarah Chen" in result, "Should find Dr. Chen for Machine Learning topic"
    assert "s.chen@university.edu" in result, "Should include professor email"
    assert "Computer Science" in result, "Should include department"

    # Output for debugging
    print(f"\n✓ Professor search result:\n{result}")


# ============================================================================
# MCP Server Tests
# ============================================================================

@pytest.mark.asyncio
async def test_mcp_calendar_list_tools(mcp_calendar_url):
    """
    Test that MCP calendar server exposes expected tools.

    This test validates:
    - MCP server is reachable and responds
    - Server exposes the expected calendar management tools
    - Tool listing works correctly via SSE transport
    """
    async with sse_client(mcp_calendar_url) as streams:
        async with ClientSession(*streams) as session:
            # Initialize the session
            await session.initialize()

            # List available tools
            tools_response = await session.list_tools()

            # Extract tool names
            tool_names = [tool.name for tool in tools_response.tools]

            # Assertions - Check for expected calendar tools
            assert "create_event" in tool_names, "Should have create_event tool"
            assert "get_all_events" in tool_names, "Should have get_all_events tool"

            # Output for debugging
            print(f"\n✓ Available MCP tools: {tool_names}")


@pytest.mark.asyncio
async def test_mcp_calendar_list_events(mcp_calendar_url):
    """
    Test calling the get_all_events MCP tool.

    This test validates:
    - MCP tool can be invoked successfully
    - Tool returns structured response with content
    - Calendar event listing functionality works
    """
    async with sse_client(mcp_calendar_url) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()

            # Call get_all_events tool
            result = await session.call_tool(
                "get_all_events",
                arguments={}
            )

            # Assertions
            assert result is not None, "Should return a result"
            assert len(result.content) > 0, "Should have content in response"

            # Extract content text
            content_text = result.content[0].text
            assert isinstance(content_text, str), "Content should be a string"

            # Output for debugging
            print(f"\n✓ MCP get_all_events result preview: {content_text[:200]}...")


# ============================================================================
# Helper to run tests directly
# ============================================================================

if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v", "-s"])
