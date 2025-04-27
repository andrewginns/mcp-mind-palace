import pytest
from unittest.mock import patch, MagicMock

from mcp.server.fastmcp import FastMCP
from app.tools import register_tools


class TestToolRegistration:
    """
    Tests for the MCP tool registration process.
    These tests verify that all tools are properly registered with the MCP server.
    """

    @pytest.fixture
    def mock_mcp(self):
        """Create a mock MCP server for testing tool registration."""
        mock = MagicMock(spec=FastMCP)
        # Create a decorator that returns a function unchanged but tracks calls
        mock.tool = MagicMock(return_value=lambda x: x)
        return mock

    @pytest.mark.timeout(2)
    def test_tool_registration(self, mock_mcp):
        """
        Test that all Mind Palace tools are properly registered with the MCP server.
        """
        # Register tools with the mock MCP server
        register_tools(mock_mcp)

        # Verify that the tool method was called for each tool
        call_count = mock_mcp.tool.call_count

        # We expect 4 tools to be registered based on the register_tools implementation
        assert call_count == 4, f"Expected 4 tool registrations, got {call_count}"

        # Check if tool decorator was called
        assert mock_mcp.tool.called

    @pytest.mark.timeout(2)
    @patch("app.tools.search.search_knowledge")
    @patch("app.tools.search.get_entry_details")
    @patch("app.tools.proposals.propose_new_knowledge")
    @patch("app.tools.proposals.suggest_knowledge_update")
    def test_all_tools_are_registered(
        self, mock_suggest, mock_propose, mock_details, mock_search, mock_mcp
    ):
        """
        Test that all expected tools are registered.
        """
        # Use the actual register_tools implementation
        register_tools(mock_mcp)

        # Verify tool decorator was called for each function
        mock_mcp.tool.assert_called()

        # Each of the four tools should have had the decorator applied
        assert mock_mcp.tool.call_count == 4, (
            f"Expected 4 tool registrations, got {mock_mcp.tool.call_count}"
        )

    @pytest.mark.timeout(2)
    def test_tool_registration_error_handling(self):
        """
        Test that tool registration handles errors gracefully.
        """
        # Create a mock MCP server
        mock_mcp = MagicMock()

        # Make tool decorator raise an exception when called to simulate a registration error
        def mock_tool_that_raises(*args, **kwargs):
            raise Exception("Test registration error")

        # Also create a return value so it can be called
        mock_decorator = MagicMock()
        mock_decorator.side_effect = mock_tool_that_raises

        # Assign our problematic mock to the MCP object
        mock_mcp.tool = mock_decorator

        # Configure a mock logger to capture log output
        mock_logger = MagicMock()

        # Test that register_tools doesn't propagate the exception
        with patch("logging.error") as mock_log_error:
            # Call register_tools and expect it to handle the error
            register_tools(mock_mcp)

            # Verify that an error was logged
            mock_log_error.assert_called()

            # Now let's verify it was called specifically with an error message
            # that mentions registration failed
            calls = mock_log_error.call_args_list

            # At least one log call should mention registration failure
            registration_error_logged = any(
                "Failed to register" in str(call) for call in calls
            )

            assert registration_error_logged, "No registration error was logged"
