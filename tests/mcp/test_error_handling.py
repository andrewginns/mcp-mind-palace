import pytest
import os
from unittest.mock import patch, MagicMock

from app.tools.search import search_knowledge, get_entry_details
from app.tools.proposals import propose_new_knowledge, suggest_knowledge_update


class TestErrorHandlingMCP:
    """
    Tests for error handling in the MCP tools.
    Focuses on ensuring tools gracefully handle error conditions and edge cases.
    """

    @pytest.mark.timeout(2)
    def test_search_knowledge_error_handling(self):
        """Test search_knowledge handles errors gracefully."""
        # Test with non-existent collection
        with patch(
            "app.tools.search.chroma_client.get_collection"
        ) as mock_get_collection:
            # Simulate ChromaDB raising an exception when collection doesn't exist
            mock_get_collection.side_effect = ValueError("Collection does not exist")

            # The function should handle the error and return an empty list
            result = search_knowledge("test query")
            assert isinstance(result, list)
            assert len(result) == 0

        # Test with embedder service failure
        with patch(
            "app.tools.search.chroma_client.get_collection"
        ) as mock_get_collection:
            mock_collection = MagicMock()
            mock_get_collection.return_value = mock_collection

            with patch("app.tools.search.Embedder.generate_embedding") as mock_embedder:
                # Simulate embedding service failure
                mock_embedder.side_effect = Exception("Embedding service unavailable")

                # The function should handle the error
                result = search_knowledge("test query")
                assert isinstance(result, list)
                assert len(result) == 0

    @pytest.mark.timeout(2)
    def test_get_entry_details_error_handling(self):
        """Test get_entry_details handles errors gracefully."""
        # Test with non-existent entry_id
        with patch(
            "app.tools.search.chroma_client.get_collection"
        ) as mock_get_collection:
            mock_collection = MagicMock()
            # Empty result when entry_id is not found
            mock_collection.get.return_value = {
                "ids": [],
                "documents": [],
                "metadatas": [],
            }
            mock_get_collection.return_value = mock_collection

            result = get_entry_details("non-existent-id")
            assert "error" in result
            assert "not found" in result["error"]

        # Test with ChromaDB connection issue
        with patch(
            "app.tools.search.chroma_client.get_collection"
        ) as mock_get_collection:
            mock_get_collection.side_effect = Exception("ChromaDB connection error")

            result = get_entry_details("test-id")
            assert "error" in result

    @pytest.mark.timeout(2)
    def test_propose_new_knowledge_validation(self):
        """Test propose_new_knowledge validates input content properly."""
        # Test with missing frontmatter
        invalid_content = "# This is just a title\n\nNo frontmatter here"

        result = propose_new_knowledge(invalid_content)
        assert "Error" in result
        assert "Frontmatter is required" in result

        # Test with incomplete frontmatter (missing required fields)
        incomplete_frontmatter = """---
title: Test Entry
---

# Test Entry

This is a test.
"""
        result = propose_new_knowledge(incomplete_frontmatter)
        assert "Error" in result
        assert "missing required fields" in result

    @pytest.mark.timeout(2)
    def test_suggest_knowledge_update_validation(self, patched_knowledge_paths):
        """Test suggest_knowledge_update validates entry existence."""
        # Mock the file system walk to return no files
        with patch("os.walk") as mock_walk:
            mock_walk.return_value = [
                (patched_knowledge_paths["active"], [], [])
            ]  # No files

            # Test with non-existent entry_id without verification
            result = suggest_knowledge_update(
                entry_id="non-existent-id",
                suggested_changes="This is a test suggestion",
                existing_content_verified=False,
            )

            # Changed assertion to be more specific about error message
            assert "not found" in result

        # Test with bypassed verification but filesystem write error
        with patch("os.makedirs") as mock_makedirs:
            # Skip the file existence check
            with patch(
                "app.tools.proposals.get_frontmatter",
                return_value={"entry_id": "test-id"},
            ):
                with patch(
                    "builtins.open", side_effect=PermissionError("Permission denied")
                ):
                    result = suggest_knowledge_update(
                        entry_id="test-id",
                        suggested_changes="This is a test suggestion",
                        existing_content_verified=True,  # Skip verification
                    )

                    # Check for an error message (case insensitive)
                    assert "error" in result.lower() or "Error" in result

    @pytest.mark.timeout(2)
    def test_malformed_yaml_handling(self):
        """Test handling of malformed YAML in frontmatter."""
        # Create content with invalid YAML in frontmatter
        invalid_yaml = """---
entry_id: test-entry
title: Test Entry
tags: [unclosed, array,
created: 2023-05-01
last_modified: 2023-05-01
status: active
---

# Test Entry

This is a test.
"""

        result = propose_new_knowledge(invalid_yaml)
        assert "Error" in result
        assert "Invalid YAML" in result

    @pytest.mark.timeout(2)
    @patch("app.tools.search.chroma_client.get_collection")
    def test_search_with_empty_results(self, mock_get_collection):
        """Test search behavior with empty results."""
        # Setup mock to return empty results
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_get_collection.return_value = mock_collection

        # Mock embedder to avoid actual API calls
        with patch(
            "app.tools.search.Embedder.generate_embedding", return_value=[0.1] * 384
        ):
            result = search_knowledge("query with no matches")

            # Should return a list with one item that has empty content and guidance
            assert isinstance(result, list)
            assert len(result) == 1
            assert "relevance_comment" in result[0]
            assert "No relevant content found" in result[0]["relevance_comment"]
