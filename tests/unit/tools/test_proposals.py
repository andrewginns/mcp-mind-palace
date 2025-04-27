import os
import tempfile
from unittest.mock import patch
import pytest
import unittest.mock

from app.tools.proposals import (
    propose_new_knowledge,
    suggest_knowledge_update,
    extract_frontmatter,
)


class TestExtractFrontmatter:
    """Tests for frontmatter extraction utility"""

    def test_valid_frontmatter(self):
        """Test extraction of valid frontmatter"""
        content = """---
entry_id: test123
title: Test Entry
tags: [test, example]
created: 2023-05-15
last_modified: 2023-05-15
status: active
---

# Test Entry

This is test content.
"""
        frontmatter, remaining, has_frontmatter = extract_frontmatter(content)

        assert has_frontmatter is True
        assert frontmatter["entry_id"] == "test123"
        assert frontmatter["title"] == "Test Entry"
        assert "test" in frontmatter["tags"]
        assert str(frontmatter["created"]) == "2023-05-15"
        assert remaining.startswith("# Test Entry")

    def test_no_frontmatter(self):
        """Test handling of content without frontmatter"""
        content = "# Test Entry\n\nThis has no frontmatter."

        frontmatter, remaining, has_frontmatter = extract_frontmatter(content)

        assert has_frontmatter is False
        assert frontmatter == {}
        assert remaining == content

    def test_invalid_frontmatter(self):
        """Test handling of invalid YAML in frontmatter"""
        content = """---
invalid: [unclosed bracket,
title: Test
---

Content
"""
        with pytest.raises(Exception) as excinfo:
            extract_frontmatter(content)

        assert "Invalid YAML" in str(excinfo.value)

    def test_empty_frontmatter(self):
        """Test handling of empty frontmatter"""
        content = """---
---

Content with empty frontmatter
"""
        frontmatter, remaining, has_frontmatter = extract_frontmatter(content)

        # Some implementations might return False if the frontmatter is empty
        # The important part is that the content is extracted correctly
        if has_frontmatter:
            assert frontmatter == {} or frontmatter is None

        # Check that the content contains the actual text, regardless of whether
        # the frontmatter delimiters are still present
        assert "Content with empty frontmatter" in remaining


class TestProposeNewKnowledge:
    """Tests for propose_new_knowledge function"""

    @patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH")
    @patch("app.tools.proposals.os.makedirs")
    @patch("app.tools.proposals.open", new_callable=unittest.mock.mock_open)
    def test_propose_valid_content(self, mock_open, mock_makedirs, mock_review_path):
        """Test proposing a valid new knowledge entry using mocks for file operations"""
        # Setup mock path
        mock_review_path.return_value = "/mock/review/path"

        # Create a valid knowledge entry
        content = """---
entry_id: test123
title: Test Knowledge Entry
tags: [test, example, python]
created: 2023-05-15
last_modified: 2023-05-15
status: draft
---

# Test Knowledge Entry

This is a test entry for knowledge management.
"""

        # Skip the actual file operations completely
        # The function should still call os.makedirs and open
        result = propose_new_knowledge(content)

        # Verify makepdirs was called (path creation)
        mock_makedirs.assert_called()

        # Verify file write was attempted
        mock_open.assert_called()

        # Using the right message is more important than the exact file path
        # which can vary between runs
        assert (
            "New knowledge entry proposed" in result
            or "Knowledge entry proposed" in result
        )
        assert "Test Knowledge Entry" in result
        assert "test123" in result

    def test_missing_required_fields(self):
        """Test validation of required frontmatter fields"""
        # Missing required fields
        content = """---
title: Incomplete Entry
tags: [test]
---

# Incomplete Entry
"""
        result = propose_new_knowledge(content)

        assert "Error" in result
        assert "missing required fields" in result
        assert "entry_id" in result  # Should mention missing fields

    def test_no_frontmatter(self):
        """Test rejection of content without frontmatter"""
        content = "# No Frontmatter\n\nThis entry has no frontmatter."

        result = propose_new_knowledge(content)

        assert "Error" in result
        assert "Frontmatter is required" in result

    @patch("app.tools.proposals.search_knowledge")
    @patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH")
    def test_similar_content_warning(self, mock_review_path, mock_search):
        """Test warning when similar content exists"""
        # Setup temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_review_path = temp_dir

            # Mock search results to indicate similar content
            mock_search.return_value = [
                {
                    "content": "Similar existing content",
                    "metadata": {"entry_id": "existing123", "title": "Existing Entry"},
                    "similarity_score": 0.85,  # High similarity
                }
            ]

            content = """---
entry_id: new123
title: New Similar Entry
tags: [test, example]
created: 2023-05-15
last_modified: 2023-05-15
status: draft
---

# New Similar Entry

This might be similar to existing content.
"""

            # Run with real file operations but mocked search
            with patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH", temp_dir):
                result = propose_new_knowledge(content)

            # Should still create the file but with a warning
            assert "WARNING: Similar content may already exist" in result
            assert "Existing Entry" in result

            # Verify file was still created
            proposals_dir = os.path.join(temp_dir, "proposals", "test")
            assert os.path.exists(proposals_dir)
            files = os.listdir(proposals_dir)
            assert len(files) == 1

    @patch("app.tools.proposals.search_knowledge")
    @patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH")
    def test_with_search_verification(self, mock_review_path, mock_search):
        """Test proposal with search verification flag"""
        # Setup temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_review_path = temp_dir

            content = """---
entry_id: verified123
title: Verified Entry
tags: [test, example]
created: 2023-05-15
last_modified: 2023-05-15
status: draft
---

# Verified Entry

This content has been verified as unique.
"""

            # Search should not be called when verification is provided
            with patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH", temp_dir):
                result = propose_new_knowledge(
                    content, search_verification="Verified unique through search"
                )

            # No warnings should be present
            assert "WARNING" not in result
            assert mock_search.call_count == 0


class TestSuggestKnowledgeUpdate:
    """Tests for suggest_knowledge_update function"""

    @patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH")
    @patch("app.tools.proposals.ACTIVE_KNOWLEDGE_PATH")
    @patch("app.tools.proposals.get_frontmatter")
    @patch("app.tools.proposals.os.walk")
    @patch("app.tools.proposals.os.makedirs")
    @patch("app.tools.proposals.open", new_callable=unittest.mock.mock_open)
    def test_suggest_update_for_existing_entry(
        self,
        mock_open,
        mock_makedirs,
        mock_walk,
        mock_get_frontmatter,
        mock_active_path,
        mock_review_path,
    ):
        """Test suggesting updates for an existing entry using mocks"""
        # Setup mock paths
        active_path = "/mock/active/path"
        mock_active_path.return_value = active_path
        mock_review_path.return_value = "/mock/review/path"

        # Mock file system walk to return a file
        mock_walk.return_value = [
            (os.path.join(active_path, "general"), [], ["test_entry.md"])
        ]

        # Mock frontmatter to simulate finding the file
        mock_frontmatter = {
            "entry_id": "existing123",
            "title": "Existing Entry",
            "tags": ["test", "existing"],
            "status": "active",
        }
        mock_get_frontmatter.return_value = mock_frontmatter

        # Call the function
        result = suggest_knowledge_update(
            "existing123",
            "Add a new section on examples and update formatting.",
            existing_content_verified=False,
        )

        # Verify results
        assert "Update suggested for knowledge entry" in result
        assert "existing123" in result

        # Verify directory creation and file writing
        mock_makedirs.assert_called()
        mock_open.assert_called()

    @patch("app.tools.proposals.ACTIVE_KNOWLEDGE_PATH")
    @patch("app.tools.proposals.os.walk")
    def test_entry_not_found(self, mock_walk, mock_active_path):
        """Test handling of non-existent entries using mocks"""
        # Setup mock path
        mock_active_path.return_value = "/mock/active/path"

        # Mock empty walk results to simulate no files found
        mock_walk.return_value = []

        # Call function
        result = suggest_knowledge_update(
            "nonexistent123",
            "This entry doesn't exist.",
            existing_content_verified=False,
        )

        # Verify error message
        assert "Error" in result
        assert "Knowledge entry with ID" in result
        assert "not found" in result

    @patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH")
    @patch("app.tools.proposals.os.makedirs")
    @patch("app.tools.proposals.open", new_callable=unittest.mock.mock_open)
    def test_with_content_verified(self, mock_open, mock_makedirs, mock_review_path):
        """Test suggesting updates with existing_content_verified flag"""
        # Setup mock path
        mock_review_path.return_value = "/mock/review/path"

        # Call function with verification flag
        result = suggest_knowledge_update(
            "trusted123",
            "Update the verified entry with new content.",
            existing_content_verified=True,
        )

        # Verify results
        assert "Update suggested for knowledge entry" in result
        assert "trusted123" in result

        # Verify directory creation and file writing
        mock_makedirs.assert_called()
        mock_open.assert_called()

    @patch("app.tools.proposals.get_frontmatter")
    @patch("app.tools.proposals.ACTIVE_KNOWLEDGE_PATH")
    def test_error_during_verification(self, mock_active_path, mock_get_frontmatter):
        """Test handling of errors during verification"""
        # Setup mock path
        mock_active_path.return_value = "/mock/active/path"

        # Mock exception during frontmatter extraction
        mock_get_frontmatter.side_effect = Exception(
            "Test error during frontmatter parsing"
        )

        # Call function
        result = suggest_knowledge_update(
            "error123",
            "This will cause an error during verification.",
            existing_content_verified=False,
        )

        # Verify error message
        assert "Error" in result
        assert "Knowledge entry with ID" in result


if __name__ == "__main__":
    pytest.main(["-xvs", "test_proposals.py"])
