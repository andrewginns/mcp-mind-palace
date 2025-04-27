import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from app.knowledge_management.embedder import Embedder


@pytest.fixture
def mock_chroma_client():
    """
    Create a mock ChromaDB client for testing.
    This allows isolating tests from actual ChromaDB operations.
    """
    mock_client = MagicMock()

    # Setup a mock collection
    mock_collection = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_client.get_or_create_collection.return_value = mock_collection

    return mock_client


@pytest.fixture
def mock_embedder():
    """
    Create a mock Embedder for testing that returns deterministic embeddings.
    This prevents making API calls during tests.
    """
    mock_emb = MagicMock(spec=Embedder)
    mock_emb.generate_embedding.return_value = [0.1] * 1536  # Mock 1536-dim embedding
    return mock_emb


@pytest.fixture
def temp_knowledge_dir():
    """
    Create a temporary directory for knowledge files during tests.
    Automatically cleaned up after test completion.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_markdown_content():
    """
    Provide sample markdown content with frontmatter for testing.
    """
    return """---
entry_id: test123
title: Test Entry
tags: [test, sample, python]
created: 2023-05-15
last_modified: 2023-05-15
status: active
---

# Test Entry

This is sample content for testing purposes.

## Section 1

- Point 1
- Point 2

## Section 2

More content here.
"""


@pytest.fixture
def sample_markdown_file(temp_knowledge_dir, sample_markdown_content):
    """
    Create a sample markdown file in the temporary directory.
    Returns the path to the created file.
    """
    file_path = os.path.join(temp_knowledge_dir, "test_entry.md")
    with open(file_path, "w") as f:
        f.write(sample_markdown_content)
    return file_path


@pytest.fixture
def setup_knowledge_dirs():
    """
    Create a temporary structure mimicking the knowledge base directory structure.
    Includes active, archive, and review directories.
    """
    with tempfile.TemporaryDirectory() as base_dir:
        # Create knowledge base structure
        active_dir = os.path.join(base_dir, "active")
        archive_dir = os.path.join(base_dir, "archive")
        review_dir = os.path.join(base_dir, "review")

        # Create subdirectories
        os.makedirs(active_dir, exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)
        os.makedirs(os.path.join(review_dir, "proposals"), exist_ok=True)
        os.makedirs(os.path.join(review_dir, "updates"), exist_ok=True)

        # Create category subdirectories in active
        os.makedirs(os.path.join(active_dir, "general"), exist_ok=True)
        os.makedirs(os.path.join(active_dir, "python"), exist_ok=True)

        yield {
            "base": base_dir,
            "active": active_dir,
            "archive": archive_dir,
            "review": review_dir,
        }


@pytest.fixture
def patched_knowledge_paths(setup_knowledge_dirs):
    """
    Patch the knowledge path configuration during tests.
    This isolates tests from the actual file system knowledge base.
    """
    paths = setup_knowledge_dirs

    with (
        patch("app.config.KNOWLEDGE_BASE_PATH", paths["base"]),
        patch("app.config.ACTIVE_KNOWLEDGE_PATH", paths["active"]),
        patch("app.config.REVIEW_KNOWLEDGE_PATH", paths["review"]),
    ):
        yield paths
