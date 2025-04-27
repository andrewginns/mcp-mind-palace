from unittest.mock import MagicMock, patch
import pytest
import uuid
import chromadb
from chromadb.config import Settings
from typing import Dict, Any, Tuple

from app.tools.search import search_knowledge, get_entry_details
from app.knowledge_management.embedder import Embedder


@pytest.fixture
def in_memory_chroma():
    """Create an in-memory ChromaDB client for testing."""
    client = chromadb.Client(
        Settings(
            is_persistent=False,
            allow_reset=True,
        )
    )
    client.reset()
    yield client


@pytest.fixture
def test_collection(in_memory_chroma):
    """Create a test collection with sample data."""
    # Create a unique collection name for this test run
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    
    # Create the collection
    collection = in_memory_chroma.get_or_create_collection(
        name=collection_name,
        embedding_function=None  # Let ChromaDB use its default embedding function
    )
    
    collection.add(
        ids=["id1", "id2", "id3"],
        documents=["Test document 1", "Test document 2", "Test document 3"],
        metadatas=[
            {
                "entry_id": "entry1",
                "title": "Entry 1",
                "chunk_index": 0,
                "tags": "test,example",
            },
            {
                "entry_id": "entry1",
                "title": "Entry 1",
                "chunk_index": 1,
                "tags": "test,example",
            },
            {
                "entry_id": "entry2",
                "title": "Entry 2",
                "chunk_index": 0,
                "tags": "example",
            },
        ],
        embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384]  # Use 384 dimensions to match ChromaDB's default
    )
    
    yield collection, collection_name
    
    try:
        in_memory_chroma.delete_collection(collection_name)
    except Exception as e:
        print(f"Error cleaning up collection: {e}")


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns deterministic embeddings."""
    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.generate_embedding.return_value = [0.1] * 384  # Match ChromaDB's default dimensions
    return mock_embedder


class TestSearchKnowledge:
    """Tests for the search_knowledge function"""

    def test_search_knowledge_basic(self, test_collection, mock_embedder):
        """Test basic search functionality using real in-memory ChromaDB"""
        collection, collection_name = test_collection
        
        with patch("app.tools.search.chroma_client.get_collection", return_value=collection), \
             patch("app.tools.search.Embedder", return_value=mock_embedder):
            
            # Run search
            results = search_knowledge("test query", top_k=2)

            # Verify results
            assert len(results) == 2
            assert "content" in results[0]
            assert "metadata" in results[0]
            assert "similarity_score" in results[0]
            assert "relevance_comment" in results[0]

            entry_ids = [result["metadata"]["entry_id"] for result in results]
            assert "entry1" in entry_ids
            assert "entry2" in entry_ids

    def test_search_knowledge_empty_results(self, in_memory_chroma, mock_embedder):
        """Test handling of empty search results using real in-memory ChromaDB"""
        # Create a unique empty collection
        empty_collection_name = f"empty_collection_{uuid.uuid4().hex[:8]}"
        empty_collection = in_memory_chroma.get_or_create_collection(
            name=empty_collection_name
        )
        
        with patch("app.tools.search.chroma_client.get_collection", return_value=empty_collection), \
             patch("app.tools.search.Embedder", return_value=mock_embedder):
            
            # Run search
            results = search_knowledge("non-existent query", top_k=5)

            # Verify appropriate response for no results
            assert len(results) == 1
            assert results[0]["content"] == ""
            assert results[0]["similarity_score"] == 0.0
            assert "No relevant content found" in results[0]["relevance_comment"]
        
        in_memory_chroma.delete_collection(empty_collection_name)

    def test_search_knowledge_error_handling(self, mock_embedder):
        """Test error handling during search"""
        # Create a mock client that raises an exception
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Test error")
        
        with patch("app.tools.search.chroma_client", mock_client), \
             patch("app.tools.search.Embedder", return_value=mock_embedder):
            
            # Run search
            results = search_knowledge("test query", top_k=2)

            # Should return empty list on error
            assert results == []

    def test_search_knowledge_relevance_categorization(self, in_memory_chroma, mock_embedder):
        """Test that search results include relevance categorization"""
        # Create a unique collection for this test
        relevance_collection_name = f"relevance_collection_{uuid.uuid4().hex[:8]}"
        relevance_collection = in_memory_chroma.get_or_create_collection(
            name=relevance_collection_name
        )
        
        # Add test data with simple embeddings
        relevance_collection.add(
            ids=["id1", "id2", "id3"],
            documents=["Test document 1", "Test document 2", "Test document 3"],
            metadatas=[
                {"entry_id": "entry1", "title": "Test Entry 1", "chunk_index": 0},
                {"entry_id": "entry2", "title": "Test Entry 2", "chunk_index": 0},
                {"entry_id": "entry3", "title": "Test Entry 3", "chunk_index": 0},
            ],
            embeddings=[
                [1.0] * 384,  # Simple test embedding
                [0.5] * 384,  # Simple test embedding
                [0.1] * 384,  # Simple test embedding
            ]
        )
        
        # Configure mock embedder to return a simple embedding
        mock_embedder.generate_embedding.return_value = [1.0] * 384
        
        with patch("app.tools.search.chroma_client.get_collection", return_value=relevance_collection), \
             patch("app.tools.search.Embedder", return_value=mock_embedder):
            
            # Run search
            results = search_knowledge("test query", top_k=3)
            
            assert len(results) > 0, "No search results returned"
            
            # Verify each result has the expected fields
            for result in results:
                assert "content" in result, "Result missing 'content' field"
                assert "metadata" in result, "Result missing 'metadata' field"
                assert "similarity_score" in result, "Result missing 'similarity_score' field"
                assert "relevance_comment" in result, "Result missing 'relevance_comment' field"
                
                # Verify relevance_comment is a non-empty string
                assert isinstance(result["relevance_comment"], str), "relevance_comment is not a string"
                assert len(result["relevance_comment"]) > 0, "relevance_comment is empty"
        
        in_memory_chroma.delete_collection(relevance_collection_name)


class TestGetEntryDetails:
    """Tests for the get_entry_details function"""

    def test_get_entry_details_found(self, in_memory_chroma):
        """Test successful retrieval of entry details using real in-memory ChromaDB"""
        # Create a unique collection for this test
        collection_name = f"entry_details_{uuid.uuid4().hex[:8]}"
        collection = in_memory_chroma.get_or_create_collection(name=collection_name)
        
        collection.add(
            ids=["chunk1", "chunk2", "chunk3"],
            documents=[
                "# Test Entry\n\nFirst part.",
                "Second part of the content.",
                "Final part of content.",
            ],
            metadatas=[
                {
                    "entry_id": "test123",
                    "title": "Test Entry",
                    "chunk_index": 0,
                    "tags": "test,example",
                    "last_modified_source": "2023-05-15",
                    "source_file": "test/path.md",
                },
                {
                    "entry_id": "test123",
                    "title": "Test Entry",
                    "chunk_index": 1,
                    "tags": "test,example",
                    "last_modified_source": "2023-05-15",
                    "source_file": "test/path.md",
                },
                {
                    "entry_id": "test123",
                    "title": "Test Entry",
                    "chunk_index": 2,
                    "tags": "test,example",
                    "last_modified_source": "2023-05-15",
                    "source_file": "test/path.md",
                },
            ],
            embeddings=[
                [0.1] * 384,
                [0.2] * 384,
                [0.3] * 384
            ]
        )
        
        # Patch the chroma_client.get_collection to use our test collection
        with patch("app.tools.search.chroma_client.get_collection", return_value=collection):
            # Run the function
            result = get_entry_details("test123")

            # Verify correct response
            assert result["entry_id"] == "test123"
            assert result["title"] == "Test Entry"
            assert "test" in result["tags"]
            assert result["last_modified"] == "2023-05-15"
            assert result["source_file"] == "test/path.md"

            # Content should be reconstructed in order
            expected_content = "\n".join(
                [
                    "# Test Entry\n\nFirst part.",
                    "Second part of the content.",
                    "Final part of content.",
                ]
            )
            assert result["content"] == expected_content
        
        in_memory_chroma.delete_collection(collection_name)

    def test_get_entry_details_not_found(self, in_memory_chroma):
        """Test handling of non-existent entries using real in-memory ChromaDB"""
        # Create a unique empty collection for this test
        collection_name = f"empty_details_{uuid.uuid4().hex[:8]}"
        collection = in_memory_chroma.get_or_create_collection(name=collection_name)
        
        # Patch the chroma_client.get_collection to use our test collection
        with patch("app.tools.search.chroma_client.get_collection", return_value=collection):
            # Run the function with a non-existent entry ID
            result = get_entry_details("nonexistent_id")

            # Verify error response
            assert "error" in result
            assert "not found" in result["error"]
        
        in_memory_chroma.delete_collection(collection_name)

    def test_get_entry_details_error_handling(self):
        """Test error handling during retrieval"""
        # Create a mock client that raises an exception
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Test error")
        
        # Patch the chroma_client
        with patch("app.tools.search.chroma_client", mock_client):
            # Run the function
            result = get_entry_details("test123")

            # Verify error response
            assert "error" in result
            assert "Test error" in result["error"]

    def test_get_entry_details_chunk_ordering(self, in_memory_chroma):
        """Test that chunks are properly ordered by chunk_index using real in-memory ChromaDB"""
        # Create a unique collection for this test
        collection_name = f"chunk_order_{uuid.uuid4().hex[:8]}"
        collection = in_memory_chroma.get_or_create_collection(name=collection_name)
        
        # Add test data with out-of-order chunks
        collection.add(
            ids=["chunk2", "chunk1", "chunk3"],
            documents=["Middle part.", "# Test Entry\n\nFirst part.", "Final part."],
            metadatas=[
                {"entry_id": "test123", "title": "Test Entry", "chunk_index": 1},
                {"entry_id": "test123", "title": "Test Entry", "chunk_index": 0},
                {"entry_id": "test123", "title": "Test Entry", "chunk_index": 2},
            ],
            embeddings=[
                [0.2] * 384,
                [0.1] * 384,
                [0.3] * 384
            ]
        )
        
        # Patch the chroma_client.get_collection to use our test collection
        with patch("app.tools.search.chroma_client.get_collection", return_value=collection):
            # Run the function
            result = get_entry_details("test123")

            # Content should be sorted by chunk_index, not by original order
            expected_content = "\n".join(
                ["# Test Entry\n\nFirst part.", "Middle part.", "Final part."]
            )
            assert result["content"] == expected_content
        
        in_memory_chroma.delete_collection(collection_name)


if __name__ == "__main__":
    pytest.main(["-xvs", "test_search.py"])
