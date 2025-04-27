from unittest.mock import MagicMock, patch
import pytest
from typing import Dict, Any

from app.tools.search import search_knowledge, get_entry_details
from app.knowledge_management.embedder import Embedder


class MockChromaCollection:
    """Mock for a ChromaDB collection that can be configured with test data"""

    def __init__(self, test_data: Dict[str, Any] = None):
        self.test_data = test_data or {
            "ids": [["id1", "id2", "id3"]],
            "documents": [["Test document 1", "Test document 2", "Test document 3"]],
            "metadatas": [
                [
                    {
                        "entry_id": "entry1",
                        "title": "Entry 1",
                        "chunk_index": 0,
                        "tags": ["test", "example"],
                    },
                    {
                        "entry_id": "entry1",
                        "title": "Entry 1",
                        "chunk_index": 1,
                        "tags": ["test", "example"],
                    },
                    {
                        "entry_id": "entry2",
                        "title": "Entry 2",
                        "chunk_index": 0,
                        "tags": ["example"],
                    },
                ]
            ],
            "distances": [
                [0.1, 0.3, 0.6]
            ],  # Closer to 0 is more similar for cosine distance
        }

    def query(self, query_embeddings, n_results, include):
        """Mock the query method to return test data"""
        return self.test_data

    def get(self, where=None, include=None):
        """Mock the get method with filtering capability"""
        if where and "entry_id" in where and "$eq" in where["entry_id"]:
            entry_id = where["entry_id"]["$eq"]

            # Filter results based on entry_id
            filtered_ids = []
            filtered_docs = []
            filtered_metadata = []

            for i, metadata in enumerate(self.test_data["metadatas"][0]):
                if metadata.get("entry_id") == entry_id:
                    filtered_ids.append(self.test_data["ids"][0][i])
                    filtered_docs.append(self.test_data["documents"][0][i])
                    filtered_metadata.append(metadata)

            # Return filtered data
            if filtered_ids:
                return {
                    "ids": filtered_ids,
                    "documents": filtered_docs,
                    "metadatas": filtered_metadata,
                }

        # Default empty response
        return {"ids": [], "documents": [], "metadatas": []}


class MockChromaClient:
    """Mock for ChromaDB client"""

    def __init__(self, collection_data: Dict[str, Any] = None):
        self.collections = {"knowledge_base": MockChromaCollection(collection_data)}

    def get_collection(self, name):
        """Return a mock collection"""
        return self.collections.get(name)


class MockEmbedder:
    """Mock for the Embedder class"""

    def generate_embedding(self, text):
        """Return a deterministic mock embedding"""
        # Create a simple mock embedding of the correct dimension
        return [0.1] * 1536  # OpenAI embeddings are 1536-dimensional


class TestSearchKnowledge:
    """Tests for the search_knowledge function"""

    @patch("app.tools.search.chroma_client")
    @patch("app.tools.search.Embedder")
    def test_search_knowledge_basic(self, mock_embedder_class, mock_chroma_client):
        """Test basic search functionality"""
        # Configure mocks
        mock_embedder = MockEmbedder()
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MockChromaCollection()
        mock_chroma_client.get_collection.return_value = mock_collection

        # Run search
        results = search_knowledge("test query", top_k=2)

        # Verify results
        assert len(results) == 2
        assert "content" in results[0]
        assert "metadata" in results[0]
        assert "similarity_score" in results[0]
        assert "relevance_comment" in results[0]

        # First result should be from entry1 (closest match)
        assert results[0]["metadata"]["entry_id"] == "entry1"

        # Second result should be from entry2
        assert results[1]["metadata"]["entry_id"] == "entry2"

    @patch("app.tools.search.chroma_client")
    @patch("app.tools.search.Embedder")
    def test_search_knowledge_empty_results(
        self, mock_embedder_class, mock_chroma_client
    ):
        """Test handling of empty search results"""
        # Configure empty results
        empty_data = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_embedder = MockEmbedder()
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MockChromaCollection(empty_data)
        mock_chroma_client.get_collection.return_value = mock_collection

        # Run search
        results = search_knowledge("non-existent query", top_k=5)

        # Verify appropriate response for no results
        assert len(results) == 1
        assert results[0]["content"] == ""
        assert results[0]["similarity_score"] == 0.0
        assert "No relevant content found" in results[0]["relevance_comment"]

    @patch("app.tools.search.chroma_client")
    @patch("app.tools.search.Embedder")
    def test_search_knowledge_error_handling(
        self, mock_embedder_class, mock_chroma_client
    ):
        """Test error handling during search"""
        # Configure mock to raise exception
        mock_embedder = MockEmbedder()
        mock_embedder_class.return_value = mock_embedder

        mock_chroma_client.get_collection.side_effect = Exception("Test error")

        # Run search
        results = search_knowledge("test query", top_k=2)

        # Should return empty list on error
        assert results == []

    @patch("app.tools.search.chroma_client")
    def test_search_knowledge_relevance_categorization(self, mock_chroma_client):
        """Test proper categorization of results by relevance"""
        # Configure custom results with different similarity scores
        custom_data = {
            "ids": [["id1", "id2", "id3", "id4", "id5"]],
            "documents": [["Doc1", "Doc2", "Doc3", "Doc4", "Doc5"]],
            "metadatas": [
                [
                    {"entry_id": "entry1", "title": "High Relevance", "chunk_index": 0},
                    {
                        "entry_id": "entry2",
                        "title": "Moderate Relevance",
                        "chunk_index": 0,
                    },
                    {
                        "entry_id": "entry3",
                        "title": "Somewhat Relevant",
                        "chunk_index": 0,
                    },
                    {"entry_id": "entry4", "title": "Low Relevance", "chunk_index": 0},
                    {
                        "entry_id": "entry5",
                        "title": "Very Low Relevance",
                        "chunk_index": 0,
                    },
                ]
            ],
            # Distances that will convert to different similarity levels
            # Similarity = 1.0 - distance
            "distances": [[0.1, 0.4, 0.7, 0.9, 0.99]],
        }

        mock_collection = MockChromaCollection(custom_data)
        mock_chroma_client.get_collection.return_value = mock_collection

        # Create a real embedder but with generate_embedding mocked
        with patch.object(Embedder, "generate_embedding", return_value=[0.1] * 1536):
            results = search_knowledge("test query", top_k=5)

        # Updated assertions to match actual output format
        assert any(
            "highly relevant" in result["relevance_comment"].lower()
            for result in results[:1]
        )
        assert any(
            "moderately relevant" in result["relevance_comment"].lower()
            for result in results[1:2]
        )
        assert any(
            "somewhat relevant" in result["relevance_comment"].lower()
            for result in results[2:3]
        )
        assert any(
            "low relevance" in result["relevance_comment"].lower()
            for result in results[3:5]
        )


class TestGetEntryDetails:
    """Tests for the get_entry_details function"""

    @patch("app.tools.search.chroma_client")
    def test_get_entry_details_found(self, mock_chroma_client):
        """Test successful retrieval of entry details"""
        # Configure mock with multi-chunk entry
        custom_data = {
            "ids": ["chunk1", "chunk2", "chunk3"],
            "documents": [
                "# Test Entry\n\nFirst part.",
                "Second part of the content.",
                "Final part of content.",
            ],
            "metadatas": [
                {
                    "entry_id": "test123",
                    "title": "Test Entry",
                    "chunk_index": 0,
                    "tags": ["test", "example"],
                    "last_modified_source": "2023-05-15",
                    "source_file": "test/path.md",
                },
                {
                    "entry_id": "test123",
                    "title": "Test Entry",
                    "chunk_index": 1,
                    "tags": ["test", "example"],
                    "last_modified_source": "2023-05-15",
                    "source_file": "test/path.md",
                },
                {
                    "entry_id": "test123",
                    "title": "Test Entry",
                    "chunk_index": 2,
                    "tags": ["test", "example"],
                    "last_modified_source": "2023-05-15",
                    "source_file": "test/path.md",
                },
            ],
        }

        mock_collection = MagicMock()
        mock_collection.get.return_value = custom_data
        mock_chroma_client.get_collection.return_value = mock_collection

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

    @patch("app.tools.search.chroma_client")
    def test_get_entry_details_not_found(self, mock_chroma_client):
        """Test handling of non-existent entries"""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_chroma_client.get_collection.return_value = mock_collection

        # Run the function
        result = get_entry_details("nonexistent_id")

        # Verify error response
        assert "error" in result
        assert "not found" in result["error"]

    @patch("app.tools.search.chroma_client")
    def test_get_entry_details_error_handling(self, mock_chroma_client):
        """Test error handling during retrieval"""
        mock_chroma_client.get_collection.side_effect = Exception("Test error")

        # Run the function
        result = get_entry_details("test123")

        # Verify error response
        assert "error" in result
        assert "Test error" in result["error"]

    @patch("app.tools.search.chroma_client")
    def test_get_entry_details_chunk_ordering(self, mock_chroma_client):
        """Test that chunks are properly ordered by chunk_index"""
        # Configure mock with out-of-order chunks
        custom_data = {
            "ids": ["chunk2", "chunk1", "chunk3"],
            "documents": ["Middle part.", "# Test Entry\n\nFirst part.", "Final part."],
            "metadatas": [
                {"entry_id": "test123", "title": "Test Entry", "chunk_index": 1},
                {"entry_id": "test123", "title": "Test Entry", "chunk_index": 0},
                {"entry_id": "test123", "title": "Test Entry", "chunk_index": 2},
            ],
        }

        mock_collection = MagicMock()
        mock_collection.get.return_value = custom_data
        mock_chroma_client.get_collection.return_value = mock_collection

        # Run the function
        result = get_entry_details("test123")

        # Content should be sorted by chunk_index, not by original order
        expected_content = "\n".join(
            ["# Test Entry\n\nFirst part.", "Middle part.", "Final part."]
        )
        assert result["content"] == expected_content


if __name__ == "__main__":
    pytest.main(["-xvs", "test_search.py"])
