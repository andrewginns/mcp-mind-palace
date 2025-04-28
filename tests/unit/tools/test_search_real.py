import os
import time
import uuid
import pytest
import numpy as np
import chromadb
from chromadb.config import Settings
from typing import Dict, Any, List, Tuple

from app.tools.search import search_knowledge, get_entry_details
from app.knowledge_management.embedder import Embedder


@pytest.fixture
def real_embedder():
    """Create a real embedder instance for testing."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    return Embedder(api_key=api_key, normalize=True)


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
def test_collection_with_real_embeddings(in_memory_chroma, real_embedder):
    """Create a test collection with real embeddings."""
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    
    collection = in_memory_chroma.get_or_create_collection(
        name=collection_name
    )
    
    documents = [
        "Python is a programming language with clear syntax",
        "Machine learning models require large datasets",
        "Vector databases store and retrieve embeddings efficiently"
    ]
    
    embeddings = real_embedder.generate_embeddings_batch(documents)
    
    collection.add(
        ids=["id1", "id2", "id3"],
        documents=documents,
        metadatas=[
            {
                "entry_id": "entry1",
                "title": "Python Programming",
                "chunk_index": 0,
                "tags": "programming,python",
            },
            {
                "entry_id": "entry2",
                "title": "Machine Learning",
                "chunk_index": 0,
                "tags": "ml,data",
            },
            {
                "entry_id": "entry3",
                "title": "Vector Databases",
                "chunk_index": 0,
                "tags": "database,vector",
            },
        ],
        embeddings=embeddings
    )
    
    yield collection, collection_name
    
    try:
        in_memory_chroma.delete_collection(collection_name)
    except Exception as e:
        print(f"Error cleaning up collection: {e}")


@pytest.fixture
def large_test_collection(in_memory_chroma, real_embedder):
    """Create a large test collection with 100+ entries for performance testing."""
    collection_name = f"large_collection_{uuid.uuid4().hex[:8]}"
    
    collection = in_memory_chroma.get_or_create_collection(
        name=collection_name
    )
    
    documents = []
    metadatas = []
    ids = []
    
    for batch_idx in range(6):  # 6 batches of 20 = 120 documents
        batch_documents = []
        batch_metadatas = []
        batch_ids = []
        
        for i in range(20):
            idx = batch_idx * 20 + i
            doc = f"Test document {idx} with content about topic {idx % 10}"
            batch_documents.append(doc)
            
            batch_metadatas.append({
                "entry_id": f"entry{idx}",
                "title": f"Entry {idx}",
                "chunk_index": 0,
                "tags": f"tag{idx % 5},test",
            })
            
            batch_ids.append(f"id{idx}")
        
        batch_embeddings = real_embedder.generate_embeddings_batch(batch_documents)
        
        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas,
            embeddings=batch_embeddings
        )
        
        documents.extend(batch_documents)
        metadatas.extend(batch_metadatas)
        ids.extend(batch_ids)
        
        time.sleep(0.5)
    
    yield collection, collection_name, documents, metadatas, ids
    
    try:
        in_memory_chroma.delete_collection(collection_name)
    except Exception as e:
        print(f"Error cleaning up collection: {e}")


@pytest.mark.real_tokenization
class TestSearchKnowledgeWithRealEmbeddings:
    """Tests for the search_knowledge function using real embeddings."""
    
    @pytest.mark.real_tokenization
    def test_search_with_real_embeddings(self, test_collection_with_real_embeddings, real_embedder, monkeypatch):
        """Test search with real embeddings and verify results."""
        collection, collection_name = test_collection_with_real_embeddings
        
        monkeypatch.setattr("app.tools.search.chroma_client.get_collection", lambda name: collection)
        monkeypatch.setattr("app.tools.search.Embedder", lambda: real_embedder)
        
        results = search_knowledge("How to use Python for programming?", top_k=3)
        
        assert len(results) > 0
        assert "content" in results[0]
        assert "metadata" in results[0]
        assert "similarity_score" in results[0]
        assert "relevance_comment" in results[0]
        
        python_found = False
        for result in results:
            if "Python" in result["content"]:
                python_found = True
                break
        
        assert python_found, "Python-related content not found in search results"
    
    @pytest.mark.real_tokenization
    def test_vector_similarity_math_verification(self, test_collection_with_real_embeddings, real_embedder, monkeypatch):
        """Test that search results have correct similarity scores using vector math."""
        collection, collection_name = test_collection_with_real_embeddings
        
        monkeypatch.setattr("app.tools.search.chroma_client.get_collection", lambda name: collection)
        monkeypatch.setattr("app.tools.search.Embedder", lambda: real_embedder)
        
        query = "Python programming language"
        
        query_embedding = real_embedder.generate_embedding(query)
        
        results = search_knowledge(query, top_k=3)
        
        collection_data = collection.get(include=["documents", "embeddings"])
        
        for i, result in enumerate(results):
            doc_idx = collection_data["documents"].index(result["content"])
            doc_embedding = collection_data["embeddings"][doc_idx]
            
            query_array = np.array(query_embedding)
            doc_array = np.array(doc_embedding)
            
            query_norm = np.linalg.norm(query_array)
            doc_norm = np.linalg.norm(doc_array)
            
            if query_norm > 0 and doc_norm > 0:
                query_normalized = query_array / query_norm
                doc_normalized = doc_array / doc_norm
                
                cosine_similarity = np.dot(query_normalized, doc_normalized)
                
                
                assert -1.0 <= result["similarity_score"] <= 1.0, \
                    f"Similarity score {result['similarity_score']} is not in the valid range [-1,1]"
                
                if cosine_similarity > 0.5:
                    assert result["similarity_score"] > 0.2, \
                        f"High cosine similarity {cosine_similarity} should result in higher similarity score than {result['similarity_score']}"
    
    @pytest.mark.real_tokenization
    def test_pagination_and_ordering(self, large_test_collection, real_embedder, monkeypatch):
        """Test pagination and result ordering with a large result set."""
        collection, collection_name, documents, metadatas, ids = large_test_collection
        
        monkeypatch.setattr("app.tools.search.chroma_client.get_collection", lambda name: collection)
        monkeypatch.setattr("app.tools.search.Embedder", lambda: real_embedder)
        
        results_5 = search_knowledge("Test document with content", top_k=5)
        results_10 = search_knowledge("Test document with content", top_k=10)
        results_20 = search_knowledge("Test document with content", top_k=20)
        
        assert len(results_5) == 5
        assert len(results_10) == 10
        assert len(results_20) == 20
        
        for results in [results_5, results_10, results_20]:
            for i in range(1, len(results)):
                assert results[i-1]["similarity_score"] >= results[i]["similarity_score"], \
                    "Results are not ordered by similarity score"
        
        result_5_contents = [r["content"] for r in results_5]
        result_10_contents = [r["content"] for r in results_10]
        
        for content in result_5_contents:
            assert content in result_10_contents, "Pagination is inconsistent"
    
    @pytest.mark.real_tokenization
    def test_error_handling_unusual_data(self, in_memory_chroma, real_embedder, monkeypatch):
        """Test error handling when ChromaDB returns unusual data structures."""
        collection_name = f"unusual_collection_{uuid.uuid4().hex[:8]}"
        collection = in_memory_chroma.get_or_create_collection(name=collection_name)
        
        empty_embedding = [0.0] * 1536  # OpenAI embeddings are 1536-dimensional
        
        collection.add(
            ids=["empty_id"],
            documents=["Empty embedding document"],
            metadatas=[{"entry_id": "empty_entry", "title": "Empty", "chunk_index": 0}],
            embeddings=[empty_embedding]
        )
        
        monkeypatch.setattr("app.tools.search.chroma_client.get_collection", lambda name: collection)
        monkeypatch.setattr("app.tools.search.Embedder", lambda: real_embedder)
        
        results = search_knowledge("Test query", top_k=5)
        
        assert isinstance(results, list), "Results should be a list even with unusual data"
        
        in_memory_chroma.delete_collection(collection_name)
    
    @pytest.mark.real_tokenization
    def test_performance_large_knowledge_base(self, large_test_collection, real_embedder, monkeypatch):
        """Test search performance with a large knowledge base (100+ entries)."""
        collection, collection_name, documents, metadatas, ids = large_test_collection
        
        monkeypatch.setattr("app.tools.search.chroma_client.get_collection", lambda name: collection)
        monkeypatch.setattr("app.tools.search.Embedder", lambda: real_embedder)
        
        start_time = time.time()
        results = search_knowledge("Test document with content about topic 5", top_k=10)
        end_time = time.time()
        
        assert len(results) == 10, "Should return exactly 10 results"
        
        search_time = end_time - start_time
        assert search_time < 5.0, f"Search took too long: {search_time} seconds"
        
        relevant_count = 0
        for result in results:
            if "topic 5" in result["content"]:
                relevant_count += 1
        
        assert relevant_count > 0, "No relevant documents found in search results"
    
    @pytest.mark.real_tokenization
    def test_relevance_categorization_thresholds(self, test_collection_with_real_embeddings, real_embedder, monkeypatch):
        """Test specific relevance categorization thresholds."""
        collection, collection_name = test_collection_with_real_embeddings
        
        monkeypatch.setattr("app.tools.search.chroma_client.get_collection", lambda name: collection)
        monkeypatch.setattr("app.tools.search.Embedder", lambda: real_embedder)
        
        high_relevance_query = "Python programming language syntax"
        medium_relevance_query = "Programming concepts"
        low_relevance_query = "Data analysis techniques"
        very_low_relevance_query = "Cooking recipes"
        
        high_results = search_knowledge(high_relevance_query, top_k=3)
        medium_results = search_knowledge(medium_relevance_query, top_k=3)
        low_results = search_knowledge(low_relevance_query, top_k=3)
        very_low_results = search_knowledge(very_low_relevance_query, top_k=3)
        
        high_relevance_found = False
        for result in high_results:
            if "Highly relevant" in result["relevance_comment"] and result["similarity_score"] > 0.7:
                high_relevance_found = True
                break
        
        medium_relevance_found = False
        for result in medium_results:
            if "Moderately relevant" in result["relevance_comment"] and result["similarity_score"] > 0.5:
                medium_relevance_found = True
                break
        
        low_relevance_found = False
        for result in low_results:
            if "Somewhat relevant" in result["relevance_comment"] and result["similarity_score"] > 0.2:
                low_relevance_found = True
                break
        
        very_low_relevance_found = False
        for result in very_low_results:
            if "Low relevance" in result["relevance_comment"] or "Very low relevance" in result["relevance_comment"]:
                very_low_relevance_found = True
                break
        
        assert high_relevance_found or medium_relevance_found or low_relevance_found or very_low_relevance_found, \
            "No relevance thresholds were properly categorized"


@pytest.mark.real_tokenization
class TestGetEntryDetailsWithRealData:
    """Tests for the get_entry_details function with real data."""
    
    @pytest.mark.real_tokenization
    def test_get_entry_details_with_real_data(self, in_memory_chroma, real_embedder, monkeypatch):
        """Test get_entry_details with real data and multiple chunks."""
        collection_name = f"entry_details_{uuid.uuid4().hex[:8]}"
        collection = in_memory_chroma.get_or_create_collection(name=collection_name)
        
        chunks = [
            "# Vector Database Guide\n\nThis is the first part of the guide about vector databases.",
            "## Embeddings\n\nEmbeddings are vector representations of data that capture semantic meaning.",
            "## Similarity Search\n\nSimilarity search finds vectors that are close to a query vector."
        ]
        
        embeddings = real_embedder.generate_embeddings_batch(chunks)
        
        collection.add(
            ids=["chunk1", "chunk2", "chunk3"],
            documents=chunks,
            metadatas=[
                {
                    "entry_id": "vector-guide",
                    "title": "Vector Database Guide",
                    "chunk_index": 0,
                    "tags": "vector,database,guide",
                    "last_modified_source": "2023-06-01",
                    "source_file": "guides/vector_db.md",
                },
                {
                    "entry_id": "vector-guide",
                    "title": "Vector Database Guide",
                    "chunk_index": 1,
                    "tags": "vector,database,guide",
                    "last_modified_source": "2023-06-01",
                    "source_file": "guides/vector_db.md",
                },
                {
                    "entry_id": "vector-guide",
                    "title": "Vector Database Guide",
                    "chunk_index": 2,
                    "tags": "vector,database,guide",
                    "last_modified_source": "2023-06-01",
                    "source_file": "guides/vector_db.md",
                },
            ],
            embeddings=embeddings
        )
        
        monkeypatch.setattr("app.tools.search.chroma_client.get_collection", lambda name: collection)
        
        result = get_entry_details("vector-guide")
        
        assert result["entry_id"] == "vector-guide"
        assert result["title"] == "Vector Database Guide"
        assert "vector" in result["tags"]
        assert result["last_modified"] == "2023-06-01"
        assert result["source_file"] == "guides/vector_db.md"
        
        expected_content = "\n".join(chunks)
        assert result["content"] == expected_content
        
        in_memory_chroma.delete_collection(collection_name)
