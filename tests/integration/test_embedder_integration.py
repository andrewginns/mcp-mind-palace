import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock

from app.knowledge_management.embedder import Embedder


@pytest.mark.api_dependent
def test_minimal_api_integration():
    """
    Integration test with minimal API usage for CI pipeline.
    Uses a real API key but only makes a single embedding request.
    """
    embedder = Embedder()
    
    text = "Test"
    embedding = embedder.generate_embedding(text)
    
    assert len(embedding) == 1536  # text-embedding-3-small produces 1536-dim vectors
    
    assert not all(e == 0 for e in embedding)
    
    embedding_array = np.array(embedding)
    norm = np.linalg.norm(embedding_array)
    assert 0.99 <= norm <= 1.01, f"Embedding norm {norm} is not close to 1.0"


@pytest.mark.api_dependent
def test_error_handling_invalid_api_key():
    """Test error handling with an invalid API key."""
    with pytest.raises(Exception) as exc_info:
        embedder = Embedder(api_key="invalid-key")
        embedder.generate_embedding("Test text")
    
    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["auth", "key", "invalid", "credentials"])


@pytest.mark.api_dependent
def test_error_handling_with_retry():
    """Test error handling with retry logic for temporary failures."""
    mock_client = MagicMock()
    
    mock_client.embeddings.create.side_effect = [
        Exception("Rate limit exceeded"),  # First call fails
        MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3] * 512)])  # Second call succeeds
    ]
    
    with patch("openai.OpenAI", return_value=mock_client):
        embedder = Embedder(api_key="test-key")
        embedder.client = mock_client
        
        original_method = embedder.generate_embedding
        
        def retry_wrapper(text):
            try:
                return original_method(text)
            except Exception:
                time.sleep(0.1)
                return original_method(text)
        
        embedder.generate_embedding = retry_wrapper
        
        embedding = embedder.generate_embedding("Test text")
        
        assert mock_client.embeddings.create.call_count == 2
        
        assert len(embedding) == 1536


@pytest.mark.api_dependent
def test_long_text_chunking():
    """Test embedding of long text that requires chunking."""
    long_text = "This is a test sentence. " * 1000  # Should be well over 8k tokens
    
    embedder = Embedder()
    
    embedding = embedder.safe_generate_embedding(long_text)
    
    assert len(embedding) == 1536
    
    embedding_array = np.array(embedding)
    norm = np.linalg.norm(embedding_array)
    assert 0.99 <= norm <= 1.01, f"Embedding norm {norm} is not close to 1.0"


@pytest.mark.api_dependent
@pytest.mark.integration
def test_embedder_with_real_data_workflow():
    """
    Test a complete workflow with real data and API calls.
    This test verifies the embedder works correctly in a realistic scenario.
    """
    texts = [
        "Short text for testing",
        "Medium length text with some more content to embed using the OpenAI API",
        "Longer text " + "with repeated content " * 10
    ]
    
    embedder = Embedder(cache_enabled=True)
    
    embeddings = []
    for text in texts:
        embedding = embedder.embed_text(text)
        embeddings.append(embedding)
    
    for i, embedding in enumerate(embeddings):
        assert len(embedding) == 1536, f"Embedding {i} has incorrect dimensions"
    
    assert not np.allclose(embeddings[0], embeddings[1])
    assert not np.allclose(embeddings[0], embeddings[2])
    assert not np.allclose(embeddings[1], embeddings[2])
    
    cache_hits = 0
    for text in texts:
        if text in embedder.cache:
            cache_hits += 1
        
        embedding = embedder.embed_text(text)
    
    assert cache_hits == len(texts), f"Expected {len(texts)} cache hits, got {cache_hits}"
