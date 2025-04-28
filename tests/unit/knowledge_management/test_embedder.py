import pytest
import uuid
import numpy as np
import time
from unittest.mock import patch, MagicMock

from app.knowledge_management.embedder import Embedder


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client that returns deterministic embeddings."""
    mock_client = MagicMock()

    # Mock response for single embedding request
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions

    mock_response = MagicMock()
    mock_response.data = [mock_embedding_data]

    # Mock client response
    mock_client.embeddings.create.return_value = mock_response

    return mock_client


@pytest.fixture
def embedder_with_mock_client(mock_openai_client):
    """Create an Embedder instance with a mock OpenAI client."""
    with patch("openai.OpenAI", return_value=mock_openai_client):
        embedder = Embedder(api_key="test-api-key")
        embedder.client = mock_openai_client
        yield embedder


def test_embedder_initialization():
    """Test Embedder initialization with valid API key."""
    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"}, clear=True),
        patch("openai.OpenAI"),
        patch("app.knowledge_management.embedder.OPENAI_API_KEY", "test-api-key"),
    ):
        embedder = Embedder()
        assert embedder.api_key == "test-api-key"
        assert embedder.model == "text-embedding-3-small"
        assert embedder.max_tokens == 8191
        assert embedder.cache_enabled is False


def test_embedder_initialization_no_api_key():
    """Test Embedder raises error with no API key."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("app.knowledge_management.embedder.OPENAI_API_KEY", None),
    ):
        with pytest.raises(ValueError) as exc_info:
            Embedder()
        assert "OpenAI API key not found" in str(exc_info.value)


def test_embedder_with_custom_params():
    """Test Embedder initialization with custom parameters."""
    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"}),
        patch("openai.OpenAI"),
    ):
        embedder = Embedder(
            api_key="custom-api-key",
            model="custom-model",
            max_tokens=4000,
            cache_enabled=True,
            normalize=True,
        )

        assert embedder.api_key == "custom-api-key"
        assert embedder.model == "custom-model"
        assert embedder.max_tokens == 4000
        assert embedder.cache_enabled is True
        assert embedder.normalize is True


def test_generate_embedding(embedder_with_mock_client, mock_openai_client):
    """Test generating embedding for a single text."""
    text = "This is a test text"
    embedding = embedder_with_mock_client.generate_embedding(text)

    # Check client was called correctly
    mock_openai_client.embeddings.create.assert_called_once_with(
        model=embedder_with_mock_client.model, input=text
    )

    # Check the result
    assert len(embedding) == 1536  # Standard size for text-embedding-3-small
    assert embedding[0] == 0.1
    assert embedding[1] == 0.2
    assert embedding[2] == 0.3


def test_embed_text_with_caching(embedder_with_mock_client, mock_openai_client):
    """Test embedding with caching enabled."""
    # Enable caching
    embedder_with_mock_client.cache_enabled = True

    # First call should use the API
    text = "Cache test"
    first_embedding = embedder_with_mock_client.embed_text(text)

    # Reset the mock to track new calls
    mock_openai_client.embeddings.create.reset_mock()

    # Second call should use the cache
    second_embedding = embedder_with_mock_client.embed_text(text)

    # Check API wasn't called the second time
    mock_openai_client.embeddings.create.assert_not_called()

    # Both embeddings should be the same
    assert first_embedding == second_embedding


def test_cache_hit_miss_ratio_validation():
    """Test cache hit/miss ratio validation with multiple texts."""
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3] * 512)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        embedder = Embedder(api_key="test-api-key", cache_enabled=True)
        embedder.client = mock_client
        
        texts = [
            "First unique text",
            "Second unique text",
            "Third unique text",
            "First unique text",  # Duplicate of first text (should be a cache hit)
            "Fourth unique text",
            "Second unique text",  # Duplicate of second text (should be a cache hit)
            "First unique text",   # Another duplicate (should be a cache hit)
        ]
        
        expected_api_calls = 0
        expected_cache_hits = 0
        
        for text in texts:
            if text not in embedder.cache:
                expected_api_calls += 1
            else:
                expected_cache_hits += 1
            
            # Generate embedding
            embedder.embed_text(text)
        
        assert mock_client.embeddings.create.call_count == expected_api_calls, \
            f"Expected {expected_api_calls} API calls, got {mock_client.embeddings.create.call_count}"
        
        unique_texts = set(texts)
        assert len(embedder.cache) == len(unique_texts), \
            f"Expected {len(unique_texts)} cache entries, got {len(embedder.cache)}"
        
        # Calculate and verify hit ratio
        total_requests = len(texts)
        expected_hit_ratio = expected_cache_hits / total_requests
        assert expected_hit_ratio == 3/7, \
            f"Expected hit ratio of 3/7, got {expected_hit_ratio}"
        
        # Verify all unique texts are in the cache
        for text in unique_texts:
            assert text in embedder.cache, f"Text '{text}' not found in cache"


def test_cache_performance_with_large_dataset():
    """Test cache performance with a larger dataset."""
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3] * 512)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        embedder = Embedder(api_key="test-api-key", cache_enabled=True)
        embedder.client = mock_client
        
        # Generate a larger dataset with controlled duplication
        base_texts = [f"Base text {i}" for i in range(20)]
        
        # Create a dataset with 50% duplication rate
        all_texts = []
        for i in range(100):
            if i % 2 == 0:
                all_texts.append(base_texts[i % len(base_texts)])
            else:
                all_texts.append(f"Unique text {i}")
        
        start_time = time.time()
        for text in all_texts:
            embedder.embed_text(text)
        end_time = time.time()
        
        # Calculate statistics
        unique_texts = set(all_texts)
        expected_api_calls = len(unique_texts)
        expected_cache_hits = len(all_texts) - expected_api_calls
        
        assert mock_client.embeddings.create.call_count == expected_api_calls, \
            f"Expected {expected_api_calls} API calls, got {mock_client.embeddings.create.call_count}"
        
        assert len(embedder.cache) == len(unique_texts), \
            f"Expected {len(unique_texts)} cache entries, got {len(embedder.cache)}"
        
        # Calculate and log performance metrics
        processing_time = end_time - start_time
        api_call_ratio = mock_client.embeddings.create.call_count / len(all_texts)
        cache_hit_ratio = 1 - api_call_ratio
        
        print(f"Processing time: {processing_time:.4f} seconds")
        print(f"Cache hit ratio: {cache_hit_ratio:.2%}")
        print(f"API call ratio: {api_call_ratio:.2%}")
        
        assert 0.35 <= cache_hit_ratio <= 0.55, \
            f"Expected cache hit ratio between 35-55%, got {cache_hit_ratio:.2%}"


def test_embed_batch(embedder_with_mock_client, mock_openai_client):
    """Test generating embeddings for a batch of texts."""
    # Modify the mock to return batch results
    mock_embedding_data1 = MagicMock()
    mock_embedding_data1.embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions

    mock_embedding_data2 = MagicMock()
    mock_embedding_data2.embedding = [0.4, 0.5, 0.6] * 512  # 1536 dimensions

    mock_response = MagicMock()
    mock_response.data = [mock_embedding_data1, mock_embedding_data2]

    mock_openai_client.embeddings.create.return_value = mock_response

    # Test batch embedding
    texts = ["First text", "Second text"]
    embeddings = embedder_with_mock_client.embed_batch(texts)

    # Check client was called correctly
    mock_openai_client.embeddings.create.assert_called_once_with(
        model=embedder_with_mock_client.model, input=texts
    )

    # Should return a list of embeddings
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1536
    assert len(embeddings[1]) == 1536


def test_normalize_embedding(embedder_with_mock_client):
    """Test normalizing embeddings to unit length."""
    embedder_with_mock_client.normalize = True

    # Create a non-normalized vector
    non_normalized = [3.0, 4.0]  # Length is 5

    # Use private method to normalize
    normalized = embedder_with_mock_client._normalize_embedding(non_normalized)

    # Check normalization (should be [0.6, 0.8] for a [3, 4] vector)
    assert pytest.approx(normalized[0]) == 0.6
    assert pytest.approx(normalized[1]) == 0.8

    # Check the length is 1
    length = np.sqrt(normalized[0] ** 2 + normalized[1] ** 2)
    assert pytest.approx(length) == 1.0


def test_normalize_embedding_mathematical_validation():
    """Test vector normalization with comprehensive mathematical validation."""
    with patch("openai.OpenAI"):
        embedder = Embedder(api_key="test-api-key", normalize=True)
        
        test_vectors = [
            [1.0, 0.0, 0.0],  # Unit vector along x-axis
            [0.0, 2.0, 0.0],  # Non-unit vector along y-axis
            [0.0, 0.0, -3.0],  # Negative non-unit vector along z-axis
            [1.0, 1.0, 1.0],  # Equal components
            [5.0, -5.0, 10.0],  # Mixed signs
            [0.1, 0.2, 0.3, 0.4, 0.5],  # Higher dimension
            [1e-8, 1e-8, 1e-8],  # Very small values
            [1e8, 1e8, 1e8],  # Very large values
        ]
        
        for vector in test_vectors:
            normalized = embedder._normalize_embedding(vector)
            
            vector_np = np.array(vector)
            normalized_np = np.array(normalized)
            
            # Calculate the expected normalized vector
            expected_norm = np.linalg.norm(vector_np)
            if expected_norm > 0:
                expected_normalized = vector_np / expected_norm
            else:
                expected_normalized = vector_np
            
            # Check that the normalized vector has unit length (or zero if input was zero)
            actual_norm = np.linalg.norm(normalized_np)
            if np.linalg.norm(vector_np) > 0:
                assert pytest.approx(actual_norm) == 1.0, f"Vector {vector} did not normalize to unit length"
            else:
                assert actual_norm == 0.0, f"Zero vector should remain zero"
            
            # Check that the normalized vector is parallel to the original
            if np.linalg.norm(vector_np) > 0:
                normalized_original = vector_np / np.linalg.norm(vector_np)
                dot_product = np.abs(np.dot(normalized_np, normalized_original))
                assert pytest.approx(dot_product) == 1.0, f"Normalized vector not parallel to original"
            
            # Check that each component matches the expected normalized vector
            for i in range(len(vector)):
                assert pytest.approx(normalized[i]) == expected_normalized[i], \
                    f"Component {i} of {vector} normalized incorrectly"


def test_normalize_embedding_with_zero_vector():
    """Test normalizing a zero vector."""
    with patch("openai.OpenAI"):
        embedder = Embedder(api_key="test-api-key", normalize=True)
        
        zero_vector = [0.0, 0.0, 0.0]
        normalized = embedder._normalize_embedding(zero_vector)
        
        assert normalized == zero_vector, "Zero vector should remain unchanged"


def test_generate_content_hash():
    """Test generating content hash."""
    content = "Test content"
    hash1 = Embedder.generate_content_hash(content)

    # Same content should produce the same hash
    hash2 = Embedder.generate_content_hash(content)
    assert hash1 == hash2

    # Different content should produce different hash
    different_hash = Embedder.generate_content_hash("Different content")
    assert hash1 != different_hash


def test_generate_chunk_id():
    """Test generating chunk ID."""
    entry_id = "test-entry"
    chunk_index = 5

    chunk_id = Embedder.generate_chunk_id(entry_id, chunk_index)

    # Check format
    assert chunk_id == f"{entry_id}-chunk-{chunk_index}"


def test_generate_uuid_from_content():
    """Test generating UUID from content."""
    content = "Test content"

    # Generate UUID
    uuid1 = Embedder.generate_uuid_from_content(content)

    # Check it's a valid UUID
    try:
        uuid.UUID(uuid1)
    except ValueError:
        pytest.fail("Not a valid UUID")

    # Same content should produce the same UUID
    uuid2 = Embedder.generate_uuid_from_content(content)
    assert uuid1 == uuid2

    # Different content should produce different UUID
    different_uuid = Embedder.generate_uuid_from_content("Different content")
    assert uuid1 != different_uuid
