import hashlib
import logging
import os
import uuid
import numpy as np
import tiktoken
from typing import List, Optional, Union, Tuple

import openai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
MAX_TOKENS = 8191  # Maximum token limit for text-embedding-3-small


class Embedder:
    """
    Class for generating embeddings using OpenAI's API.

    Optimized for text-embedding-3-small model with support for handling texts
    that exceed the model's maximum token limit.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
        max_tokens: int = MAX_TOKENS,
        encoding_name: str = "cl100k_base",
        cache_enabled: bool = False,
        normalize: bool = False,
    ):
        """
        Initialize the Embedder with API key and model.

        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: OpenAI embedding model to use
            max_tokens: Maximum token limit for the model
            encoding_name: Tokenizer encoding to use
            cache_enabled: Whether to enable caching of embeddings
            normalize: Whether to normalize embeddings to unit length
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

        self.model = model
        self.embedding_model = model  # For backward compatibility with tests
        self.max_tokens = max_tokens
        self.encoding_name = encoding_name
        self.cache_enabled = cache_enabled
        self.normalize = normalize
        self.cache = {}

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text. Alias for generate_embedding.
        Supports caching if enabled.

        Args:
            text: Text to generate embedding for

        Returns:
            List of floats representing the embedding
        """
        # Check cache if enabled
        if self.cache_enabled and text in self.cache:
            return self.cache[text]

        try:
            # Generate embedding
            embedding = self.generate_embedding(text)

            # Cache the result if caching is enabled
            if self.cache_enabled:
                self.cache[text] = embedding

            return embedding
        except Exception as e:
            logger.error(f"Error in embed_text: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts. Alias for generate_embeddings_batch.

        Args:
            texts: Texts to generate embeddings for

        Returns:
            List of embeddings
        """
        return self.generate_embeddings_batch(texts)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        Note: This method will fail if text exceeds model's token limit.
        Use safe_generate_embedding for longer texts.

        Args:
            text: Text to generate embedding for

        Returns:
            List of floats representing the embedding
        """
        try:
            # Check token count
            token_count = len(self.tokenizer.encode(text))
            if token_count > self.max_tokens:
                logger.warning(
                    f"Text exceeds token limit ({token_count} > {self.max_tokens}). "
                    "Consider using safe_generate_embedding instead."
                )

            response = self.client.embeddings.create(model=self.model, input=text)
            embedding = response.data[0].embedding

            # Normalize if required
            if self.normalize:
                embedding = self._normalize_embedding(embedding)

            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        Note: This method will fail if any text exceeds model's token limit.
        Use safe_generate_embeddings_batch for longer texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        try:
            # Check token counts
            for i, text in enumerate(texts):
                token_count = len(self.tokenizer.encode(text))
                if token_count > self.max_tokens:
                    logger.warning(
                        f"Text at index {i} exceeds token limit ({token_count} > {self.max_tokens}). "
                        "Consider using safe_generate_embeddings_batch instead."
                    )

            response = self.client.embeddings.create(model=self.model, input=texts)
            embeddings = [item.embedding for item in response.data]

            # Normalize if required
            if self.normalize:
                embeddings = [self._normalize_embedding(emb) for emb in embeddings]

            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings batch: {e}")
            raise

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize an embedding to unit length.

        Args:
            embedding: The embedding to normalize

        Returns:
            Normalized embedding
        """
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            normalized = embedding_array / norm
            return normalized.tolist()
        return embedding

    def safe_generate_embedding(self, text: str, chunk_size: int = 1000) -> List[float]:
        """
        Safely generate embedding for text of any length by chunking and averaging.

        For text exceeding the model's token limit, this method splits the text into chunks,
        generates embeddings for each chunk, and returns a weighted average.

        Args:
            text: Text to generate embedding for (can be any length)
            chunk_size: Chunk size in tokens for processing long text

        Returns:
            List of floats representing the averaged embedding
        """
        tokens = self.tokenizer.encode(text)

        # If text is within token limit, use standard embedding
        if len(tokens) <= self.max_tokens:
            return self.generate_embedding(text)

        # Otherwise, process in chunks
        embeddings = []
        chunk_lengths = []

        # Process token chunks
        for i in range(0, len(tokens), chunk_size):
            end_idx = min(i + chunk_size, len(tokens))
            chunk_tokens = tokens[i:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            try:
                chunk_embedding = self.generate_embedding(chunk_text)
                embeddings.append(chunk_embedding)
                chunk_lengths.append(len(chunk_tokens))
            except Exception as e:
                logger.error(f"Error embedding chunk {i // chunk_size}: {e}")

        if not embeddings:
            raise ValueError("Failed to generate any embeddings for the text")

        # Calculate weighted average based on chunk lengths
        embeddings_array = np.array(embeddings)
        weights = np.array(chunk_lengths) / sum(chunk_lengths)

        # Weighted average of embeddings
        avg_embedding = np.average(embeddings_array, axis=0, weights=weights)

        # Normalize the embedding to unit length
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        return avg_embedding.tolist()

    def safe_generate_embeddings_batch(
        self, texts: List[str], chunk_size: int = 1000
    ) -> List[List[float]]:
        """
        Safely generate embeddings for a batch of texts of any length.

        For texts exceeding the model's token limit, this method processes each text
        using the safe_generate_embedding method.

        Args:
            texts: List of texts to generate embeddings for (can be any length)
            chunk_size: Chunk size in tokens for processing long texts

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        embeddings = []

        for text in texts:
            embedding = self.safe_generate_embedding(text, chunk_size)
            embeddings.append(embedding)

        return embeddings

    @staticmethod
    def generate_chunk_id(entry_id: str, chunk_index: int) -> str:
        """
        Generate a unique ID for a chunk.

        Args:
            entry_id: The ID of the knowledge entry
            chunk_index: The index of the chunk

        Returns:
            A unique ID for the chunk
        """
        return f"{entry_id}-chunk-{chunk_index}"

    @staticmethod
    def generate_content_hash(content: str) -> str:
        """
        Generate a hash of the content for change detection.

        Args:
            content: Content to hash

        Returns:
            SHA-256 hash of the content
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def generate_uuid_from_content(content: str, namespace=uuid.NAMESPACE_URL) -> str:
        """
        Generate a UUID5 based on content for deduplication.

        Args:
            content: Content to generate UUID from
            namespace: UUID namespace to use

        Returns:
            UUID string
        """
        return str(uuid.uuid5(namespace, content))
