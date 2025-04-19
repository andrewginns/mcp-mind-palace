import os
from typing import List, Dict, Any, Optional
import hashlib
import uuid
import logging
import openai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

class Embedder:
    """
    Class for generating embeddings using OpenAI's API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = EMBEDDING_MODEL):
        """
        Initialize the Embedder with API key and model.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: OpenAI embedding model to use
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        openai.api_key = self.api_key
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding
        """
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        try:
            response = openai.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings batch: {e}")
            raise
    
    @staticmethod
    def generate_chunk_id(entry_id: str, chunk_index: int) -> str:
        """
        Generate a deterministic ID for a chunk based on entry_id and chunk_index.
        
        Args:
            entry_id: Unique identifier of the knowledge entry
            chunk_index: Index of the chunk within the entry
            
        Returns:
            String ID for the chunk
        """
        return f"{entry_id}_chunk_{chunk_index}"
    
    @staticmethod
    def generate_content_hash(content: str) -> str:
        """
        Generate a hash of the content for change detection.
        
        Args:
            content: Content to hash
            
        Returns:
            SHA-256 hash of the content
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
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
