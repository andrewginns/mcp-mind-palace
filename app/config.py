import os

import chromadb
from dotenv import load_dotenv

load_dotenv()

# Default paths relative to the config file location
DEFAULT_CHROMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".chroma"
)
DEFAULT_KNOWLEDGE_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base"
)
DEFAULT_ACTIVE_KNOWLEDGE_PATH = os.path.join(DEFAULT_KNOWLEDGE_BASE_PATH, "active")

# Use environment variables if set, otherwise use defaults
CHROMA_PATH = os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH)
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", DEFAULT_KNOWLEDGE_BASE_PATH)
ACTIVE_KNOWLEDGE_PATH = os.getenv(
    "ACTIVE_KNOWLEDGE_PATH", DEFAULT_ACTIVE_KNOWLEDGE_PATH
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
