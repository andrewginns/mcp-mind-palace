import os

import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".chroma"
)
KNOWLEDGE_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base"
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
