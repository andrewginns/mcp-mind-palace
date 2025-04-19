from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from app.config import chroma_client
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan_context(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan manager for the FastAPI app.
    Handles initialization and cleanup of resources.
    """
    logger.info("=== APP LIFESPAN STARTING ===")
    from app.knowledge_management.synchronization import KnowledgeSync
    from app.config import KNOWLEDGE_BASE_PATH
    
    knowledge_sync = None
    
    try:
        logger.info("Initializing ChromaDB collection...")
        collection = chroma_client.get_or_create_collection(name="knowledge_base")
        logger.info(f"ChromaDB collection 'knowledge_base' initialized with {collection.count()} entries")
        
        logger.info("Creating KnowledgeSync instance...")
        knowledge_sync = KnowledgeSync(KNOWLEDGE_BASE_PATH, chroma_client)
        logger.info("Starting knowledge synchronization...")
        knowledge_sync.start()
        logger.info(f"Knowledge synchronization started, watching {KNOWLEDGE_BASE_PATH}")
        
        logger.info("Performing initial synchronization of knowledge base...")
        knowledge_sync.sync()
        logger.info(f"Initial synchronization complete. ChromaDB now has {collection.count()} entries")
        
    except Exception as e:
        import traceback
        logger.error(f"Error during initialization: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("Server starting, all systems initialized.")
    
    yield
    
    logger.info("=== APP LIFESPAN CLEANUP ===")
    if knowledge_sync:
        knowledge_sync.stop()
        logger.info("Knowledge synchronization stopped.")
    
    logger.info("Server shutting down.")

app = FastAPI(lifespan=lifespan_context)

mcp = FastMCP("LocalKnowledgeServer")

from app.resources import register_resources
from app.tools import register_tools

try:
    register_resources(mcp)
    register_tools(mcp)
except Exception as e:
    import traceback
    logger.error(f"Error during resource or tool registration: {e}")
    logger.error(traceback.format_exc())

try:
    app.mount("/mcp", mcp.sse_app())
except Exception as e:
    import traceback
    logger.error(f"Error mounting MCP app: {e}")
    logger.error(traceback.format_exc())

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
