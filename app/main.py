from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from app.config import chroma_client
import logging
import sys
import threading
import time

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

global_knowledge_sync = None
initialization_complete = False

def get_initialization_status():
    """
    Get the current initialization status.
    
    Returns:
        Dictionary with initialization status information
    """
    global global_knowledge_sync, initialization_complete
    
    return {
        "complete": initialization_complete,
        "knowledge_sync_initialized": global_knowledge_sync is not None,
        "watching_enabled": global_knowledge_sync.enable_watchdog if global_knowledge_sync else False,
        "collection_count": global_knowledge_sync.collection.count() if global_knowledge_sync else 0
    }

def run_sync_in_background(knowledge_sync):
    """
    Run the knowledge base synchronization in a background thread.
    
    Args:
        knowledge_sync: KnowledgeSync instance
    """
    global initialization_complete
    
    try:
        logger.info("Starting background knowledge base synchronization...")
        knowledge_sync.sync()
        logger.info("Background knowledge base synchronization completed.")
    except Exception as e:
        import traceback
        logger.error(f"Error during background synchronization: {e}")
        logger.error(traceback.format_exc())
    finally:
        initialization_complete = True
        logger.info("Initialization marked as complete.")

@asynccontextmanager
async def lifespan_context(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan manager for the FastAPI app.
    Handles initialization and cleanup of resources.
    """
    global global_knowledge_sync
    
    logger.info("=== APP LIFESPAN STARTING ===")
    
    logger.info("Server starting, yielding control to allow fast MCP initialization...")
    yield
    
    knowledge_sync = None
    
    try:
        from app.knowledge_management.synchronization import KnowledgeSync
        from app.config import KNOWLEDGE_BASE_PATH
        
        logger.info("Performing post-initialization setup...")
        logger.info("Initializing ChromaDB collection...")
        collection = chroma_client.get_or_create_collection(name="knowledge_base")
        logger.info(f"ChromaDB collection 'knowledge_base' initialized with {collection.count()} entries")
        
        logger.info("Creating KnowledgeSync instance...")
        knowledge_sync = KnowledgeSync(
            KNOWLEDGE_BASE_PATH, 
            chroma_client,
            embedding_timeout=30,
            batch_size=10
        )
        global_knowledge_sync = knowledge_sync
        
        logger.info("Starting knowledge synchronization...")
        knowledge_sync.start()
        logger.info(f"Knowledge synchronization started, watching {KNOWLEDGE_BASE_PATH}")
        
        logger.info("Starting initial synchronization in background thread...")
        sync_thread = threading.Thread(
            target=run_sync_in_background,
            args=(knowledge_sync,),
            daemon=True
        )
        sync_thread.start()
        
    except Exception as e:
        import traceback
        logger.error(f"Error during post-initialization setup: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("=== APP LIFESPAN CLEANUP ===")
    if knowledge_sync:
        knowledge_sync.stop()
        logger.info("Knowledge synchronization stopped.")
    
    logger.info("Server shutting down.")

app = FastAPI(lifespan=lifespan_context)

@app.on_event("startup")
async def startup_event():
    """
    Handle startup events for the FastAPI app.
    Log server startup information.
    """
    logger.info("FastAPI server startup event triggered")

@app.get("/status")
async def get_status():
    """
    Get the status of the server, including knowledge base synchronization status.
    """
    global global_knowledge_sync
    
    status = {
        "server": "running",
        "initialization": "complete" if global_knowledge_sync is not None else "in_progress",
        "knowledge_sync": {
            "initialized": global_knowledge_sync is not None,
            "watching_enabled": global_knowledge_sync.enable_watchdog if global_knowledge_sync else False,
            "collection_count": global_knowledge_sync.collection.count() if global_knowledge_sync else 0,
            "sync_status": global_knowledge_sync.get_sync_status() if global_knowledge_sync else {"is_syncing": False}
        }
    }
    
    return status

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring server status.
    """
    global global_knowledge_sync, initialization_complete
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "initialization_complete": initialization_complete,
        "knowledge_sync_available": global_knowledge_sync is not None,
        "uptime": "unknown",  # Could be enhanced with actual uptime tracking
        "port_conflict_resolution": "automatic_port_selection_enabled"
    }

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
    import os
    import socket
    
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    preferred_port = int(os.getenv("MCP_SERVER_PORT", "8080"))
    
    logger.info(f"Starting server on {host}:{preferred_port}...")
    
    try:
        uvicorn.run(app, host=host, port=preferred_port)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
