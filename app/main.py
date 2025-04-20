import logging
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from app.config import ACTIVE_KNOWLEDGE_PATH, chroma_client
from app.resources import register_resources
from app.tools import register_tools
from app.prompts import register_prompts

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
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
        "watching_enabled": global_knowledge_sync.enable_watchdog
        if global_knowledge_sync
        else False,
        "collection_count": global_knowledge_sync.collection.count()
        if global_knowledge_sync
        else 0,
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
async def lifespan_context(server: FastMCP) -> AsyncIterator[None]:
    """
    Lifespan manager for the MCP server.
    Handles initialization and cleanup of resources.
    """
    global global_knowledge_sync

    logger.info("=== MCP SERVER LIFESPAN STARTING ===")

    yield

    knowledge_sync = None

    try:
        from app.knowledge_management.synchronization import KnowledgeSync

        logger.info("Performing post-initialization setup...")
        logger.info("Initializing ChromaDB collection...")
        collection = chroma_client.get_or_create_collection(name="knowledge_base")
        logger.info(
            f"ChromaDB collection 'knowledge_base' initialized with {collection.count()} entries"
        )

        logger.info("Creating KnowledgeSync instance...")
        knowledge_sync = KnowledgeSync(
            ACTIVE_KNOWLEDGE_PATH, chroma_client, embedding_timeout=30, batch_size=10
        )
        global_knowledge_sync = knowledge_sync

        logger.info("Starting knowledge synchronization...")
        knowledge_sync.start()
        logger.info(
            f"Knowledge synchronization started, watching {ACTIVE_KNOWLEDGE_PATH}"
        )

        logger.info("Starting initial synchronization in background thread...")
        sync_thread = threading.Thread(
            target=run_sync_in_background, args=(knowledge_sync,), daemon=True
        )
        sync_thread.start()

    except Exception as e:
        import traceback

        logger.error(f"Error during post-initialization setup: {e}")
        logger.error(traceback.format_exc())

    logger.info("=== MCP SERVER LIFESPAN CLEANUP ===")
    if knowledge_sync:
        knowledge_sync.stop()
        logger.info("Knowledge synchronization stopped.")

    logger.info("Server shutting down.")


mcp = FastMCP("MindPalaceServer", lifespan=lifespan_context)

try:
    register_resources(mcp)
    register_tools(mcp)
    register_prompts(mcp)
except Exception as e:
    import traceback

    logger.error(f"Error during resource or tool registration: {e}")
    logger.error(traceback.format_exc())
