import logging

logger = logging.getLogger(__name__)

def register_resources(mcp):
    """
    Register all MCP resources.
    This function will be called from main.py.
    """
    from app.resources.knowledge_resources import list_knowledge_entries, get_knowledge_entry
    
    logger.info("Registering MCP Resources...")
    mcp.resource("knowledge://entries?tag={tag}")(list_knowledge_entries)
    mcp.resource("knowledge://entry/{entry_id}")(get_knowledge_entry)
    logger.info("MCP Resources registered successfully")
