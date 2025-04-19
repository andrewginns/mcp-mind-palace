import logging

logger = logging.getLogger(__name__)


def register_tools(mcp):
    """
    Register all MCP tools.
    This function will be called from main.py.
    """
    from app.tools.proposals import propose_new_knowledge, suggest_knowledge_update
    from app.tools.search import get_entry_details, search_knowledge

    logger.info("Registering MCP Tools...")
    mcp.tool()(search_knowledge)
    mcp.tool()(get_entry_details)
    mcp.tool()(propose_new_knowledge)
    mcp.tool()(suggest_knowledge_update)
    logger.info("MCP Tools registered successfully")
