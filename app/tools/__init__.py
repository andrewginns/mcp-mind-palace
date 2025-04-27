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

    tools = [
        (search_knowledge, "search_knowledge"),
        (get_entry_details, "get_entry_details"),
        (propose_new_knowledge, "propose_new_knowledge"),
        (suggest_knowledge_update, "suggest_knowledge_update"),
    ]

    for tool_func, tool_name in tools:
        try:
            mcp.tool()(tool_func)
            logger.info(f"Registered tool: {tool_name}")
        except Exception as e:
            logging.error(f"Failed to register tool {tool_name}: {str(e)}")

    logger.info("MCP Tools registration completed")
