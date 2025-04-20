import logging

logger = logging.getLogger(__name__)


def register_prompts(mcp):
    """
    Register all MCP prompts.
    This function will be called from main.py.
    """
    from app.prompts.knowledge_management_prompt import knowledge_management_workflow

    logger.info("Registering MCP Prompts...")
    mcp.prompt()(knowledge_management_workflow)
    logger.info("MCP Prompts registered successfully")
