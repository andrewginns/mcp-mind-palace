import sys
import os
import logging
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.main import mcp

load_dotenv()

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting MCP server with stdio transport...")
    mcp.run(transport="stdio")
