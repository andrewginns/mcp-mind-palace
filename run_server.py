import logging
import os
import sys
import asyncio

from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.main import mcp

load_dotenv()

logger = logging.getLogger(__name__)


async def main():
    transport = os.getenv("TRANSPORT", "stdio")
    logger.info(f"Starting MCP server with {transport} transport...")

    if transport.lower() == "sse":
        # Run the MCP server with SSE transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
