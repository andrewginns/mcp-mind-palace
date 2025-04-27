import asyncio
import logging
import sys

import logfire
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

# Configure logging to file only
logging.getLogger().handlers = []
logging.basicConfig(
    filename="mind-palace.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Remove console handlers
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        logging.getLogger().removeHandler(handler)

# Configure logfire without console output
logfire.configure(send_to_logfire="if-token-present", console=False)
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

logger = logging.getLogger(__name__)

server = MCPServerStdio(
    command="uv",
    args=[
        "--directory",
        ".",
        "run",
        "run_server.py",
        "stdio",
    ],
)
agent = Agent("openai:gpt-4.1", mcp_servers=[server])
Agent.instrument_all()


async def main():
    async with agent.run_mcp_servers():
        result = await agent.run("Give me the mind palace entry for python type hints")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
