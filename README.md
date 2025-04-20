# Mind Palace Model Context Protocol (MCP) Server for Markdown-Based Knowledge Management

This is a Python-based Model Context Protocol (MCP) server that provides a persistent, searchable knowledge base for Large Language Models (LLMs). The knowledge base is maintained as a collection of human-readable Markdown files, with ChromaDB used as a vector database for efficient semantic search.

![Visual metaphor of a "mind palace" – a grand, ornate library.
](docs/images/mind_palace_hero.jpg)
## Features

- Store knowledge in human-readable Markdown files with YAML frontmatter
- Efficient semantic search using ChromaDB vector database
- MCP-compliant server using FastMCP
- LLM can search for relevant knowledge based on task description
- LLM can propose new knowledge entries or suggest updates to existing ones
- Human-in-the-loop workflow for knowledge management
- Guided knowledge management workflow for LLMs to ensure proper content organization

## Installation

1. Clone this repository
2. Install uv:
   ```
   curl -Lsf https://astral.sh/uv/install.sh | sh
   ```
3. Install dependencies using uv:
   ```
   cd mcp_mind_palace
   uv sync
   ```

4. Copy `.env.example` to `.env` and add your OpenAI API key:
   ```
   cp .env.example .env
   # Edit .env to add your API key
   ```

## Server Configuration

The MCP server port can be configured using environment variables:

1. Copy the `.env.example` file to `.env`
2. Edit the `.env` file to set the desired port:
   ```
   MCP_SERVER_PORT=8000
   ```

If the specified port is already in use, the server will automatically find and use the next available port.

## Usage

### Running the Server

With uv (recommended):
```
# Create and activate uv virtual environment
uv venv
source .venv/bin/activate

# Run with stdio transport (default)
uv run run_server.py

# Run with SSE transport
TRANSPORT=sse uv run run_server.py

# Explicitly specify stdio transport
TRANSPORT=stdio uv run run_server.py
```

Alternatively, use the provided run script:
```
./run_with_uv.sh
```

Without uv:
```
# Run with stdio transport (default)
python run_server.py

# Run with SSE transport
TRANSPORT=sse python run_server.py
```

### Using MCP Inspector

The MCP Inspector is a developer tool that helps debug and test MCP servers. It provides a visual interface to explore your server's available tools, resources, and prompts, and allows you to call them interactively.

To use MCP Inspector with Mind Palace:

```bash
# Install MCP Inspector (if not already installed)
npm install -g @modelcontextprotocol/inspector

# Run Mind Palace with the inspector
npx @modelcontextprotocol/inspector uv run run_server.py
```

Or use the provided Makefile:

```bash
make debug
```

The Inspector will open in your browser, allowing you to:
- Browse all available tools, resources, and prompts
- Test tool calls with custom parameters
- View response data in a structured format
- Debug issues with your MCP server implementation

This is especially useful during development to ensure your tools are working correctly before integrating with AI assistants.

### Using Docker

A Dockerfile is provided to easily containerize and run the server:

```bash
# Build the Docker image
docker build -t mind-palace .

# Run with default settings (SSE transport on port 8050)
docker run -p 8050:8050 mind-palace

# Run with custom port
docker build --build-arg PORT=9000 -t mind-palace .
docker run -p 9000:9000 mind-palace

# Override transport at runtime
docker run -p 8050:8050 -e TRANSPORT=stdio mind-palace
```

Environment variables that can be set:
- `TRANSPORT`: Set to `sse` or `stdio` (default: `stdio` when running locally, `sse` in Docker)
- `PORT`: The port to use for SSE transport (default: `8050`)
- `HOST`: The host to bind to (default: `0.0.0.0` in Docker, localhost otherwise)

### Adding to MCP Client Configuration

To use Mind Palace with an MCP-compatible client (like Claude Desktop), add it to your client's configuration:

```json
{
  "mcpServers": {
    "mind-palace": {
      "command": "uv",
      "args": ["run", "run_server.py"]
    }
  }
}
```

For Docker:

```json
{
  "mcpServers": {
    "mind-palace": {
      "command": "docker",
      "args": ["run", "--rm", "-p", "8050:8050", "mind-palace"]
    }
  }
}
```

### Agent Demo Script

A demonstration script is provided in `scripts/agent_mcp_demo.py` that shows how to use the Mind Palace MCP server with the pydantic-ai Agent library:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

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

async def main():
    async with agent.run_mcp_servers():
        result = await agent.run("Give me the mind palace entry for python type hints")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
```

To run this demo:

```bash
uv run scripts/agent_mcp_demo.py
```

### MCP Protocol Usage

MCP servers follow a specific protocol for client-server communication using stdio transport. Clients interact with the server through the MCP API.

#### Example client usage with MCP Python SDK:

```python
from mcp import ClientSession, StdioServerParameters

async def example_client():
    # Create server parameters for stdio connection
    server_params = StdioServerParameters(
        command="uv",  # Executable
        args=["run", "run_server.py"],  # Script to run
    )
    
    # Connect to the server
    async with mcp.client.stdio.stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # List available resources
            resources = await session.list_resources()
            print(f"Resources: {resources}")
            
            # List available tools
            tools = await session.list_tools()
            print(f"Tools: {tools}")
            
            # Call search_knowledge tool
            result = await session.call_tool(
                "search_knowledge", 
                arguments={"task_description": "Python type hints best practices"}
            )
            print(f"Search results: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_client())
```

### Knowledge Base Structure

Knowledge is stored in Markdown files with YAML frontmatter in the `knowledge_base` directory. Each file should have the following structure:

```markdown
---
entry_id: unique-identifier
title: Knowledge Entry Title
tags: [tag1, tag2, tag3]
created: 2024-08-15
last_modified: 2024-08-15
status: active
---

# Knowledge Entry Title

Content of the knowledge entry...
```

### MCP Resources and Tools

The server exposes the following MCP Resources and Tools:

#### Resources

- `knowledge://entries` - List all knowledge entries
- `knowledge://entry/{entry_id}` - Get a specific knowledge entry

#### Tools

- `search_knowledge(task_description: str, top_k: int = 5)` - Search for relevant knowledge
- `get_entry_details(entry_id: str)` - Get details of a specific entry
- `propose_new_knowledge(proposed_content: str, suggested_tags: List[str] = [], search_verification: Optional[str] = None)` - Propose a new knowledge entry. Intelligently handles content with or without frontmatter.
- `suggest_knowledge_update(entry_id: str, suggested_changes: str, existing_content_verified: bool = False)` - Suggest updates to an existing entry

#### Prompts

- `knowledge_management_workflow` - Provides guidance to LLMs on the proper process for managing knowledge

- `knowledge_retreival_workflow` - Provides guidance to LLMs on the proper process for retreiving knowledge

For a comprehensive analysis of how these tools work and interact with the system, see the [MCP Tools Analysis](docs/tools/mcp_tools_analysis.md) documentation.



### MCP Client Configuration

To add this MCP server to your client's mcp.json:

```json
{
  "mcpServers": {
    "mind-palace": {
      "command": "uv",
      "args": ["run", "run_server.py"]
    }
  }
}
```

## Development

### Synchronization

The server synchronizes Markdown files with ChromaDB at startup and when manually triggered. The synchronization process:

1. Scans the knowledge base directory for Markdown files
2. Computes content hashes to detect changes
3. Processes new and modified files (parsing, chunking, embedding)
4. Removes entries for deleted files
5. Updates the state file to track the current state

### Adding New Knowledge

To add new knowledge:

1. Create a new Markdown file in the `knowledge_base` directory
2. Include YAML frontmatter with required fields (entry_id, title, tags)
3. Add the content in Markdown format
4. Restart the server or trigger a manual sync

### Security

API keys for the embedding service are stored in the `.env` file, which should never be committed to version control. The `.gitignore` file is configured to exclude this file.

