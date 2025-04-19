# Mind Palace Model Context Protocol (MCP) Server for Markdown-Based Knowledge Management

This is a Python-based Model Context Protocol (MCP) server that provides a persistent, searchable knowledge base for Large Language Models (LLMs). The knowledge base is maintained as a collection of human-readable Markdown files, with ChromaDB used as a vector database for efficient semantic search.

## Features

- Store knowledge in human-readable Markdown files with YAML frontmatter
- Efficient semantic search using ChromaDB vector database
- MCP-compliant server using FastMCP
- LLM can search for relevant knowledge based on task description
- LLM can propose new knowledge entries or suggest updates to existing ones
- Human-in-the-loop workflow for knowledge management

## Installation

1. Clone this repository
2. Install uv:
   ```
   curl -Lsf https://astral.sh/uv/install.sh | sh
   ```
3. Install dependencies using uv:
   ```
   cd mcp_server
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

# For development with MCP Inspector
python -m mcp dev app/main.py

# For Claude Desktop integration
python -m mcp install app/main.py

# As a standalone server
python run_server.py
```

Alternatively, use the provided run script:
```
./run_with_uv.sh
```

Without uv:
```
# For development with MCP Inspector
mcp dev app/main.py

# For Claude Desktop integration
mcp install app/main.py

# As a standalone server
python run_server.py
```

### MCP Protocol Usage

MCP servers follow a specific protocol for client-server communication. Clients interact with the server by sending POST requests to the `/mcp/messages` endpoint with specific JSON payloads.

#### Example client usage:

```python
import asyncio
import httpx
import uuid

async def example_client():
    session_id = str(uuid.uuid4())
    base_url = "http://localhost:8082/mcp/messages"
    
    async with httpx.AsyncClient() as client:
        # List available resources
        response = await client.post(
            f"{base_url}/",
            params={"session_id": session_id},
            json={"type": "list_resources_request"}
        )
        print(f"Resources: {response.json()}")
        
        # List available tools
        response = await client.post(
            f"{base_url}/",
            params={"session_id": session_id},
            json={"type": "list_tools_request"}
        )
        print(f"Tools: {response.json()}")
        
        # Call search_knowledge tool
        response = await client.post(
            f"{base_url}/",
            params={"session_id": session_id},
            json={
                "type": "call_tool_request",
                "tool_name": "search_knowledge",
                "arguments": {
                    "task_description": "Python type hints best practices"
                }
            }
        )
        print(f"Search results: {response.json()}")

if __name__ == "__main__":
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
- `propose_new_knowledge(proposed_content: str, suggested_tags: List[str] = [])` - Propose a new knowledge entry
- `suggest_knowledge_update(entry_id: str, suggested_changes: str)` - Suggest updates to an existing entry

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

## License

This project is open source and available under the MIT License.
