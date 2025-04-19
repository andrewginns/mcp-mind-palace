# MCP Mind Palace: Comprehensive System Analysis

This document provides a detailed analysis of the MCP Mind Palace system architecture and components.

## MCP Tools Implementation

The repository implements four MCP tools that provide core functionality:

### 1. `search_knowledge` Tool
**Purpose**: Performs semantic search on the knowledge base using vector embeddings.

**Implementation**:
- Takes a task description and optional top_k parameter
- Generates an embedding for the query using OpenAI's API
- Queries ChromaDB for similar content using vector similarity
- Returns formatted results with content, metadata, and similarity scores

**Key Code**:
```python
def search_knowledge(task_description: str, top_k: int = 5) -> List[Dict[str, Any]]:
    collection = chroma_client.get_collection("knowledge_base")
    embedder = Embedder()
    query_embedding = embedder.generate_embedding(task_description)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format and return results
```

### 2. `get_entry_details` Tool
**Purpose**: Retrieves complete information for a specific knowledge entry.

**Implementation**:
- Takes an entry_id parameter
- Queries ChromaDB for all chunks with the matching entry_id
- Sorts chunks by chunk_index to reconstruct the original order
- Combines chunks to recreate the full content
- Returns structured metadata and content

**Key Code**:
```python
def get_entry_details(entry_id: str) -> Dict[str, Any]:
    collection = chroma_client.get_collection("knowledge_base")
    
    results = collection.get(
        where={"entry_id": entry_id},
        include=["documents", "metadatas"]
    )
    
    # Sort chunks by chunk_index
    sorted_indices = sorted(range(len(results["ids"])), 
                           key=lambda i: results["metadatas"][i].get("chunk_index", 0))
    
    # Reconstruct full content and return with metadata
```

### 3. `propose_new_knowledge` Tool
**Purpose**: Allows proposing new knowledge entries for review.

**Implementation**:
- Takes proposed_content and optional suggested_tags
- Extracts a title from the content
- Generates a unique entry_id
- Creates YAML frontmatter with metadata
- Saves the proposal as a Markdown file in the proposals directory

**Key Code**:
```python
def propose_new_knowledge(proposed_content: str, suggested_tags: List[str] = []) -> str:
    proposals_dir = os.path.join(KNOWLEDGE_BASE_PATH, "proposals")
    os.makedirs(proposals_dir, exist_ok=True)
    
    # Extract title and generate entry_id
    lines = proposed_content.strip().split('\n')
    title = lines[0].strip('# ') if lines and lines[0].startswith('#') else "Untitled Proposal"
    timestamp = int(time.time())
    entry_id = f"proposal-{'-'.join(title.lower().split()[:3])}-{timestamp}"
    
    # Create frontmatter and save file
```

### 4. `suggest_knowledge_update` Tool
**Purpose**: Allows suggesting updates to existing knowledge entries.

**Implementation**:
- Takes entry_id and suggested_changes parameters
- Generates a unique update_id
- Creates metadata for the update suggestion
- Saves the update suggestion as a Markdown file in the updates directory

**Key Code**:
```python
def suggest_knowledge_update(entry_id: str, suggested_changes: str) -> str:
    updates_dir = os.path.join(KNOWLEDGE_BASE_PATH, "updates")
    os.makedirs(updates_dir, exist_ok=True)
    
    # Generate update_id
    timestamp = int(time.time())
    update_id = f"update-{entry_id}-{timestamp}"
    
    # Create metadata and save file
```

## Embedding Process

The `Embedder` class handles vector embedding generation:

**Key Features**:
- Uses OpenAI's API to generate embeddings
- Supports both single and batch embedding generation
- Implements utility methods for generating IDs and content hashes
- Enables efficient change detection through content hashing

**Implementation**:
```python
class Embedder:
    def __init__(self, api_key: Optional[str] = None, model: str = EMBEDDING_MODEL):
        # Initialize with API key and model
        
    def generate_embedding(self, text: str) -> List[float]:
        # Generate embedding for a single text
        
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        # Generate embeddings for a batch of texts efficiently
```

## Knowledge Synchronization Mechanism

The `KnowledgeSync` class maintains synchronization between Markdown files and ChromaDB:

**Key Features**:
- Uses Watchdog to monitor file system events in real-time
- Processes file creation, modification, and deletion events
- Tracks file state using content hashes for efficient updates
- Chunks content and generates embeddings in batches
- Updates ChromaDB with new or modified content

**Implementation**:
```python
class KnowledgeSync:
    def __init__(self, knowledge_base_path, chroma_client, collection_name="knowledge_base", 
                 state_file_path=None, embedder=None, enable_watchdog=True,
                 embedding_timeout=30, batch_size=10):
        # Initialize with paths, client, and configuration
        
    def _setup_watchdog(self):
        # Set up watchdog observer for file monitoring
        
    def process_file_event(self, file_path, event_type):
        # Process file events (created, modified, deleted)
        
    def _process_file(self, file_path, content_hash):
        # Process a file: parse, chunk, embed, and update ChromaDB
```

## MCP Resources Implementation

The repository implements two MCP resources:

### 1. `knowledge://entries?tag={tag}` Resource
**Purpose**: Lists available knowledge entries, optionally filtered by tag.

**Implementation**:
```python
def list_knowledge_entries(tag: Optional[str] = None) -> List[Dict]:
    # Walk through knowledge base directory
    # Extract frontmatter from each file
    # Filter by tag if provided
    # Return list of entries with metadata
```

### 2. `knowledge://entry/{entry_id}` Resource
**Purpose**: Retrieves the complete original Markdown content for a specific entry.

**Implementation**:
```python
def get_knowledge_entry(entry_id: str) -> Dict:
    # Walk through knowledge base directory
    # Find file with matching entry_id
    # Extract content and metadata
    # Return complete entry details
```

## System Integration and Data Flow

The system follows a layered architecture with clear data flow:

1. **File System Changes → Knowledge Synchronization**:
   - Watchdog detects file system events
   - Events are debounced to prevent redundant processing
   - KnowledgeSync processes the file events

2. **Knowledge Synchronization → ChromaDB Updates**:
   - Markdown files are parsed to extract content and frontmatter
   - Content is chunked for efficient processing
   - Embeddings are generated for chunks
   - ChromaDB is updated with new or modified content

3. **ChromaDB → MCP Tools and Resources**:
   - MCP tools query ChromaDB for semantic search
   - MCP resources access the file system directly
   - Clients interact with the system through the MCP protocol

## Client Usage Example

Here's how a client would use the MCP Mind Palace:

```python
from mcp import ClientSession, StdioServerParameters

async def main():
    # Connect to the server
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "run_server.py"],
    )
    
    async with mcp.client.stdio.stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # Search for knowledge
            results = await session.call_tool(
                "search_knowledge", 
                arguments={"task_description": "Python type hints best practices"}
            )
            
            # Get details for a specific entry
            if results:
                entry_id = results[0]["metadata"]["entry_id"]
                details = await session.call_tool(
                    "get_entry_details", 
                    arguments={"entry_id": entry_id}
                )
                
            # Browse entries with a specific tag
            entries = await session.get_resource("knowledge://entries?tag=python")
            
            # Propose new knowledge
            await session.call_tool(
                "propose_new_knowledge", 
                arguments={
                    "proposed_content": "# New Knowledge Entry\n\nContent here...",
                    "suggested_tags": ["python", "tutorial"]
                }
            )
```

## Key Technical Insights

1. **Efficient Vector Search**: The system uses ChromaDB for efficient vector similarity search, enabling semantic understanding beyond keyword matching.

2. **Real-time Synchronization**: The Watchdog integration provides real-time monitoring of file changes, ensuring the knowledge base stays up-to-date.

3. **Batched Processing**: Embedding generation and ChromaDB updates are performed in batches for efficiency.

4. **Content Hashing**: SHA-256 hashes are used to detect content changes, avoiding unnecessary processing.

5. **Standardized Protocol**: The MCP protocol provides a standardized interface for clients to interact with the knowledge base.
