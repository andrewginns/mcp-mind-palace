import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def knowledge_retrieval_workflow() -> Dict[str, Any]:
    """
    Prompt that guides the proper workflow for knowledge retrieval.

    Returns:
        Dictionary with prompt content
    """

    return {
        "name": "knowledge_retrieval_workflow",
        "description": "Guidelines for retrieving knowledge from the knowledge base",
        "content": """# Knowledge Retrieval Workflow

When retrieving knowledge from the knowledge base, follow this systematic workflow to ensure proper retrieval:

## Step 1: Search Existing Knowledge
Before retrieving knowledge, always search the existing knowledge base to check if related information already exists:

1. Use the `search_knowledge` tool with a clear description of the topic
2. Review all results carefully to identify relevant entries
3. For promising entries, use `get_entry_details` to examine the full content

## Step 2: Retrieve Knowledge
Based on your search results, decide the appropriate action:

- **If similar content exists**: Use the `get_entry_details` tool to retrieve the full content
- **If no relevant content exists**: Inform the user that no relevant content was found
""",
    }
