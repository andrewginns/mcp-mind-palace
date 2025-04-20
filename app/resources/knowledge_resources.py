import logging
import os
from typing import Dict, List, Optional

import markdown

from app.config import ACTIVE_KNOWLEDGE_PATH
from app.knowledge_management.markdown_parser import (
    get_frontmatter,
    parse_markdown_file,
)

logger = logging.getLogger(__name__)


def list_knowledge_entries(tag: Optional[str] = None) -> List[Dict]:
    """
    Returns a list of available knowledge entries, including their entry_id, title, and tags.
    Optionally filters by a specific tag.

    Args:
        tag: Optional tag to filter entries by

    Returns:
        List of dictionaries containing entry_id, title, and tags for each knowledge entry
    """

    results = []

    for root, _, files in os.walk(ACTIVE_KNOWLEDGE_PATH):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    metadata = get_frontmatter(file_path)
                    if metadata and "entry_id" in metadata and "title" in metadata:
                        if tag is None or (
                            "tags" in metadata
                            and isinstance(metadata["tags"], list)
                            and tag in metadata["tags"]
                        ):
                            results.append(
                                {
                                    "entry_id": metadata["entry_id"],
                                    "title": metadata["title"],
                                    "tags": metadata.get("tags", []),
                                }
                            )
                except Exception as e:
                    logger.error(f"Error parsing {file_path}: {e}")

    return results


def get_knowledge_entry(entry_id: str) -> Dict:
    """
    Retrieves the full, original Markdown content and associated metadata
    for a specific knowledge entry identified by its entry_id.

    Args:
        entry_id: The unique identifier of the knowledge entry

    Returns:
        Dictionary containing the entry's metadata and content
    """
    for root, _, files in os.walk(ACTIVE_KNOWLEDGE_PATH):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    metadata = get_frontmatter(file_path)
                    if metadata and metadata.get("entry_id") == entry_id:
                        content, _ = parse_markdown_file(file_path)
                        return {
                            "metadata": metadata,
                            "content": content,
                            "html": markdown.markdown(content),
                        }
                except Exception as e:
                    logger.error(f"Error parsing {file_path}: {e}")

    return {"error": f"Knowledge entry with ID '{entry_id}' not found"}
