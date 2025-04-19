import logging
import os
import time
from typing import List

import yaml

from app.config import KNOWLEDGE_BASE_PATH

logger = logging.getLogger(__name__)


def propose_new_knowledge(proposed_content: str, suggested_tags: List[str] = []) -> str:
    """
    Allows LLM to suggest a new knowledge entry.
    Saves the proposed content to a designated "review" area.

    Args:
        proposed_content: Content of the proposed knowledge entry
        suggested_tags: List of tags for the proposed entry

    Returns:
        Confirmation message
    """
    try:
        proposals_dir = os.path.join(KNOWLEDGE_BASE_PATH, "proposals")
        os.makedirs(proposals_dir, exist_ok=True)

        lines = proposed_content.strip().split("\n")
        title = (
            lines[0].strip("# ")
            if lines and lines[0].startswith("#")
            else "Untitled Proposal"
        )

        timestamp = int(time.time())
        entry_id = f"proposal-{'-'.join(title.lower().split()[:3])}-{timestamp}"

        frontmatter = {
            "entry_id": entry_id,
            "title": title,
            "tags": suggested_tags or [],
            "created": time.strftime("%Y-%m-%d"),
            "status": "proposed",
        }

        frontmatter_yaml = yaml.dump(frontmatter, default_flow_style=False)

        full_content = f"---\n{frontmatter_yaml}---\n\n{proposed_content}"

        file_path = os.path.join(proposals_dir, f"{entry_id}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        return f"Proposal submitted for review. Saved as {file_path}"

    except Exception as e:
        logger.error(f"Error saving proposal: {e}")
        return f"Error saving proposal: {str(e)}"


def suggest_knowledge_update(entry_id: str, suggested_changes: str) -> str:
    """
    Allows LLM to suggest modifications to an existing knowledge entry.
    Logs the entry_id and suggested_changes for review.

    Args:
        entry_id: Unique identifier of the knowledge entry to update
        suggested_changes: Description of the suggested changes

    Returns:
        Confirmation message
    """
    try:
        updates_dir = os.path.join(KNOWLEDGE_BASE_PATH, "updates")
        os.makedirs(updates_dir, exist_ok=True)

        timestamp = int(time.time())
        update_id = f"update-{entry_id}-{timestamp}"

        update_metadata = {
            "entry_id": entry_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "proposed",
        }

        metadata_yaml = yaml.dump(update_metadata, default_flow_style=False)

        full_content = f"---\n{metadata_yaml}---\n\n# Suggested Update for {entry_id}\n\n{suggested_changes}"

        file_path = os.path.join(updates_dir, f"{update_id}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        return f"Update suggestion for {entry_id} submitted for review. Saved as {file_path}"

    except Exception as e:
        logger.error(f"Error saving update suggestion: {e}")
        return f"Error saving update suggestion: {str(e)}"
