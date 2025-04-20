import logging
import os
import re
import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import yaml

from app.config import KNOWLEDGE_BASE_PATH, ACTIVE_KNOWLEDGE_PATH
from app.tools.search import search_knowledge
from app.knowledge_management.markdown_parser import get_frontmatter

logger = logging.getLogger(__name__)


def extract_frontmatter(content: str) -> Tuple[Dict[str, Any], str, bool]:
    """
    Extract frontmatter from content if it exists.

    Args:
        content: Markdown content that might contain frontmatter

    Returns:
        Tuple of (frontmatter dict, remaining content, has_frontmatter)
    """
    # Check for frontmatter pattern (---\n...\n---)
    frontmatter_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match = frontmatter_pattern.match(content.strip())

    if match:
        try:
            # Parse the YAML frontmatter
            frontmatter_yaml = match.group(1)
            frontmatter = yaml.safe_load(frontmatter_yaml)

            # Get the remaining content after frontmatter
            remaining_content = content[match.end() :].strip()

            return frontmatter, remaining_content, True
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse frontmatter YAML: {e}")

    # No valid frontmatter found
    return {}, content, False


def propose_new_knowledge(
    proposed_content: str,
    suggested_tags: List[str] = [],
    search_verification: Optional[str] = None,
) -> str:
    """
    Propose adding a new knowledge entry.
    This tool saves the proposed content to a designated "review" area.
    Encourages verification that similar content doesn't already exist.

    The proposed content MUST contain frontmatter with at least the required fields:
    entry_id, title, tags, created, last_modified, and status.

    Args:
        proposed_content: Content of the proposed knowledge entry
        suggested_tags: List of tags for the proposed entry
        search_verification: Optional confirmation that a search was performed

    Returns:
        Confirmation message
    """
    try:
        # Extract frontmatter if present
        frontmatter, remaining_content, has_frontmatter = extract_frontmatter(
            proposed_content.strip()
        )

        # Check if frontmatter exists and contains required fields
        required_fields = [
            "entry_id",
            "title",
            "tags",
            "created",
            "last_modified",
            "status",
        ]
        if not has_frontmatter:
            return "Error: Frontmatter is required. Please include frontmatter with required fields: entry_id, title, tags, created, last_modified, and status."

        # Check for missing required fields
        missing_fields = [
            field for field in required_fields if field not in frontmatter
        ]
        if missing_fields:
            return f"Error: Frontmatter is missing required fields: {', '.join(missing_fields)}"

        # Get title from frontmatter
        title = frontmatter["title"]
        entry_id = frontmatter["entry_id"]

        # Create a meaningful filename from the title
        title_slug = "-".join(re.sub(r"[^a-z0-9\s-]", "", title.lower()).split()[:3])
        timestamp = int(time.time())

        # If no search verification provided, create a warning
        verification_note = ""
        if not search_verification:
            # Perform a quick search to check for similar content
            logger.info(
                f"No search verification provided, performing quick verification search for: {title}"
            )
            search_results = search_knowledge(title, top_k=3)

            if search_results and any(
                result.get("similarity_score", 0) > 0.75 for result in search_results
            ):
                verification_note = (
                    "\n\nWARNING: Similar content may already exist in the knowledge base. "
                    "Please verify using search_knowledge before creating new entries. "
                    "Consider using suggest_knowledge_update instead if appropriate."
                )

                # Add relevant entries to the warning
                verification_note += "\n\nPotentially related entries:"
                for result in search_results:
                    if result.get("similarity_score", 0) > 0.6:
                        result_entry_id = result.get("metadata", {}).get(
                            "entry_id", "unknown"
                        )
                        result_title = result.get("metadata", {}).get(
                            "title", "untitled"
                        )
                        verification_note += (
                            f"\n- {result_title} (ID: {result_entry_id})"
                        )

        # Determine the appropriate category based on tags
        category = "general"
        if frontmatter.get("tags"):
            for tag in frontmatter["tags"]:
                # Use the first tag that seems like a good category name
                if tag and len(tag) > 3 and "-" not in tag and "/" not in tag:
                    category = tag
                    break

        # Create proposals directory structure that mirrors active directory
        proposals_dir = os.path.join(KNOWLEDGE_BASE_PATH, "proposals", category)
        os.makedirs(proposals_dir, exist_ok=True)

        # Reconstruct content with the validated frontmatter
        final_content = proposed_content

        # Create the proposal file
        filename = f"{title_slug}-{timestamp}.md"
        filepath = os.path.join(proposals_dir, filename)

        with open(filepath, "w") as f:
            f.write(final_content)

        return f"New knowledge entry proposed: '{title}' (ID: {entry_id})\nSaved to: {filepath}{verification_note}"

    except Exception as e:
        logger.error(f"Error proposing new knowledge: {e}")
        return f"Error proposing new knowledge: {str(e)}"


def suggest_knowledge_update(
    entry_id: str,
    suggested_changes: str,
    existing_content_verified: bool = False,
) -> str:
    """
    Suggest updates to an existing knowledge entry.
    This tool logs the entry_id and the suggested_changes for that entry for review.
    Enforces verification that the entry_id is a valid entry in `knowledge_base/active` folder.

    Args:
        entry_id: Unique identifier of the knowledge entry to update
        suggested_changes: Description of the suggested changes
        existing_content_verified: Flag indicating whether existence was verified

    Returns:
        Confirmation message
    """
    try:
        # Verify the entry exists
        if not existing_content_verified:
            # Search for the entry in the active directory
            found = False
            target_category = None

            for root, _, files in os.walk(ACTIVE_KNOWLEDGE_PATH):
                for file in files:
                    if file.endswith(".md"):
                        file_path = os.path.join(root, file)
                        try:
                            metadata = get_frontmatter(file_path)
                            if metadata and metadata.get("entry_id") == entry_id:
                                found = True
                                # Get category from the directory structure
                                rel_path = os.path.relpath(
                                    os.path.dirname(file_path), ACTIVE_KNOWLEDGE_PATH
                                )
                                if rel_path == ".":
                                    target_category = "general"
                                else:
                                    target_category = rel_path
                                break
                        except Exception as e:
                            logger.error(f"Error checking entry existence: {e}")
                if found:
                    break

            if not found:
                return f"Error: Knowledge entry with ID '{entry_id}' not found. Please verify the entry exists before suggesting updates."

        # Create updates directory structure that mirrors active directory
        updates_dir = os.path.join(KNOWLEDGE_BASE_PATH, "updates")
        if target_category:
            updates_dir = os.path.join(updates_dir, target_category)
        os.makedirs(updates_dir, exist_ok=True)

        # Create a unique update ID
        timestamp = int(time.time())
        update_id = f"update-{entry_id}-{timestamp}"

        # Format the update metadata
        update_metadata = {
            "update_id": update_id,
            "entry_id": entry_id,
            "suggested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "pending",
        }

        # Create the update file
        update_content = "---\n"
        update_content += yaml.dump(update_metadata, default_flow_style=False)
        update_content += "---\n\n"
        update_content += f"# Suggested Update for Entry '{entry_id}'\n\n"
        update_content += suggested_changes

        filename = f"{update_id}.md"
        filepath = os.path.join(updates_dir, filename)

        with open(filepath, "w") as f:
            f.write(update_content)

        return (
            f"Update suggested for knowledge entry '{entry_id}'\nSaved to: {filepath}"
        )

    except Exception as e:
        logger.error(f"Error suggesting knowledge update: {e}")
        return f"Error suggesting knowledge update: {str(e)}"
