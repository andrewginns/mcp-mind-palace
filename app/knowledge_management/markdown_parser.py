import logging
import os
import re
from typing import Any, Dict, Optional, Tuple, List

import yaml

logger = logging.getLogger(__name__)

# Define required metadata fields
REQUIRED_FIELDS = ["entry_id", "title", "tags", "created", "last_modified", "status"]


def get_frontmatter(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract YAML frontmatter from a Markdown file.

    Args:
        file_path: Path to the Markdown file

    Returns:
        Dictionary containing the frontmatter metadata or None if no frontmatter is found
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.search(frontmatter_pattern, content, re.DOTALL)

    if match:
        try:
            frontmatter_yaml = match.group(1)
            return yaml.safe_load(frontmatter_yaml)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing frontmatter in {file_path}: {e}")
            return None

    return None


def parse_markdown_file(file_path: str) -> Dict[str, Any]:
    """
    Parse a Markdown file, extracting both content and frontmatter.

    Args:
        file_path: Path to the Markdown file

    Returns:
        Dictionary containing:
        - metadata: Dictionary containing the frontmatter metadata
        - content: The Markdown content without the frontmatter

    Raises:
        FileNotFoundError: If the file does not exist
        Exception: If frontmatter is missing or invalid, or if required fields are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.search(frontmatter_pattern, content, re.DOTALL)

    if not match:
        raise Exception(f"No frontmatter found in {file_path}")

    try:
        frontmatter_yaml = match.group(1)
        frontmatter = yaml.safe_load(frontmatter_yaml)

        # Validate required fields
        missing_fields = [
            field for field in REQUIRED_FIELDS if field not in frontmatter
        ]
        if missing_fields:
            raise Exception(
                f"Missing required fields in frontmatter: {', '.join(missing_fields)}"
            )

        content_without_frontmatter = re.sub(frontmatter_pattern, "", content, 1)

        return {"metadata": frontmatter, "content": content_without_frontmatter}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing frontmatter in {file_path}: {e}")
        raise Exception(f"Invalid YAML in frontmatter: {e}")
