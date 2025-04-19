import os
import re
import yaml
from typing import Dict, Tuple, Optional, Any

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
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.search(frontmatter_pattern, content, re.DOTALL)
    
    if match:
        try:
            frontmatter_yaml = match.group(1)
            return yaml.safe_load(frontmatter_yaml)
        except yaml.YAMLError as e:
            print(f"Error parsing frontmatter in {file_path}: {e}")
            return None
    
    return None

def parse_markdown_file(file_path: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Parse a Markdown file, extracting both content and frontmatter.
    
    Args:
        file_path: Path to the Markdown file
        
    Returns:
        Tuple containing (content, frontmatter)
        - content: The Markdown content without the frontmatter
        - frontmatter: Dictionary containing the frontmatter metadata or None if no frontmatter is found
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.search(frontmatter_pattern, content, re.DOTALL)
    
    if match:
        try:
            frontmatter_yaml = match.group(1)
            frontmatter = yaml.safe_load(frontmatter_yaml)
            
            content_without_frontmatter = re.sub(frontmatter_pattern, '', content, 1)
            return content_without_frontmatter, frontmatter
        except yaml.YAMLError as e:
            print(f"Error parsing frontmatter in {file_path}: {e}")
            return content, None
    
    return content, None
