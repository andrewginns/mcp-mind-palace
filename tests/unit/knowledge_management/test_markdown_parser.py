import os
import pytest
from app.knowledge_management.markdown_parser import (
    get_frontmatter,
    parse_markdown_file,
)


def test_get_frontmatter_valid(sample_markdown_file):
    """Test extracting frontmatter from a valid markdown file."""
    frontmatter = get_frontmatter(sample_markdown_file)

    assert frontmatter is not None
    assert frontmatter["entry_id"] == "test123"
    assert frontmatter["title"] == "Test Entry"
    assert frontmatter["tags"] == ["test", "sample", "python"]
    assert frontmatter["status"] == "active"


def test_get_frontmatter_nonexistent_file():
    """Test get_frontmatter with a nonexistent file."""
    with pytest.raises(FileNotFoundError):
        get_frontmatter("nonexistent_file.md")


def test_get_frontmatter_no_frontmatter(temp_knowledge_dir):
    """Test get_frontmatter with a file that has no frontmatter."""
    # Create a file without frontmatter
    file_path = os.path.join(temp_knowledge_dir, "no_frontmatter.md")
    with open(file_path, "w") as f:
        f.write("# Just a title\n\nNo frontmatter here.")

    frontmatter = get_frontmatter(file_path)
    assert frontmatter is None


def test_get_frontmatter_invalid_yaml(temp_knowledge_dir):
    """Test get_frontmatter with invalid YAML in frontmatter."""
    # Create a file with invalid YAML frontmatter
    file_path = os.path.join(temp_knowledge_dir, "invalid_frontmatter.md")
    with open(file_path, "w") as f:
        f.write("""---
title: This is invalid: yaml
---

# Content
""")

    frontmatter = get_frontmatter(file_path)
    assert frontmatter is None


def test_parse_markdown_file_valid(sample_markdown_file):
    """Test parsing a valid markdown file."""
    result = parse_markdown_file(sample_markdown_file)

    assert "metadata" in result
    assert "content" in result
    assert result["metadata"]["title"] == "Test Entry"
    assert "# Test Entry" in result["content"]
    assert "## Section 1" in result["content"]


def test_parse_markdown_file_nonexistent():
    """Test parsing a nonexistent file."""
    with pytest.raises(FileNotFoundError):
        parse_markdown_file("nonexistent_file.md")


def test_parse_markdown_file_no_frontmatter(temp_knowledge_dir):
    """Test parsing a file without frontmatter."""
    # Create a file without frontmatter
    file_path = os.path.join(temp_knowledge_dir, "no_frontmatter.md")
    with open(file_path, "w") as f:
        f.write("# Just a title\n\nNo frontmatter here.")

    with pytest.raises(Exception) as exc_info:
        parse_markdown_file(file_path)

    assert "No frontmatter found" in str(exc_info.value)


def test_parse_markdown_file_missing_required_fields(temp_knowledge_dir):
    """Test parsing a file with missing required fields in frontmatter."""
    # Create a file with incomplete frontmatter
    file_path = os.path.join(temp_knowledge_dir, "incomplete_frontmatter.md")
    with open(file_path, "w") as f:
        f.write("""---
title: Just a title
tags: [test]
---

# Content
""")

    with pytest.raises(Exception) as exc_info:
        parse_markdown_file(file_path)

    # Check that the error message mentions missing fields
    for field in ["entry_id", "created", "last_modified", "status"]:
        assert field in str(exc_info.value)


def test_parse_markdown_file_invalid_yaml(temp_knowledge_dir):
    """Test parsing a file with invalid YAML in frontmatter."""
    # Create a file with invalid YAML frontmatter
    file_path = os.path.join(temp_knowledge_dir, "invalid_frontmatter.md")
    with open(file_path, "w") as f:
        f.write("""---
title: This is invalid: yaml
---

# Content
""")

    with pytest.raises(Exception) as exc_info:
        parse_markdown_file(file_path)

    assert "Invalid YAML in frontmatter" in str(exc_info.value)
