import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def knowledge_management_workflow() -> Dict[str, Any]:
    """
    Prompt that guides the proper workflow for knowledge management.
    This prompt emphasizes the need to check existing knowledge before deciding
    whether to update existing entries or create new ones.

    Returns:
        Dictionary with prompt content
    """

    return {
        "name": "knowledge_management_workflow",
        "description": "Guidelines for managing knowledge entries properly",
        "content": """# Knowledge Management Workflow

When adding or updating knowledge to this knowledge base, follow this systematic workflow to ensure proper knowledge management:

## Step 1: Search Existing Knowledge
Before creating new content, always search the existing knowledge base to check if related information already exists:

1. Use the `search_knowledge` tool with a clear description of the topic
2. Review all results carefully to identify relevant entries
3. For promising entries, use `get_entry_details` to examine the full content

## Step 2: Decision Point
Based on your search results, decide the appropriate action:

- **If similar content exists that can be enhanced or corrected**: Suggest an update using `suggest_knowledge_update`
- **If no relevant content exists**: Create new knowledge using `propose_new_knowledge`
- **If partial information exists but a separate entry is justified**: Ensure clear cross-references to related entries

## Step 3: Updating Existing Knowledge
When suggesting updates to existing entries:

1. Retrieve the full entry using `get_entry_details` with the entry_id
2. Clearly describe what changes should be made and why
3. Use the `suggest_knowledge_update` tool, providing:
   - The exact entry_id of the knowledge to update
   - A comprehensive description of your suggested changes
   - Include how the changes improve or correct the existing content

## Step 4: Creating New Knowledge
When creating new knowledge:

1. Ensure the content is unique and not redundant with existing entries
2. Structure the content with clear headings and sections
3. Use the `propose_new_knowledge` tool, providing:
   - Well-formatted content (properly structured Markdown)
   - Relevant tags to categorize the knowledge
   - If related to existing entries, mention them for cross-reference

### Knowledge Base Structure

The knowledge base is organized in a specific directory structure:

- `active/`: Contains all approved knowledge entries
  - Subdirectories are organized by knowledge category (e.g., `python`, `javascript`, etc.)
- `proposals/`: Contains proposed new knowledge entries awaiting review
  - Mirror subdirectory structure of the `active` directory
- `updates/`: Contains suggested updates to existing knowledge entries
  - Mirror subdirectory structure of the `active` directory

When creating new knowledge or suggesting updates, the system will automatically place your contribution in the appropriate category directory within `proposals/` or `updates/`.

### Content Formatting Guidelines

When creating new content, you MUST include complete frontmatter and well-structured content:

```
---
entry_id: unique-entry-identifier
title: Title of Knowledge Entry
tags: [tag1, tag2, tag3]
created: YYYY-MM-DD
last_modified: YYYY-MM-DD
status: proposed
---

# Title of Knowledge Entry

An introductory paragraph that explains the topic clearly and concisely.

## Section 1

Content for section 1...

## Section 2

Content for section 2...

### Subsection (if needed)

More detailed information...

## Resources (if applicable)

- [Resource Name](URL)
- [Resource Name](URL)
```

**Required Frontmatter Fields**:
- `entry_id`: Unique identifier for the entry (use a descriptive kebab-case format)
- `title`: Clear, descriptive title matching the first heading
- `tags`: Array of relevant tags to categorize the knowledge
- `created`: Current date in YYYY-MM-DD format
- `last_modified`: Same as creation date for new entries, in YYYY-MM-DD format
- `status`: Always set to "proposed" for new entries

**Content Structure Requirements**:
1. Start with a level-1 heading (`#`) that matches the title in frontmatter
2. Include an introductory paragraph explaining the topic
3. Use level-2 headings (`##`) to organize the content into logical sections
4. Include code examples where appropriate, formatted in language-specific code blocks
5. For technical content, include a Resources section with relevant links

**Example Based on Existing Knowledge**:
Here's a simplified version of our Python Type Hints entry:

```
---
entry_id: python-type-hints
title: Using Type Hints in Python Code
tags: [python, type-hints, best-practice]
created: 2024-05-01
last_modified: 2024-05-01
status: proposed
---

# Using Type Hints in Python Code

Type hints are a powerful feature in Python that allows developers to specify the expected types of variables, function parameters, and return values.

## Benefits of Type Hints

- **Improved IDE Support**: Better code completion and error detection
- **Static Type Checking**: Catch errors before runtime
- **Self-Documenting Code**: Easier to understand function signatures

## Type Hinting Guidelines

### 1. Always Type Hint Function Signatures

```python
def calculate_price(base_price: float, discount: int) -> float:
    return base_price * (1 - discount/100)
```

## Resources

- [Python Type Checking Guide](https://mypy.readthedocs.io/)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
```

## Example Workflow
```
// First, search for relevant knowledge
results = search_knowledge("Python type hints best practices")

// If relevant entries found, get full details
if (results.length > 0) {
  details = get_entry_details(results[0].metadata.entry_id)
  
  // If entry needs updates
  if (needs_updates) {
    suggest_knowledge_update(
      details.entry_id,
      "The section on generic types should be updated to include recent PEP changes..."
    )
  }
} else {
  // If nothing relevant found, propose new knowledge
  propose_new_knowledge(
    "# Python Type Hints Best Practices\n\n## Introduction\n...",
    ["python", "type-hints", "best-practices"]
  )
}
```

Always remember that quality and organization are priorities for the knowledge base. Updates to existing content are preferred over creating duplicate information.""",
    }
