import os
import tempfile
import pytest
import shutil
import time
from unittest.mock import patch

import chromadb

from app.tools.proposals import (
    propose_new_knowledge,
    suggest_knowledge_update,
    extract_frontmatter,
)
from app.knowledge_management.synchronization import KnowledgeSync
from app.knowledge_management.embedder import Embedder


@pytest.fixture
def temp_knowledge_base():
    """Create a temporary directory structure for knowledge base testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        active_dir = os.path.join(temp_dir, "active")
        review_dir = os.path.join(temp_dir, "review")
        proposals_dir = os.path.join(review_dir, "proposals")
        updates_dir = os.path.join(review_dir, "updates")
        
        os.makedirs(active_dir, exist_ok=True)
        os.makedirs(proposals_dir, exist_ok=True)
        os.makedirs(updates_dir, exist_ok=True)
        
        categories = ["general", "python", "testing"]
        for category in categories:
            os.makedirs(os.path.join(active_dir, category), exist_ok=True)
        
        sample_entry_path = os.path.join(active_dir, "general", "sample_entry.md")
        with open(sample_entry_path, "w") as f:
            f.write("""---
entry_id: sample123
title: Sample Knowledge Entry
tags: [sample, test, general]
created: 2023-05-15
last_modified: 2023-05-15
status: active
---


This is a sample knowledge entry for testing.
""")
        
        yield {
            "base_dir": temp_dir,
            "active_dir": active_dir,
            "review_dir": review_dir,
            "proposals_dir": proposals_dir,
            "updates_dir": updates_dir,
            "sample_entry_path": sample_entry_path,
        }


@pytest.fixture
def real_chroma_client():
    """Create a real ChromaDB client with in-memory storage."""
    client = chromadb.Client()
    return client


@pytest.fixture
def real_embedder():
    """Create a real Embedder instance."""
    return Embedder(normalize=True)


@pytest.fixture
def real_knowledge_sync(real_chroma_client, real_embedder, temp_knowledge_base):
    """Create a KnowledgeSync instance with real dependencies but watchdog disabled."""
    knowledge_sync = KnowledgeSync(
        knowledge_base_path=temp_knowledge_base["active_dir"],
        chroma_client=real_chroma_client,
        embedder=real_embedder,
        enable_watchdog=False,  # Disable watchdog to control file events manually
        embedding_timeout=10,
        batch_size=5,
    )
    
    yield knowledge_sync
    
    knowledge_sync.stop()


@pytest.mark.integration
class TestProposalsWithRealFileOperations:
    """Tests for proposals functionality with real file operations."""
    
    def test_propose_new_knowledge_with_real_files(self, temp_knowledge_base):
        """Test proposing a new knowledge entry with real file operations."""
        with patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH", temp_knowledge_base["review_dir"]):
            content = """---
entry_id: test123
title: Test Knowledge Entry
tags: [test, example, python]
created: 2023-05-15
last_modified: 2023-05-15
status: draft
---


This is a test entry for knowledge management.
"""
            
            result = propose_new_knowledge(content)
            
            assert "New knowledge entry proposed" in result
            assert "Test Knowledge Entry" in result
            assert "test123" in result
            
            proposals_base_dir = os.path.join(temp_knowledge_base["review_dir"], "proposals")
            
            found_files = []
            for root, _, files in os.walk(proposals_base_dir):
                for file in files:
                    if file.endswith(".md"):
                        found_files.append(os.path.join(root, file))
            
            assert len(found_files) > 0, "No markdown files found in proposals directory"
            
            found_content = False
            for file_path in found_files:
                with open(file_path, "r") as f:
                    file_content = f.read()
                    if "Test Knowledge Entry" in file_content and "entry_id: test123" in file_content:
                        found_content = True
                        break
            
            assert found_content, "Expected content not found in any proposal file"
    
    def test_propose_new_knowledge_with_malformed_frontmatter(self, temp_knowledge_base):
        """Test proposing a new knowledge entry with malformed frontmatter."""
        with patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH", temp_knowledge_base["review_dir"]):
            content = """---
entry_id: malformed123
title: Malformed Entry
tags: [unclosed bracket,
created: 2023-05-15
last_modified: 2023-05-15
status: draft
---


This entry has malformed frontmatter.
"""
            
            result = propose_new_knowledge(content)
            
            assert "Error" in result
            assert "Invalid YAML" in result
    
    def test_propose_new_knowledge_with_duplicate_entry_id(self, temp_knowledge_base):
        """Test proposing a new knowledge entry with a duplicate entry ID."""
        with patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH", temp_knowledge_base["review_dir"]):
            content1 = """---
entry_id: duplicate123
title: First Entry
tags: [test, duplicate]
created: 2023-05-15
last_modified: 2023-05-15
status: draft
---


This is the first entry with a specific ID.
"""
            
            content2 = """---
entry_id: duplicate123
title: Second Entry
tags: [test, duplicate]
created: 2023-05-16
last_modified: 2023-05-16
status: draft
---


This is the second entry with the same ID.
"""
            
            result1 = propose_new_knowledge(content1)
            assert "New knowledge entry proposed" in result1
            
            result2 = propose_new_knowledge(content2)
            assert "New knowledge entry proposed" in result2
            
            proposals_dir = os.path.join(temp_knowledge_base["proposals_dir"], "test")
            assert os.path.exists(proposals_dir)
            
            files = os.listdir(proposals_dir)
            assert len(files) == 2
    
    def test_category_directory_creation_with_various_tags(self, temp_knowledge_base):
        """Test category directory creation logic with various tag combinations."""
        with patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH", temp_knowledge_base["review_dir"]):
            tag_combinations = [
                (["python", "test", "code"], "python"),  # First tag used
                (["short", "python", "test"], "python"),  # Skip short tag
                (["a", "b", "testing"], "testing"),  # Skip short tags
                (["hyphenated-tag", "underscored_tag"], "general"),  # Skip tags with special chars
                ([], "general"),  # No tags
            ]
            
            for i, (tags, _) in enumerate(tag_combinations):
                content = f"""---
entry_id: tags_test_{i}
title: Tag Test {i}
tags: {tags}
created: 2023-05-15
last_modified: 2023-05-15
status: draft
---


Testing category creation with tags: {tags}
"""
                
                result = propose_new_knowledge(content)
                assert "New knowledge entry proposed" in result
                
                proposals_base_dir = os.path.join(temp_knowledge_base["review_dir"], "proposals")
                
                found_files = []
                for root, _, files in os.walk(proposals_base_dir):
                    for file in files:
                        if file.endswith(".md"):
                            found_files.append(os.path.join(root, file))
                
                found_content = False
                for file_path in found_files:
                    with open(file_path, "r") as f:
                        file_content = f.read()
                        if f"entry_id: tags_test_{i}" in file_content:
                            found_content = True
                            break
                
                assert found_content, f"File for entry_id tags_test_{i} not found in proposals directory"
    
    def test_suggest_knowledge_update_with_real_files(self, temp_knowledge_base):
        """Test suggesting updates for an existing entry with real file operations."""
        with patch("app.tools.proposals.ACTIVE_KNOWLEDGE_PATH", temp_knowledge_base["active_dir"]), \
             patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH", temp_knowledge_base["review_dir"]), \
             patch("app.tools.proposals.KNOWLEDGE_BASE_PATH", temp_knowledge_base["base_dir"]):
            
            result = suggest_knowledge_update(
                "sample123",
                "Add a new section on examples and update formatting.",
                existing_content_verified=False,
            )
            
            assert "Update suggested for knowledge entry" in result
            assert "sample123" in result
            
            updates_dir = os.path.join(temp_knowledge_base["base_dir"], "updates", "general")
            assert os.path.exists(updates_dir)
            
            files = os.listdir(updates_dir)
            assert len(files) == 1
            
            with open(os.path.join(updates_dir, files[0]), "r") as f:
                file_content = f.read()
                assert "Suggested Update for Entry 'sample123'" in file_content
                assert "Add a new section on examples" in file_content
    
    def test_file_permissions_handling(self, temp_knowledge_base):
        """Test handling of file permission errors."""
        read_only_dir = os.path.join(temp_knowledge_base["proposals_dir"], "readonly")
        os.makedirs(read_only_dir, exist_ok=True)
        os.chmod(read_only_dir, 0o555)  # Read and execute, but not write
        
        try:
            with patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH", temp_knowledge_base["review_dir"]):
                content = """---
entry_id: readonly123
title: Read Only Test
tags: [readonly]
created: 2023-05-15
last_modified: 2023-05-15
status: draft
---


This should fail due to permissions.
"""
                
                result = propose_new_knowledge(content)
                
                assert "Error" in result
                assert "Permission" in result or "denied" in result
        finally:
            os.chmod(read_only_dir, 0o755)
    
    @pytest.mark.integration
    def test_with_real_file_synchronization(self, temp_knowledge_base, real_knowledge_sync):
        """Test proposals with real file synchronization."""
        with patch("app.tools.proposals.REVIEW_KNOWLEDGE_PATH", temp_knowledge_base["review_dir"]), \
             patch("app.tools.proposals.ACTIVE_KNOWLEDGE_PATH", temp_knowledge_base["active_dir"]):
            
            content = """---
entry_id: sync_test
title: Synchronization Test
tags: [test, sync]
created: 2023-05-15
last_modified: 2023-05-15
status: draft
---


This entry tests synchronization with ChromaDB.
"""
            
            result = propose_new_knowledge(content)
            assert "New knowledge entry proposed" in result
            
            proposals_dir = os.path.join(temp_knowledge_base["proposals_dir"], "test")
            files = [f for f in os.listdir(proposals_dir) if f.endswith(".md")]
            assert len(files) > 0
            
            proposal_file = os.path.join(proposals_dir, files[0])
            active_file = os.path.join(temp_knowledge_base["active_dir"], "test", "sync_test.md")
            
            os.makedirs(os.path.dirname(active_file), exist_ok=True)
            
            shutil.copy2(proposal_file, active_file)
            
            real_knowledge_sync.process_file_event(active_file, "created")
            
            results = real_knowledge_sync.collection.get(
                where={"entry_id": "sync_test"}
            )
            
            assert len(results["ids"]) > 0
            assert "Synchronization Test" in str(results["documents"])
    


@pytest.mark.integration
class TestExtractFrontmatterEdgeCases:
    """Tests for frontmatter extraction with edge cases."""
    
    def test_frontmatter_with_special_characters(self):
        """Test frontmatter with special characters."""
        content = """---
entry_id: special_chars
title: "Special: Characters & Symbols"
tags: ["tag with spaces", "tag-with-dash", "tag_with_underscore"]
created: 2023-05-15
last_modified: 2023-05-15
status: active
---

This is content with special characters in the frontmatter.
"""
        frontmatter, remaining, has_frontmatter = extract_frontmatter(content)
        
        assert has_frontmatter is True
        assert frontmatter["entry_id"] == "special_chars"
        assert frontmatter["title"] == "Special: Characters & Symbols"
        assert "tag with spaces" in frontmatter["tags"]
        assert "tag-with-dash" in frontmatter["tags"]
        assert "tag_with_underscore" in frontmatter["tags"]
    
    def test_frontmatter_with_nested_structures(self):
        """Test frontmatter with nested structures."""
        content = """---
entry_id: nested_structures
title: Nested Structures
tags: [test]
metadata:
  author: Test Author
  reviewers:
    - Reviewer 1
    - Reviewer 2
  scores:
    quality: 9
    relevance: 8
created: 2023-05-15
last_modified: 2023-05-15
status: active
---

This is content with nested structures in the frontmatter.
"""
        frontmatter, remaining, has_frontmatter = extract_frontmatter(content)
        
        assert has_frontmatter is True
        assert frontmatter["entry_id"] == "nested_structures"
        assert frontmatter["metadata"]["author"] == "Test Author"
        assert "Reviewer 1" in frontmatter["metadata"]["reviewers"]
        assert frontmatter["metadata"]["scores"]["quality"] == 9
    
    def test_frontmatter_with_multiline_strings(self):
        """Test frontmatter with multiline strings."""
        content = """---
entry_id: multiline_strings
title: Multiline Strings
description: |
  This is a multiline description
  that spans multiple lines
  and should be preserved as is.
tags: [test]
created: 2023-05-15
last_modified: 2023-05-15
status: active
---

This is content with multiline strings in the frontmatter.
"""
        frontmatter, remaining, has_frontmatter = extract_frontmatter(content)
        
        assert has_frontmatter is True
        assert frontmatter["entry_id"] == "multiline_strings"
        assert "multiline description" in frontmatter["description"]
        assert "spans multiple lines" in frontmatter["description"]
    
    def test_frontmatter_with_folded_strings(self):
        """Test frontmatter with folded strings."""
        content = """---
entry_id: folded_strings
title: Folded Strings
description: >
  This is a folded string
  that spans multiple lines
  but should be folded into a single line
  with spaces.
tags: [test]
created: 2023-05-15
last_modified: 2023-05-15
status: active
---

This is content with folded strings in the frontmatter.
"""
        frontmatter, remaining, has_frontmatter = extract_frontmatter(content)
        
        assert has_frontmatter is True
        assert frontmatter["entry_id"] == "folded_strings"
        assert "folded string that spans multiple lines" in frontmatter["description"]
    
    def test_frontmatter_with_empty_values(self):
        """Test frontmatter with empty values."""
        content = """---
entry_id: empty_values
title: 
tags: []
description: ""
created: 2023-05-15
last_modified: 2023-05-15
status: active
---

This is content with empty values in the frontmatter.
"""
        frontmatter, remaining, has_frontmatter = extract_frontmatter(content)
        
        assert has_frontmatter is True
        assert frontmatter["entry_id"] == "empty_values"
        assert frontmatter["title"] == "" or frontmatter["title"] is None
        assert frontmatter["tags"] == []
        assert frontmatter["description"] == ""
    
    def test_frontmatter_with_non_standard_delimiters(self):
        """Test handling of non-standard frontmatter delimiters."""
        content = """----
entry_id: non_standard
title: Non-standard Delimiters
tags: [test]
created: 2023-05-15
last_modified: 2023-05-15
status: active
----

This is content with non-standard frontmatter delimiters.
"""
        frontmatter, remaining, has_frontmatter = extract_frontmatter(content)
        
        assert has_frontmatter is False
        assert frontmatter == {}
        assert remaining == content
