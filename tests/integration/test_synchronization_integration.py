import json
import os
import pytest
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import chromadb

from app.knowledge_management.synchronization import KnowledgeSync, MarkdownEventHandler
from app.knowledge_management.embedder import Embedder


@pytest.fixture
def temp_knowledge_dir_with_files():
    """Create a temporary directory with some markdown files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file1_path = os.path.join(temp_dir, "test1.md")
        file2_path = os.path.join(temp_dir, "test2.md")
        
        with open(file1_path, "w") as f:
            f.write("""---
entry_id: test-entry-1
title: Test Entry 1
tags: [test, sample]
created: 2023-05-15
last_modified: 2023-05-15
status: active
---


This is test content for file 1.
""")
        
        with open(file2_path, "w") as f:
            f.write("""---
entry_id: test-entry-2
title: Test Entry 2
tags: [test, example]
created: 2023-05-16
last_modified: 2023-05-16
status: active
---


This is test content for file 2.
""")
        
        yield temp_dir


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
def real_knowledge_sync(real_chroma_client, real_embedder, temp_knowledge_dir_with_files):
    """Create a KnowledgeSync instance with real dependencies but watchdog disabled."""
    knowledge_sync = KnowledgeSync(
        knowledge_base_path=temp_knowledge_dir_with_files,
        chroma_client=real_chroma_client,
        embedder=real_embedder,
        enable_watchdog=False,  # Disable watchdog to control file events manually
        embedding_timeout=10,
        batch_size=5,
    )
    
    yield knowledge_sync
    
    knowledge_sync.stop()


@pytest.mark.integration
def test_sync_with_real_files(real_knowledge_sync):
    """Test synchronization with real files and minimal mocking."""
    real_knowledge_sync.sync()
    
    assert len(real_knowledge_sync.state) == 2
    
    results = real_knowledge_sync.collection.get()
    assert len(results["ids"]) > 0
    
    entry_ids = set()
    for metadata in results["metadatas"]:
        entry_ids.add(metadata["entry_id"])
    
    assert "test-entry-1" in entry_ids
    assert "test-entry-2" in entry_ids


@pytest.mark.integration
def test_file_modification(real_knowledge_sync, temp_knowledge_dir_with_files):
    """Test file modification detection and processing."""
    real_knowledge_sync.sync()
    
    initial_state = real_knowledge_sync.state.copy()
    
    file_path = os.path.join(temp_knowledge_dir_with_files, "test1.md")
    with open(file_path, "a") as f:
        f.write("\nThis is additional content added during the test.")
    
    real_knowledge_sync.process_file_event(file_path, "modified")
    
    assert real_knowledge_sync.state[file_path] != initial_state[file_path]
    
    results = real_knowledge_sync.collection.get(
        where={"entry_id": "test-entry-1"}
    )
    
    found_new_content = False
    for doc in results["documents"]:
        if "This is additional content added during the test" in doc:
            found_new_content = True
            break
    
    assert found_new_content, "Modified content was not found in the collection"


@pytest.mark.integration
def test_file_deletion(real_knowledge_sync, temp_knowledge_dir_with_files):
    """Test file deletion detection and processing."""
    real_knowledge_sync.sync()
    
    initial_results = real_knowledge_sync.collection.get(
        where={"entry_id": "test-entry-2"}
    )
    assert len(initial_results["ids"]) > 0
    
    file_path = os.path.join(temp_knowledge_dir_with_files, "test2.md")
    os.remove(file_path)
    
    real_knowledge_sync.process_file_event(file_path, "deleted")
    
    assert file_path not in real_knowledge_sync.state
    
    results = real_knowledge_sync.collection.get(
        where={"entry_id": "test-entry-2"}
    )
    assert len(results["ids"]) == 0


@pytest.mark.integration
def test_file_event_ordering(real_knowledge_sync, temp_knowledge_dir_with_files):
    """Test file event ordering (create→modify→delete) with real file operations."""
    file_path = os.path.join(temp_knowledge_dir_with_files, "test_ordering.md")
    with open(file_path, "w") as f:
        f.write("""---
entry_id: test-ordering
title: Test Ordering
tags: [test, ordering]
created: 2023-05-17
last_modified: 2023-05-17
status: active
---


Initial content for ordering test.
""")
    
    real_knowledge_sync.process_file_event(file_path, "created")
    
    assert file_path in real_knowledge_sync.state
    
    results = real_knowledge_sync.collection.get(
        where={"entry_id": "test-ordering"}
    )
    assert len(results["ids"]) > 0
    initial_ids = results["ids"]
    
    with open(file_path, "a") as f:
        f.write("\nModified content for ordering test.")
    
    real_knowledge_sync.process_file_event(file_path, "modified")
    
    results = real_knowledge_sync.collection.get(
        where={"entry_id": "test-ordering"}
    )
    assert len(results["ids"]) > 0
    
    found_modified_content = False
    for doc in results["documents"]:
        if "Modified content for ordering test" in doc:
            found_modified_content = True
            break
    
    assert found_modified_content, "Modified content was not found in the collection"
    
    os.remove(file_path)
    
    real_knowledge_sync.process_file_event(file_path, "deleted")
    
    assert file_path not in real_knowledge_sync.state
    
    results = real_knowledge_sync.collection.get(
        where={"entry_id": "test-ordering"}
    )
    assert len(results["ids"]) == 0


@pytest.mark.integration
def test_race_conditions(real_knowledge_sync, temp_knowledge_dir_with_files):
    """Test handling of race conditions between file operations."""
    file_paths = []
    threads = []
    
    for i in range(5):
        file_path = os.path.join(temp_knowledge_dir_with_files, f"race_test_{i}.md")
        with open(file_path, "w") as f:
            f.write(f"""---
entry_id: race-test-{i}
title: Race Test {i}
tags: [test, race]
created: 2023-05-18
last_modified: 2023-05-18
status: active
---


Content for race condition test {i}.
""")
        file_paths.append(file_path)
    
    def process_file(file_path, event_type):
        real_knowledge_sync.process_file_event(file_path, event_type)
    
    for file_path in file_paths:
        thread = threading.Thread(target=process_file, args=(file_path, "created"))
        threads.append(thread)
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join(timeout=10)
    
    for file_path in file_paths:
        assert file_path in real_knowledge_sync.state
    
    for i in range(5):
        results = real_knowledge_sync.collection.get(
            where={"entry_id": f"race-test-{i}"}
        )
        assert len(results["ids"]) > 0


@pytest.mark.integration
def test_debouncing_behavior():
    """Test debouncing behavior with timed file modifications."""
    mock_sync = MagicMock()
    
    handler = MarkdownEventHandler(mock_sync)
    
    with tempfile.NamedTemporaryFile(suffix=".md") as temp_file:
        for _ in range(5):
            handler._debounce(
                temp_file.name,
                "modified",
                lambda: mock_sync.process_file_event(temp_file.name, "modified")
            )
            time.sleep(0.1)  # Small delay between events
        
        time.sleep(1.5)  # Slightly longer than the default 1.0s debounce delay
        
        mock_sync.process_file_event.assert_called_once_with(temp_file.name, "modified")


@pytest.mark.integration
def test_collection_updates_after_sync(real_knowledge_sync, temp_knowledge_dir_with_files):
    """Test that collection is properly updated after synchronization."""
    real_knowledge_sync.sync()
    
    initial_results = real_knowledge_sync.collection.get()
    initial_count = len(initial_results["ids"])
    
    file_path = os.path.join(temp_knowledge_dir_with_files, "new_file.md")
    with open(file_path, "w") as f:
        f.write("""---
entry_id: new-test-entry
title: New Test Entry
tags: [test, new]
created: 2023-05-19
last_modified: 2023-05-19
status: active
---


This is content for a new test entry.
""")
    
    real_knowledge_sync.sync()
    
    updated_results = real_knowledge_sync.collection.get()
    updated_count = len(updated_results["ids"])
    
    assert updated_count > initial_count
    
    results = real_knowledge_sync.collection.get(
        where={"entry_id": "new-test-entry"}
    )
    assert len(results["ids"]) > 0
    
    with open(file_path, "a") as f:
        f.write("\nAdditional content for the new test entry.")
    
    real_knowledge_sync.sync()
    
    results = real_knowledge_sync.collection.get(
        where={"entry_id": "new-test-entry"}
    )
    
    found_additional_content = False
    for doc in results["documents"]:
        if "Additional content for the new test entry" in doc:
            found_additional_content = True
            break
    
    assert found_additional_content, "Modified content was not found in the collection"
    
    os.remove(file_path)
    
    real_knowledge_sync.sync()
    
    results = real_knowledge_sync.collection.get(
        where={"entry_id": "new-test-entry"}
    )
    assert len(results["ids"]) == 0
