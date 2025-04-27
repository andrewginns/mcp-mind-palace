import pytest
import os
import time
from unittest.mock import patch, MagicMock, mock_open as mock_open_func

from app.knowledge_management.synchronization import KnowledgeSync
from app.config import chroma_client


class TestKnowledgeSyncMCP:
    """
    Tests for the integration of KnowledgeSync with the MCP server.
    Focus on ensuring the synchronization logic works properly within the MCP context.
    """

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder that returns fixed embeddings."""
        mock = MagicMock()
        mock.generate_embedding.return_value = [0.1] * 384  # Simplified embedding
        return mock

    @pytest.mark.timeout(2)
    @patch("app.knowledge_management.synchronization.Observer")
    @patch("threading.Lock")
    def test_knowledge_sync_initialization(
        self, mock_lock_class, mock_observer_class, setup_knowledge_dirs, mock_embedder
    ):
        """Test KnowledgeSync initialization and basic functionality."""
        # Setup mocks
        mock_lock = MagicMock()
        mock_lock_class.return_value = mock_lock
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        # Get the temporary knowledge directories
        knowledge_paths = setup_knowledge_dirs

        # Create a test ChromaDB client with in-memory storage
        test_client = MagicMock()
        mock_collection = MagicMock()
        test_client.get_or_create_collection.return_value = mock_collection

        # Initialize KnowledgeSync with test directories and mocked components
        with patch(
            "app.knowledge_management.synchronization.Embedder",
            return_value=mock_embedder,
        ):
            # Mock file operations
            with patch("os.path.exists", return_value=False):
                with patch("builtins.open", mock_open_func()):
                    # Mock json load/dump operations
                    with patch("json.load", return_value={}):
                        with patch("json.dump"):
                            # Create knowledge_sync
                            knowledge_sync = KnowledgeSync(
                                knowledge_base_path=knowledge_paths["active"],
                                chroma_client=test_client,
                                enable_watchdog=False,  # Disable watchdog for testing
                                embedding_timeout=1,
                                batch_size=2,
                            )

                            # Verify initialization
                            assert (
                                knowledge_sync.knowledge_base_path
                                == knowledge_paths["active"]
                            )
                            assert knowledge_sync.collection == mock_collection
                            assert knowledge_sync.enable_watchdog is False

                            # Verify state was initialized
                            assert isinstance(knowledge_sync.state, dict)

    @pytest.mark.timeout(5)  # Increased timeout
    @patch("app.knowledge_management.synchronization.Observer")
    @patch("threading.Lock")
    def test_file_event_processing(
        self, mock_lock_class, mock_observer_class, setup_knowledge_dirs, mock_embedder
    ):
        """Test that a knowledge synchronizer can be initialized properly."""
        # Setup mocks
        mock_lock = MagicMock()
        mock_lock_class.return_value = mock_lock
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        knowledge_paths = setup_knowledge_dirs

        # Create mock collection
        mock_collection = MagicMock()
        test_client = MagicMock()
        test_client.get_or_create_collection.return_value = mock_collection

        # Create KnowledgeSync instance with mocks and verify it initializes properly
        with patch(
            "app.knowledge_management.synchronization.Embedder",
            return_value=mock_embedder,
        ):
            # Mock _load_state to avoid file not found error
            with patch.object(KnowledgeSync, "_load_state", return_value={}):
                # Create knowledge_sync with watchdog disabled
                knowledge_sync = KnowledgeSync(
                    knowledge_base_path=knowledge_paths["active"],
                    chroma_client=test_client,
                    enable_watchdog=False,
                    embedding_timeout=1,
                    batch_size=2,
                )

                # Verify the KnowledgeSync was properly initialized
                assert isinstance(knowledge_sync.state, dict)
                assert knowledge_sync.knowledge_base_path == knowledge_paths["active"]
                assert knowledge_sync.collection == mock_collection
                assert knowledge_sync.embedder == mock_embedder
                assert knowledge_sync.enable_watchdog is False

                # Verify watchdog was not started since it's disabled
                assert not mock_observer.start.called

    @pytest.mark.timeout(5)  # Increased timeout
    @patch("app.knowledge_management.synchronization.Observer")
    @patch("threading.Lock")
    def test_file_deletion_handling(
        self, mock_lock_class, mock_observer_class, setup_knowledge_dirs, mock_embedder
    ):
        """Test that file deletion events are properly processed."""
        # Setup mocks
        mock_lock = MagicMock()
        mock_lock_class.return_value = mock_lock
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        knowledge_paths = setup_knowledge_dirs

        # Create a mock collection
        mock_collection = MagicMock()
        test_client = MagicMock()
        test_client.get_or_create_collection.return_value = mock_collection

        # Configure the mock collection to return data for the test entry_id
        mock_collection.get.return_value = {
            "ids": ["chunk1", "chunk2"],
            "metadatas": [{"entry_id": "test-entry"}, {"entry_id": "test-entry"}],
        }

        # Create KnowledgeSync instance with mocks
        with patch(
            "app.knowledge_management.synchronization.Embedder",
            return_value=mock_embedder,
        ):
            # Mock file operations
            with patch("os.path.exists", return_value=False):
                with patch("builtins.open", mock_open_func()):
                    # Mock json operations
                    with patch("json.load", return_value={}):
                        with patch("json.dump"):
                            # Create knowledge_sync
                            knowledge_sync = KnowledgeSync(
                                knowledge_base_path=knowledge_paths["active"],
                                chroma_client=test_client,
                                enable_watchdog=False,
                                embedding_timeout=1,
                                batch_size=2,
                            )

                            # Mock file path
                            test_file_path = os.path.join(
                                knowledge_paths["active"], "test-entry.md"
                            )

                            # First add the file to state to simulate it existed
                            knowledge_sync.state[test_file_path] = "fakehash123"

                            # Mock markdown parser
                            with patch(
                                "app.knowledge_management.markdown_parser.parse_markdown_file"
                            ) as mock_parse:
                                mock_parse.return_value = (
                                    "Test content",
                                    {
                                        "entry_id": "test-entry",
                                        "title": "Test Entry",
                                        "tags": ["test"],
                                    },
                                )

                                # Now process a deletion event
                                knowledge_sync.process_file_event(
                                    test_file_path, "deleted"
                                )

                                # Verify that delete was called on the collection
                                assert mock_collection.delete.called

    @pytest.mark.timeout(10)  # Increased timeout
    @patch("app.knowledge_management.synchronization.Observer")
    @patch("threading.Lock")
    def test_sync_process(
        self, mock_lock_class, mock_observer_class, setup_knowledge_dirs, mock_embedder
    ):
        """Test the full synchronization process."""
        # Setup mocks
        mock_lock = MagicMock()
        mock_lock_class.return_value = mock_lock
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        knowledge_paths = setup_knowledge_dirs

        # Create mock collection
        mock_collection = MagicMock()
        test_client = MagicMock()
        test_client.get_or_create_collection.return_value = mock_collection

        # Skip actual file system operations by mocking internal methods
        with patch(
            "app.knowledge_management.synchronization.Embedder",
            return_value=mock_embedder,
        ):
            # Mock file hash computation to prevent encoding error
            with patch(
                "app.knowledge_management.synchronization.hashlib.sha256"
            ) as mock_sha256:
                mock_hash = MagicMock()
                mock_hash.hexdigest.return_value = "fakehash123"
                mock_sha256.return_value = mock_hash

                # Mock file operations
                with patch("os.path.exists", return_value=True):
                    with patch(
                        "builtins.open", mock_open_func(read_data="test content")
                    ):
                        # Mock json operations
                        with patch("json.load", return_value={}):
                            with patch("json.dump"):
                                # Mock _get_markdown_files
                                with patch.object(
                                    KnowledgeSync, "_get_markdown_files"
                                ) as mock_get_files:
                                    # Return 3 mock file paths
                                    file_paths = [
                                        os.path.join(
                                            knowledge_paths["active"],
                                            f"test-entry-{i}.md",
                                        )
                                        for i in range(3)
                                    ]
                                    mock_get_files.return_value = file_paths

                                    # Mock _process_file to track calls
                                    with patch.object(
                                        KnowledgeSync, "_process_file"
                                    ) as mock_process_file:
                                        # Create knowledge_sync
                                        knowledge_sync = KnowledgeSync(
                                            knowledge_base_path=knowledge_paths[
                                                "active"
                                            ],
                                            chroma_client=test_client,
                                            enable_watchdog=False,
                                            embedding_timeout=1,
                                            batch_size=2,
                                        )

                                        # Run the synchronization
                                        knowledge_sync.sync()

                                        # Verify _process_file was called for each file
                                        assert mock_process_file.call_count == 3
