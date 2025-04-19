import hashlib
import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional

from watchdog.events import (
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from app.knowledge_management.chunking import chunk_markdown, create_chunk_metadata
from app.knowledge_management.embedder import Embedder
from app.knowledge_management.markdown_parser import (
    get_frontmatter,
    parse_markdown_file,
)

logger = logging.getLogger(__name__)


class MarkdownEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler for Markdown file changes.
    """

    def __init__(self, knowledge_sync):
        """
        Initialize the event handler with a reference to the KnowledgeSync instance.

        Args:
            knowledge_sync: KnowledgeSync instance to handle file events
        """
        self.knowledge_sync = knowledge_sync
        self._debounce_timers = {}
        self._lock = threading.Lock()

    def _debounce(self, file_path, event_type, callback, delay=1.0):
        """
        Debounce file events to prevent multiple rapid updates.

        Args:
            file_path: Path to the file
            event_type: Type of event (created, modified, deleted)
            callback: Function to call after debounce delay
            delay: Debounce delay in seconds
        """
        key = (file_path, event_type)

        with self._lock:
            if key in self._debounce_timers:
                self._debounce_timers[key].cancel()

            timer = threading.Timer(delay, callback)
            timer.daemon = True
            self._debounce_timers[key] = timer
            timer.start()

    def on_created(self, event):
        """
        Handle file creation events.

        Args:
            event: FileCreatedEvent from watchdog
        """
        if not event.is_directory and str(event.src_path).endswith(".md"):
            self._debounce(
                event.src_path,
                "created",
                lambda: self.knowledge_sync.process_file_event(
                    event.src_path, "created"
                ),
            )

    def on_modified(self, event):
        """
        Handle file modification events.

        Args:
            event: FileModifiedEvent from watchdog
        """
        if not event.is_directory and str(event.src_path).endswith(".md"):
            self._debounce(
                event.src_path,
                "modified",
                lambda: self.knowledge_sync.process_file_event(
                    event.src_path, "modified"
                ),
            )

    def on_deleted(self, event):
        """
        Handle file deletion events.

        Args:
            event: FileDeletedEvent from watchdog
        """
        if not event.is_directory and str(event.src_path).endswith(".md"):
            self._debounce(
                event.src_path,
                "deleted",
                lambda: self.knowledge_sync.process_file_event(
                    event.src_path, "deleted"
                ),
            )


class KnowledgeSync:
    """
    Class for synchronizing Markdown files with ChromaDB.
    Uses watchdog for real-time file monitoring.
    """

    def __init__(
        self,
        knowledge_base_path: str,
        chroma_client,
        collection_name: str = "knowledge_base",
        state_file_path: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        enable_watchdog: bool = True,
        embedding_timeout: int = 30,  # Timeout for embedding operations in seconds
        batch_size: int = 10,  # Batch size for embedding operations
    ):
        """
        Initialize the KnowledgeSync with ChromaDB client and paths.

        Args:
            chroma_client: ChromaDB client instance
            knowledge_base_path: Path to the knowledge base directory
            collection_name: Name of the ChromaDB collection
            state_file_path: Path to the state file (defaults to knowledge_base_path/.sync_state.json)
            embedder: Embedder instance for generating embeddings
            enable_watchdog: Whether to enable watchdog file monitoring
            embedding_timeout: Timeout for embedding operations in seconds
            batch_size: Batch size for embedding operations
        """
        self.chroma_client = chroma_client
        self.knowledge_base_path = knowledge_base_path
        self.collection_name = collection_name
        self.state_file_path = state_file_path or os.path.join(
            knowledge_base_path, ".sync_state.json"
        )
        self.embedder = embedder or Embedder()
        self.enable_watchdog = enable_watchdog
        self.embedding_timeout = embedding_timeout
        self.batch_size = batch_size
        self._lock = threading.Lock()

        # Synchronization status tracking
        self.is_syncing = False
        self.sync_progress = {
            "total_files": 0,
            "processed_files": 0,
            "total_chunks": 0,
            "processed_chunks": 0,
            "errors": 0,
            "last_error": None,
            "last_processed_file": None,
        }

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

        self.state = self._load_state()

        self.observer = None
        if self.enable_watchdog:
            self._setup_watchdog()

    def _setup_watchdog(self):
        """
        Set up watchdog observer for file monitoring.
        """
        self.observer = Observer()
        event_handler = MarkdownEventHandler(self)
        self.observer.schedule(event_handler, self.knowledge_base_path, recursive=True)
        self.observer.daemon = True
        self.observer.start()
        logger.info(f"Watchdog observer started for {self.knowledge_base_path}")

    def start(self):
        """
        Start the knowledge sync service.
        """
        if self.enable_watchdog and not self.observer:
            self._setup_watchdog()

    def stop(self):
        """
        Stop the knowledge sync service.
        Stops the watchdog observer if running.
        """
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Watchdog observer stopped")

    def _load_state(self) -> Dict[str, str]:
        """
        Load the state file or initialize an empty state.

        Returns:
            Dictionary mapping file paths to content hashes
        """
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(
                    f"Error loading state file {self.state_file_path}. Initializing empty state."
                )
                return {}
        return {}

    def _save_state(self):
        """
        Save the current state to the state file.
        """
        with self._lock:
            with open(self.state_file_path, "w") as f:
                json.dump(self.state, f, indent=2)

    def _get_markdown_files(self) -> List[str]:
        """
        Get all Markdown files in the knowledge base directory.

        Returns:
            List of absolute paths to Markdown files
        """
        markdown_files = []
        for root, _, files in os.walk(self.knowledge_base_path):
            for file in files:
                if file.endswith(".md"):
                    markdown_files.append(os.path.join(root, file))
        return markdown_files

    def _compute_content_hash(self, file_path: str) -> str:
        """
        Compute the SHA-256 hash of a file's content.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash of the file's content
        """
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _process_file(self, file_path: str, content_hash: str):
        """
        Process a Markdown file and update ChromaDB.

        Args:
            file_path: Path to the Markdown file
            content_hash: Hash of the file's content
        """
        try:
            content, frontmatter = parse_markdown_file(file_path)

            if not frontmatter or "entry_id" not in frontmatter:
                logger.warning(
                    f"Skipping {file_path}: Missing required frontmatter (entry_id)"
                )
                return

            entry_id = frontmatter["entry_id"]
            title = frontmatter.get("title", entry_id)
            tags = frontmatter.get("tags", [])
            last_modified = frontmatter.get("last_modified", time.strftime("%Y-%m-%d"))

            self._delete_entry_chunks(entry_id)

            chunks = chunk_markdown(content)

            with self._lock:
                self.sync_progress["total_chunks"] += len(chunks)

            for i in range(0, len(chunks), self.batch_size):
                batch_chunks = chunks[i : i + self.batch_size]

                try:
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            self.embedder.generate_embeddings_batch, batch_chunks
                        )
                        embeddings = future.result(timeout=self.embedding_timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(
                        f"Embedding operation timed out after {self.embedding_timeout} seconds for {file_path}"
                    )
                    continue
                except Exception as e:
                    logger.error(f"Error generating embeddings for {file_path}: {e}")
                    continue

                ids = []
                metadatas = []
                documents = []

                for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_index = i + j
                    chunk_id = self.embedder.generate_chunk_id(entry_id, chunk_index)

                    metadata = create_chunk_metadata(
                        chunk_index=chunk_index,
                        source_file=os.path.relpath(
                            file_path, self.knowledge_base_path
                        ),
                        entry_id=entry_id,
                        title=title,
                        tags=tags,
                        last_modified=last_modified,
                        content_hash=content_hash,
                    )

                    ids.append(chunk_id)
                    metadatas.append(metadata)
                    documents.append(chunk)

                try:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        documents=documents,
                    )

                    with self._lock:
                        self.sync_progress["processed_chunks"] += len(batch_chunks)

                except Exception as e:
                    logger.error(
                        f"Error adding chunks to ChromaDB for {file_path}: {e}"
                    )

            logger.info(
                f"Processed {file_path}: {len(chunks)} chunks added to ChromaDB"
            )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            with self._lock:
                self.sync_progress["errors"] += 1
                self.sync_progress["last_error"] = str(e)

    def _delete_entry_chunks(self, entry_id: str):
        """
        Delete all chunks for a specific entry from ChromaDB.

        Args:
            entry_id: Unique identifier of the knowledge entry
        """
        try:
            results = self.collection.get(where={"entry_id": entry_id})

            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(
                    f"Deleted {len(results['ids'])} existing chunks for entry_id: {entry_id}"
                )
        except Exception as e:
            logger.error(f"Error deleting chunks for entry_id {entry_id}: {e}")

    def process_file_event(self, file_path: str, event_type: str):
        """
        Process a file event from watchdog.

        Args:
            file_path: Path to the file
            event_type: Type of event (created, modified, deleted)
        """
        with self._lock:
            try:
                if event_type in ("created", "modified"):
                    content_hash = self._compute_content_hash(file_path)

                    if (
                        file_path not in self.state
                        or self.state[file_path] != content_hash
                    ):
                        logger.info(f"Processing {event_type} file: {file_path}")
                        self._process_file(file_path, content_hash)
                        self.state[file_path] = content_hash
                        self._save_state()

                elif event_type == "deleted":
                    if file_path in self.state:
                        entry_id = None

                        filename = os.path.basename(file_path)
                        if filename.endswith(".md"):
                            entry_id = filename[:-3]  # Remove .md extension

                        if entry_id:
                            self._delete_entry_chunks(entry_id)

                        del self.state[file_path]
                        self._save_state()
                        logger.info(f"Processed deleted file: {file_path}")

            except Exception as e:
                logger.error(
                    f"Error processing {event_type} event for {file_path}: {e}"
                )

    def get_sync_status(self):
        """
        Get the current synchronization status.

        Returns:
            Dictionary with synchronization status information
        """
        with self._lock:
            return {
                "is_syncing": self.is_syncing,
                "progress": self.sync_progress.copy(),
            }

    def sync(self):
        """
        Synchronize the knowledge base with ChromaDB.
        Detects new, modified, and deleted files and updates ChromaDB accordingly.
        """
        logger.info("Starting knowledge base synchronization...")

        with self._lock:
            self.is_syncing = True

            self.sync_progress = {
                "total_files": 0,
                "processed_files": 0,
                "total_chunks": 0,
                "processed_chunks": 0,
                "errors": 0,
                "last_error": None,
                "last_processed_file": None,
            }

            try:
                # Get current Markdown files
                current_files = self._get_markdown_files()
                current_files_set = set(current_files)

                deleted_files = set(self.state.keys()) - current_files_set

                self.sync_progress["total_files"] = len(current_files) + len(
                    deleted_files
                )

                for file_path in current_files:
                    try:
                        content_hash = self._compute_content_hash(file_path)

                        if (
                            file_path not in self.state
                            or self.state[file_path] != content_hash
                        ):
                            logger.info(
                                f"Processing {'new' if file_path not in self.state else 'modified'} file: {file_path}"
                            )
                            self._process_file(file_path, content_hash)
                            self.state[file_path] = content_hash

                            self.sync_progress["processed_files"] += 1
                            self.sync_progress["last_processed_file"] = file_path
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        self.sync_progress["errors"] += 1
                        self.sync_progress["last_error"] = str(e)

                for file_path in deleted_files:
                    try:
                        frontmatter = (
                            get_frontmatter(file_path)
                            if os.path.exists(file_path)
                            else None
                        )

                        if frontmatter and "entry_id" in frontmatter:
                            entry_id = frontmatter["entry_id"]
                            self._delete_entry_chunks(entry_id)

                        del self.state[file_path]
                        logger.info(f"Processed deleted file: {file_path}")

                        self.sync_progress["processed_files"] += 1
                        self.sync_progress["last_processed_file"] = file_path
                    except Exception as e:
                        logger.error(f"Error handling deleted file {file_path}: {e}")
                        self.sync_progress["errors"] += 1
                        self.sync_progress["last_error"] = str(e)

                self._save_state()

                logger.info(
                    f"Synchronization completed. Processed {len(current_files)} files, {len(deleted_files)} deletions."
                )

            finally:
                self.is_syncing = False
