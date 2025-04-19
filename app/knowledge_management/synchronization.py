import os
import json
import time
import threading
from typing import Dict, List, Any, Optional
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent, FileDeletedEvent
from app.knowledge_management.markdown_parser import parse_markdown_file, get_frontmatter
from app.knowledge_management.chunking import chunk_markdown, create_chunk_metadata
from app.knowledge_management.embedder import Embedder

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
        if not event.is_directory and str(event.src_path).endswith('.md'):
            self._debounce(
                event.src_path, 
                'created', 
                lambda: self.knowledge_sync.process_file_event(event.src_path, 'created')
            )
    
    def on_modified(self, event):
        """
        Handle file modification events.
        
        Args:
            event: FileModifiedEvent from watchdog
        """
        if not event.is_directory and str(event.src_path).endswith('.md'):
            self._debounce(
                event.src_path, 
                'modified', 
                lambda: self.knowledge_sync.process_file_event(event.src_path, 'modified')
            )
    
    def on_deleted(self, event):
        """
        Handle file deletion events.
        
        Args:
            event: FileDeletedEvent from watchdog
        """
        if not event.is_directory and str(event.src_path).endswith('.md'):
            self._debounce(
                event.src_path, 
                'deleted', 
                lambda: self.knowledge_sync.process_file_event(event.src_path, 'deleted')
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
        enable_watchdog: bool = True
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
        """
        self.chroma_client = chroma_client
        self.knowledge_base_path = knowledge_base_path
        self.collection_name = collection_name
        self.state_file_path = state_file_path or os.path.join(knowledge_base_path, ".sync_state.json")
        self.embedder = embedder or Embedder()
        self.enable_watchdog = enable_watchdog
        self._lock = threading.Lock()
        
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
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
        print(f"Watchdog observer started for {self.knowledge_base_path}")
    
    def start(self):
        """
        Start the knowledge sync service.
        Performs initial synchronization and starts watchdog if enabled.
        """
        self.sync()
        
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
            print("Watchdog observer stopped")
    
    def _load_state(self) -> Dict[str, str]:
        """
        Load the state file or initialize an empty state.
        
        Returns:
            Dictionary mapping file paths to content hashes
        """
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading state file {self.state_file_path}. Initializing empty state.")
                return {}
        return {}
    
    def _save_state(self):
        """
        Save the current state to the state file.
        """
        with self._lock:
            with open(self.state_file_path, 'w') as f:
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
                if file.endswith('.md'):
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
        with open(file_path, 'rb') as f:
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
            
            if not frontmatter or 'entry_id' not in frontmatter:
                print(f"Skipping {file_path}: Missing required frontmatter (entry_id)")
                return
            
            entry_id = frontmatter['entry_id']
            title = frontmatter.get('title', entry_id)
            tags = frontmatter.get('tags', [])
            last_modified = frontmatter.get('last_modified', time.strftime('%Y-%m-%d'))
            
            self._delete_entry_chunks(entry_id)
            
            chunks = chunk_markdown(content)
            
            batch_size = 10  # Adjust based on API limits
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                
                embeddings = self.embedder.generate_embeddings_batch(batch_chunks)
                
                ids = []
                metadatas = []
                documents = []
                
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_index = i + j
                    chunk_id = self.embedder.generate_chunk_id(entry_id, chunk_index)
                    
                    metadata = create_chunk_metadata(
                        chunk_index=chunk_index,
                        source_file=os.path.relpath(file_path, self.knowledge_base_path),
                        entry_id=entry_id,
                        title=title,
                        tags=tags,
                        last_modified=last_modified,
                        content_hash=content_hash
                    )
                    
                    ids.append(chunk_id)
                    metadatas.append(metadata)
                    documents.append(chunk)
                
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
            
            print(f"Processed {file_path}: {len(chunks)} chunks added to ChromaDB")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def _delete_entry_chunks(self, entry_id: str):
        """
        Delete all chunks for a specific entry from ChromaDB.
        
        Args:
            entry_id: Unique identifier of the knowledge entry
        """
        try:
            results = self.collection.get(
                where={"entry_id": entry_id}
            )
            
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"Deleted {len(results['ids'])} existing chunks for entry_id: {entry_id}")
        except Exception as e:
            print(f"Error deleting chunks for entry_id {entry_id}: {e}")
    
    def process_file_event(self, file_path: str, event_type: str):
        """
        Process a file event from watchdog.
        
        Args:
            file_path: Path to the file
            event_type: Type of event (created, modified, deleted)
        """
        with self._lock:
            try:
                if event_type in ('created', 'modified'):
                    content_hash = self._compute_content_hash(file_path)
                    
                    if file_path not in self.state or self.state[file_path] != content_hash:
                        print(f"Processing {event_type} file: {file_path}")
                        self._process_file(file_path, content_hash)
                        self.state[file_path] = content_hash
                        self._save_state()
                
                elif event_type == 'deleted':
                    if file_path in self.state:
                        entry_id = None
                        
                        filename = os.path.basename(file_path)
                        if filename.endswith('.md'):
                            entry_id = filename[:-3]  # Remove .md extension
                        
                        if entry_id:
                            self._delete_entry_chunks(entry_id)
                        
                        del self.state[file_path]
                        self._save_state()
                        print(f"Processed deleted file: {file_path}")
            
            except Exception as e:
                print(f"Error processing {event_type} event for {file_path}: {e}")
    
    def sync(self):
        """
        Synchronize the knowledge base with ChromaDB.
        Detects new, modified, and deleted files and updates ChromaDB accordingly.
        """
        print("Starting knowledge base synchronization...")
        
        with self._lock:
            # Get current Markdown files
            current_files = self._get_markdown_files()
            current_files_set = set(current_files)
            
            deleted_files = set(self.state.keys()) - current_files_set
            
            for file_path in current_files:
                try:
                    content_hash = self._compute_content_hash(file_path)
                    
                    if file_path not in self.state or self.state[file_path] != content_hash:
                        print(f"Processing {'new' if file_path not in self.state else 'modified'} file: {file_path}")
                        self._process_file(file_path, content_hash)
                        self.state[file_path] = content_hash
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
            
            for file_path in deleted_files:
                try:
                    frontmatter = get_frontmatter(file_path) if os.path.exists(file_path) else None
                    
                    if frontmatter and 'entry_id' in frontmatter:
                        entry_id = frontmatter['entry_id']
                        self._delete_entry_chunks(entry_id)
                    
                    del self.state[file_path]
                    print(f"Processed deleted file: {file_path}")
                except Exception as e:
                    print(f"Error handling deleted file {file_path}: {e}")
            
            self._save_state()
            
            print(f"Synchronization completed. Processed {len(current_files)} files, {len(deleted_files)} deletions.")
