#!/usr/bin/env python3
import logging
import os
import sys
import time

# Add the project root to the path so we can import from app
# Go up one directory level from scripts to the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import ACTIVE_KNOWLEDGE_PATH, chroma_client
from app.knowledge_management.synchronization import KnowledgeSync
from app.knowledge_management.embedder import Embedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def force_update_embeddings():
    """
    Force update all embeddings for markdown files in the knowledge base.
    Ignores the file hash check to ensure all files are re-processed.
    """
    logger.info("Starting forced update of all knowledge base embeddings...")
    logger.info(f"Knowledge base path: {ACTIVE_KNOWLEDGE_PATH}")

    # Create a KnowledgeSync instance
    knowledge_sync = KnowledgeSync(
        ACTIVE_KNOWLEDGE_PATH,
        chroma_client,
        embedding_timeout=60,  # Increase timeout for larger files
        batch_size=5,  # Reduce batch size to minimize memory usage
        enable_watchdog=False,  # No need for file watching in one-time operation
    )

    # Clear the state to force re-processing of all files
    knowledge_sync.state = {}

    # Get a list of all markdown files
    markdown_files = knowledge_sync._get_markdown_files()
    total_files = len(markdown_files)

    logger.info(f"Found {total_files} markdown files to process")

    # Process each file
    for i, file_path in enumerate(markdown_files, 1):
        try:
            logger.info(f"[{i}/{total_files}] Processing {file_path}")

            # Compute the hash but we'll force processing regardless
            content_hash = knowledge_sync._compute_content_hash(file_path)

            # Process the file (this will re-chunk and re-embed)
            knowledge_sync._process_file(file_path, content_hash)

            # Update the state
            knowledge_sync.state[file_path] = content_hash

            logger.info(f"[{i}/{total_files}] Completed processing {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Save the final state
    knowledge_sync._save_state()

    # Report stats
    logger.info("Forced update completed!")
    logger.info(f"Processed {total_files} files")
    logger.info(f"Created {knowledge_sync.sync_progress['processed_chunks']} chunks")
    if knowledge_sync.sync_progress["errors"] > 0:
        logger.warning(f"Encountered {knowledge_sync.sync_progress['errors']} errors")
        logger.warning(f"Last error: {knowledge_sync.sync_progress['last_error']}")


if __name__ == "__main__":
    start_time = time.time()
    force_update_embeddings()
    end_time = time.time()
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")
