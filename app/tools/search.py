import logging
from typing import Any, Dict, List

from app.config import chroma_client
from app.knowledge_management.embedder import Embedder

logger = logging.getLogger(__name__)


def search_knowledge(task_description: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Tool for searching the knowledge base for similar content.
    Performs semantic search on ChromaDB using the embedding of task_description.
    Returns top k relevant chunks with content, metadata, and similarity score.
    Also includes a relevance_comment to help guide how to interpret results.

    Args:
        task_description: Description of the current task
        top_k: Number of results to return

    Returns:
        List of dictionaries containing content, metadata, similarity score, and relevance guidance
    """
    try:
        collection = chroma_client.get_collection("knowledge_base")

        embedder = Embedder()
        query_embedding = embedder.generate_embedding(task_description)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                similarity = 1.0 - results["distances"][0][i]

                # Add a relevance comment to guide the LLM
                if similarity > 0.9:
                    relevance = "Highly relevant - consider updating this entry instead of creating new content"
                elif similarity > 0.75:
                    relevance = "Moderately relevant - examine full content to determine if updates are needed"
                elif similarity > 0.6:
                    relevance = "Somewhat relevant - may contain partial information, consider cross-referencing"
                else:
                    relevance = "Low relevance - likely covers different aspects, new content may be justified"

                formatted_results.append(
                    {
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": similarity,
                        "relevance_comment": relevance,
                    }
                )

        # If no results, provide guidance for creating new content
        if not formatted_results:
            logger.info(f"No relevant results found for: {task_description}")
            return [
                {
                    "content": "",
                    "metadata": {},
                    "similarity_score": 0.0,
                    "relevance_comment": "No relevant content found - creating new knowledge may be appropriate",
                }
            ]

        return formatted_results

    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        return []


def get_entry_details(entry_id: str) -> Dict[str, Any]:
    """
    Tool for retrieving full details (original Markdown content, metadata) for a specific entry_id.

    Args:
        entry_id: Unique identifier of the knowledge entry

    Returns:
        Dictionary containing the entry's metadata and content
    """
    try:
        collection = chroma_client.get_collection("knowledge_base")

        results = collection.get(
            where={"entry_id": entry_id}, include=["documents", "metadatas"]
        )

        if not results or not results["ids"]:
            return {"error": f"Knowledge entry with ID '{entry_id}' not found"}

        sorted_indices = sorted(
            range(len(results["ids"])),
            key=lambda i: results["metadatas"][i].get("chunk_index", 0),
        )

        full_content = "\n".join([results["documents"][i] for i in sorted_indices])

        metadata = results["metadatas"][sorted_indices[0]]

        return {
            "entry_id": entry_id,
            "title": metadata.get("title", entry_id),
            "tags": metadata.get("tags", []),
            "last_modified": metadata.get("last_modified_source", ""),
            "source_file": metadata.get("source_file", ""),
            "content": full_content,
        }

    except Exception as e:
        logger.error(f"Error getting entry details for {entry_id}: {e}")
        return {"error": f"Error retrieving knowledge entry: {str(e)}"}
