import re
import tiktoken
from typing import Any, Dict, List


def chunk_markdown(
    content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    model_name: str = "text-embedding-3-small",
) -> List[str]:
    """
    Split Markdown content into chunks based on tokens for optimal embedding with text-embedding-3-small.

    The text-embedding-3-small model has a maximum context length of 8191 tokens, but
    for better semantic retrieval, we use a smaller chunk size with overlap between chunks.

    Args:
        content: Markdown content to chunk
        chunk_size: Target size of each chunk in tokens (default is 1000 tokens)
        chunk_overlap: Number of tokens to overlap between chunks (default is 200 tokens, 20% of chunk_size)
        model_name: The embedding model being used (to select the appropriate tokenizer)

    Returns:
        List of text chunks optimized for the specified embedding model
    """
    # Get the appropriate tokenizer
    encoding_name = (
        "cl100k_base"  # Default for text-embedding-3-small and text-embedding-3-large
    )
    encoding = tiktoken.get_encoding(encoding_name)

    # Ensure reasonable chunk size (max 8000 to stay under the 8191 token limit with some margin)
    if chunk_size > 8000:
        chunk_size = 8000

    # Check if content fits in a single chunk
    content_tokens = encoding.encode(content)
    if len(content_tokens) <= chunk_size:
        return [content]

    # Split by headers to preserve document structure
    header_pattern = r"(^|\n)(#{1,6}\s+[^\n]+)"
    sections = re.split(header_pattern, content)

    # Regroup the sections properly - every 3 items form a group (match1, header, content)
    chunks = []
    for i in range(1, len(sections), 3):
        if i + 1 < len(sections):
            header = sections[i]
            content_section = sections[i + 1] if i + 1 < len(sections) else ""

            # Create a section with header
            full_section = f"{header}\n{content_section}"
            section_tokens = encoding.encode(full_section)

            # If section fits in chunk, add it directly
            if len(section_tokens) <= chunk_size:
                chunks.append(full_section)
            else:
                # Split this section
                paragraphs = re.split(r"\n\s*\n", full_section)
                current_chunk = paragraphs[0]  # Start with header
                current_tokens = encoding.encode(current_chunk)

                for paragraph in paragraphs[1:]:
                    paragraph_tokens = encoding.encode(paragraph)

                    # Check if adding paragraph exceeds chunk size
                    if len(current_tokens) + len(paragraph_tokens) > chunk_size:
                        chunks.append(current_chunk)
                        # Start new chunk with header for context
                        current_chunk = f"{header}\n\n{paragraph}"
                        current_tokens = encoding.encode(current_chunk)
                    else:
                        current_chunk += f"\n\n{paragraph}"
                        current_tokens.extend(paragraph_tokens)

                # Add final chunk from this section
                if current_chunk:
                    chunks.append(current_chunk)

    # If no chunks were created (which might happen if the regex splitting didn't work properly)
    # fall back to a simpler chunking approach
    if not chunks:
        # Split by paragraphs
        paragraphs = re.split(r"\n\s*\n", content)
        current_chunk = ""
        current_tokens = []

        for paragraph in paragraphs:
            paragraph_tokens = encoding.encode(paragraph)

            # Check if adding paragraph exceeds chunk size
            if (
                current_tokens
                and len(current_tokens) + len(paragraph_tokens) > chunk_size
            ):
                chunks.append(current_chunk)

                # Handle overlap
                if chunk_overlap > 0 and len(current_tokens) > chunk_overlap:
                    overlap_tokens = current_tokens[-chunk_overlap:]
                    overlap_text = encoding.decode(overlap_tokens)
                    current_chunk = f"{overlap_text}\n\n{paragraph}"
                else:
                    current_chunk = paragraph

                current_tokens = encoding.encode(current_chunk)
            else:
                if current_chunk:
                    current_chunk += f"\n\n{paragraph}"
                else:
                    current_chunk = paragraph
                current_tokens.extend(paragraph_tokens)

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

    # Process any remaining chunks that are still too large
    final_chunks = []
    for chunk in chunks:
        chunk_tokens = encoding.encode(chunk)
        if len(chunk_tokens) <= chunk_size:
            final_chunks.append(chunk)
        else:
            # Handle very long chunks by forced splitting
            for i in range(0, len(chunk_tokens), chunk_size - chunk_overlap):
                end_idx = min(i + chunk_size, len(chunk_tokens))
                sub_chunk = encoding.decode(chunk_tokens[i:end_idx])
                if sub_chunk.strip():
                    final_chunks.append(sub_chunk)

    return final_chunks


def create_chunk_metadata(
    chunk_index: int,
    source_file: str,
    entry_id: str,
    title: str,
    tags: List[str],
    last_modified: Any,
    content_hash: str,
) -> Dict[str, Any]:
    """
    Create metadata for a chunk to be stored in ChromaDB.

    Args:
        chunk_index: Index of the chunk within the source document
        source_file: Path to the source Markdown file
        entry_id: Unique identifier of the knowledge entry
        title: Title of the knowledge entry
        tags: List of tags associated with the knowledge entry
        last_modified: Last modification timestamp of the source file
        content_hash: Hash of the source file's content

    Returns:
        Dictionary containing the metadata
    """
    tags_str = ",".join(tags) if tags else ""

    if hasattr(last_modified, "isoformat"):  # Check if it's a date-like object
        last_modified_str = last_modified.isoformat()
    else:
        last_modified_str = str(last_modified)

    return {
        "chunk_index": chunk_index,
        "source_file": source_file,
        "entry_id": entry_id,
        "title": title,
        "tags": tags_str,
        "last_modified_source": last_modified_str,
        "content_hash": content_hash,
    }
