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

    # First split by headers to preserve document structure
    header_pattern = r"(^|\n)#{1,6}\s+[^\n]+"
    sections = re.split(header_pattern, content)
    sections = [s for s in sections if s.strip()]

    chunks = []

    for section in sections:
        # Tokenize the section
        section_tokens = encoding.encode(section)

        # If the section fits in a chunk, keep it as is
        if len(section_tokens) <= chunk_size:
            chunks.append(section)
        else:
            # Split the section into paragraphs
            paragraphs = re.split(r"\n\s*\n", section)
            current_chunk_tokens = []
            current_chunk_text = ""

            for paragraph in paragraphs:
                paragraph_tokens = encoding.encode(paragraph)

                # If adding this paragraph would exceed the chunk size, save the current chunk and start a new one
                if len(current_chunk_tokens) + len(paragraph_tokens) > chunk_size:
                    if current_chunk_text:
                        chunks.append(current_chunk_text.strip())

                    # Handle overlap for context preservation
                    if chunk_overlap > 0 and len(current_chunk_tokens) > chunk_overlap:
                        # Get overlap tokens
                        overlap_tokens = current_chunk_tokens[-chunk_overlap:]
                        overlap_text = encoding.decode(overlap_tokens)

                        # Try to find a natural breakpoint (end of sentence)
                        sentence_end = max(
                            overlap_text.rfind("."),
                            overlap_text.rfind("!"),
                            overlap_text.rfind("?"),
                        )

                        if sentence_end != -1:
                            overlap_text = overlap_text[sentence_end + 1 :]

                        # Start new chunk with overlap
                        current_chunk_text = overlap_text + "\n\n" + paragraph + "\n\n"
                        current_chunk_tokens = encoding.encode(current_chunk_text)
                    else:
                        # Start fresh with this paragraph
                        current_chunk_text = paragraph + "\n\n"
                        current_chunk_tokens = paragraph_tokens
                else:
                    # Add paragraph to current chunk
                    current_chunk_text += paragraph + "\n\n"
                    current_chunk_tokens.extend(paragraph_tokens)

            # Don't forget the last chunk
            if current_chunk_text:
                chunks.append(current_chunk_text.strip())

    # Handle very long paragraphs that exceed chunk size
    final_chunks = []
    for chunk in chunks:
        chunk_tokens = encoding.encode(chunk)
        if len(chunk_tokens) <= chunk_size:
            final_chunks.append(chunk)
        else:
            # Split the chunk into smaller pieces (forced splitting)
            for i in range(0, len(chunk_tokens), chunk_size - chunk_overlap):
                end_idx = min(i + chunk_size, len(chunk_tokens))
                sub_chunk = encoding.decode(chunk_tokens[i:end_idx])
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
