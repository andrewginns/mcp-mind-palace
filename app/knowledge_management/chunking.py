from typing import List, Dict, Any
import re

def chunk_markdown(content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split Markdown content into chunks of approximately chunk_size characters,
    with chunk_overlap characters of overlap between chunks.
    
    Args:
        content: Markdown content to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    header_pattern = r'(^|\n)#{1,6}\s+[^\n]+'
    sections = re.split(header_pattern, content)
    
    sections = [s for s in sections if s.strip()]
    
    chunks = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            paragraphs = re.split(r'\n\s*\n', section)
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) <= chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    if len(current_chunk) > chunk_overlap:
                        overlap_text = current_chunk[-chunk_overlap:]
                        sentence_end = max(overlap_text.rfind('.'), overlap_text.rfind('!'), overlap_text.rfind('?'))
                        
                        if sentence_end != -1:
                            overlap_text = overlap_text[sentence_end+1:]
                        
                        current_chunk = overlap_text + paragraph + "\n\n"
                    else:
                        current_chunk = paragraph + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
    
    return chunks

def create_chunk_metadata(
    chunk_index: int, 
    source_file: str, 
    entry_id: str, 
    title: str, 
    tags: List[str],
    last_modified: Any,
    content_hash: str
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
    
    if hasattr(last_modified, 'isoformat'):  # Check if it's a date-like object
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
        "content_hash": content_hash
    }
