import pytest
import tiktoken
import time
from app.knowledge_management.chunking import chunk_markdown, create_chunk_metadata


@pytest.mark.real_tokenization
def test_real_tokenizer_short_content():
    """Test chunking with real tokenizer and content shorter than chunk size."""
    content = (
        "# Test Header\n\nThis is a short paragraph that should fit in a single chunk."
    )

    chunks = chunk_markdown(content, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 1
    assert chunks[0] == content


@pytest.mark.real_tokenization
def test_real_tokenizer_header_splitting():
    """Test that content is split at headers when appropriate using real tokenizer."""
    paragraphs = []
    for i in range(5):
        paragraphs.append(f"# Header {i}\n\nThis is content under header {i}. " + 
                         "It contains enough text to make sure we have a substantial paragraph. " * 5)
    
    content = "\n\n".join(paragraphs)
    
    chunks = chunk_markdown(content, chunk_size=50, chunk_overlap=10)
    
    assert len(chunks) >= 2
    
    header_count = sum(1 for chunk in chunks if chunk.strip().startswith("#"))
    assert header_count > 0, "No headers found in chunks"


@pytest.mark.real_tokenization
def test_real_tokenizer_boundary_exact_chunk_size():
    """Test chunking with content exactly at the token limit."""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    content = "This is a test sentence. " * 20
    
    actual_tokens = encoding.encode(content)
    token_count = len(actual_tokens)
    
    assert 90 <= token_count <= 130, f"Expected approximately 100-130 tokens, got {token_count}"
    
    chunks = chunk_markdown(content, chunk_size=token_count, chunk_overlap=20)
    
    assert len(chunks) == 1
    assert chunks[0] == content


@pytest.mark.real_tokenization
def test_real_tokenizer_boundary_just_over_chunk_size():
    """Test chunking with content just over the token limit."""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    base_content = "This is a test sentence. " * 20
    base_tokens = encoding.encode(base_content)
    base_token_count = len(base_tokens)
    
    content = base_content + "Extra words to push over the limit. " * 10
    
    actual_tokens = encoding.encode(content)
    
    chunk_size = base_token_count - 10
    
    assert len(actual_tokens) > chunk_size, f"Expected >{chunk_size} tokens, got {len(actual_tokens)}"
    
    chunks = chunk_markdown(content, chunk_size=chunk_size, chunk_overlap=20)
    
    assert len(chunks) > 1
    
    for i, chunk in enumerate(chunks):
        chunk_tokens = encoding.encode(chunk)
        assert len(chunk_tokens) <= chunk_size, f"Chunk {i} exceeds size limit: {len(chunk_tokens)} tokens"


@pytest.mark.real_tokenization
def test_real_tokenizer_complex_markdown():
    """Test chunking with complex markdown featuring tables, code blocks, and nested headers."""
    content = """# Complex Markdown Document


| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |


```python
def example_function():
    result = 0
    for i in range(10):
        result += i
    return result
```



This is content under a deeply nested header.


Even deeper nesting with some content.


1. First item
2. Second item
   - Nested bullet
   - Another nested bullet
3. Third item

"""

    chunks = chunk_markdown(content, chunk_size=300, chunk_overlap=50)
    
    assert len(chunks) > 0
    
    assert any("| Header 1 | Header 2 | Header 3 |" in chunk for chunk in chunks)
    assert any("```python" in chunk for chunk in chunks)
    assert any("def example_function():" in chunk for chunk in chunks)
    assert any("This is content under a deeply nested header." in chunk for chunk in chunks)
    
    encoding = tiktoken.get_encoding("cl100k_base")
    for i, chunk in enumerate(chunks):
        chunk_tokens = encoding.encode(chunk)
        assert len(chunk_tokens) <= 300, f"Chunk {i} exceeds size limit: {len(chunk_tokens)} tokens"


@pytest.mark.real_tokenization
def test_real_tokenizer_chunk_overlap():
    """Test that chunks have proper overlap with real token counts."""
    sections = []
    for i in range(10):
        sections.append(f"Section {i}: This is the content for section {i} with enough text to make it substantial.")
    
    content = "\n\n".join(sections)
    
    chunk_size = 50
    overlap = 20
    chunks = chunk_markdown(content, chunk_size=chunk_size, chunk_overlap=overlap)
    
    assert len(chunks) > 1
    
    encoding = tiktoken.get_encoding("cl100k_base")
    
    for i in range(len(chunks) - 1):
        current_tokens = encoding.encode(chunks[i])
        next_tokens = encoding.encode(chunks[i + 1])
        
        common_sections = []
        for section_num in range(10):
            section_text = f"Section {section_num}:"
            if section_text in chunks[i] and section_text in chunks[i + 1]:
                common_sections.append(section_num)
        
        assert len(common_sections) > 0 or any(
            section_text in chunks[i] and section_text in chunks[i + 1]
            for section_text in [f"Section {j}:" for j in range(10)]
        ), f"No overlap detected between chunks {i} and {i+1}"


@pytest.mark.real_tokenization
@pytest.mark.slow
def test_real_tokenizer_performance_large_document():
    """Test performance with large documents (>10,000 tokens)."""
    paragraphs = []
    for i in range(500):  # This should create >10,000 tokens
        paragraphs.append(f"Paragraph {i}: This is test paragraph number {i} with enough content to make it substantial. " * 3)
    
    content = "\n\n".join(paragraphs)
    
    encoding = tiktoken.get_encoding("cl100k_base")
    content_tokens = encoding.encode(content)
    assert len(content_tokens) > 10000, f"Test document only has {len(content_tokens)} tokens, expected >10,000"
    
    start_time = time.time()
    chunks = chunk_markdown(content, chunk_size=1000, chunk_overlap=200)
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"Processing time for {len(content_tokens)} tokens: {processing_time:.2f} seconds")
    
    assert len(chunks) > 10
    
    for i, chunk in enumerate(chunks):
        chunk_tokens = encoding.encode(chunk)
        assert len(chunk_tokens) <= 1000, f"Chunk {i} exceeds size limit: {len(chunk_tokens)} tokens"
    
    assert "Paragraph 0:" in chunks[0]
    assert f"Paragraph {499}:" in chunks[-1]
