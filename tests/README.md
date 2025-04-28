# MCP Mind Palace Testing Guide

This directory contains tests for the MCP Mind Palace server, which provides long-term memory and human-readable knowledge management capabilities.

## Test Structure

The tests are organized as follows:

```
tests/
  conftest.py               # Common fixtures and utilities
  README.md                 # This guide
  unit/                     # Unit tests for individual components
    knowledge_management/   # Tests for knowledge management components
      test_markdown_parser.py
      test_chunking.py
      test_embedder.py
      test_synchronization.py
    tools/                  # Tests for MCP tool implementations
      test_search.py        # Tests for search functions
      test_proposals.py     # Tests for knowledge proposals/updates
  integration/              # Integration tests for multiple components
    test_tools_integration.py
    test_chroma_integration.py
    test_filesystem_integration.py
  mcp/                      # Tests for MCP server integration
    test_server.py
    test_tool_registration.py
```

## Testing Guidelines

### 1. Unit Testing

The goal of unit tests is to verify that individual components work correctly in isolation. Follow these principles:

- Mock external dependencies (ChromaDB, filesystem, embedding API)
- Test both success and error paths
- Keep tests focused on a single function or feature
- Use descriptive test names that explain what is being tested
- Make liberal use of fixtures to set up test data

### 2. Integration Testing

Integration tests verify that components work together correctly. Guidelines:

- Focus on testing component interactions
- Use real implementations where possible (like real ChromaDB with in-memory storage)
- Mock only expensive or external dependencies (like embedding API calls)
- Test complete user flows and scenarios
- Verify file system operations with temporary directories

### 3. Mock Usage

Use mocks strategically to avoid them becoming a crutch:

- Implement mock-free test paths that test critical components without mocks
- Define and test contracts for interfaces to ensure mocks behave like real implementations
- Always test real file operations in integration tests
- Include "end-to-end" tests that minimize mocking except for external APIs

### 4. Handling Threading and Async Operations

When testing code that involves threading or asynchronous operations, follow these practices to avoid test hangs:

- Use `@pytest.mark.timeout(2)` decorator to prevent tests from hanging indefinitely
- For filesystem watchers (like watchdog), disable real event processing in tests:
  - Set `enable_watchdog=False` when creating `KnowledgeSync` instances for testing
  - Verify mock configuration instead of waiting for actual events
  - Example: `test_file_creation_detection` skips calling `process_file_event` directly
- For thread-based operations, wrap test execution in threads with timeouts:
  ```python
  test_thread = threading.Thread(target=function_to_test)
  test_thread.daemon = True
  test_thread.start()
  test_thread.join(timeout=5)  # Wait max 5 seconds
  assert not test_thread.is_alive(), "Function is taking too long and may be hanging"
  ```
- Use threading locks properly in test code to avoid deadlocks
- For debounced operations that use timers, directly invoke the callback function instead of waiting:
  ```python
  # Instead of waiting for the timer
  handler._debounce_timers[key].cancel()  # Cancel the timer
  handler._debounce_timers[key].function()  # Call the function directly
  ```
- Patch long-running operations with mocks that return immediately
- Avoid testing the actual file watching system; instead test the underlying file event handlers directly

### 5. Running Tests

Run tests using pytest:

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/tools/test_search.py

# Run with coverage
uv run pytest --cov=app tests/
```

## Adding New Tests

When adding new tests:

1. Follow the established directory structure
2. Reuse fixtures from conftest.py when appropriate
3. Create new fixtures for test-specific setup
4. Ensure tests are isolated from environment and can run anywhere
5. Add both positive and negative test cases
6. Test edge cases and error conditions

## Running Tests with External Dependencies

Some tests require external API access or real ChromaDB instances. These are marked with special pytest markers:

- `api_dependent`: Tests that require OpenAI API credentials
- `integration`: Full integration tests between components
- `real_tokenization`: Tests that use real tokenizers but don't need API credentials

By default, all API-dependent and integration tests are skipped. To run them:

```bash
# Set required environment variables
export OPENAI_API_KEY=your_key_here
export EMBEDDING_MODEL=text-embedding-3-small

# Run all tests including API dependent ones
uv run pytest -v

# Run only the API dependent tests
uv run pytest -v -m "api_dependent"

# Run only integration tests
uv run pytest -v -m "integration"
```

Ensure your `.env` file is properly configured with the required credentials before running these tests.

## Test Scenarios for Mind Palace

Test complete flows like:

- Knowledge entry creation → embedding → search → retrieval
- File modification → synchronization → updated embeddings → updated search results
- Knowledge entry proposal → validation → filesystem operations
- Knowledge update suggestion → validation → filesystem operations

## Best Practices

- Test both normal and error flows
- Include tests for edge cases
- Verify input validation and error handling
- Keep test runs fast by minimizing external API calls
- Use parameterized tests for multiple similar test cases
- Include coverage reporting in CI pipeline

# Comprehensive Testing Improvement Plan for MCP Mind Palace

## 1. Fix Chunking Tests
- ✅ Replace mock tokenizer with real tokenizer for at least 50% of tests
- ✅ Add boundary testing with documents exactly at token limits
- ✅ Test with complex markdown featuring tables, code blocks, and nested headers
- ✅ Validate chunk overlap is working correctly with real token counts
- ✅ Add performance testing for large documents (>10,000 tokens)

## 2. Enhance Embedder Tests
- ✅ Add integration test with minimal API key (for CI pipeline)
- ✅ Implement proper error handling tests for API rate limits and failures
- ✅ Test vector normalization with mathematical validation
- ✅ Add cache hit/miss ratio validation
- ✅ Test embedding of long text that requires chunking

## 3. Improve Synchronization Tests
- ✅ Reduce mock usage by 60%, test with actual file operations
- ✅ Add tests for race conditions between file operations
- ✅ Test debouncing behavior with timed file modifications
- ✅ Implement file event ordering tests (create→modify→delete)
- ✅ Validate collection updates after file synchronization

## 4. Add Comprehensive Integration Tests
- ✅ Created end-to-end test for knowledge entry lifecycle:
  ```python
  def test_knowledge_entry_lifecycle():
      # Create entry → embed → search → update → re-embed → delete → verify
  ```
- ✅ Implemented search functionality with real vector database
- ✅ Added validation for proposal and review workflows
- ✅ Created API-dependent tests that can be run with OpenAI API key
- ✅ Added cross-component validation (chunking+embedding+storage)
- ✅ Implemented proper test isolation with unique collection names

## 5. Strengthen Tool Search Tests
- ✅ Replace the mock `MockChromaCollection` implementation with actual ChromaDB in-memory instances
- ✅ Test with real embeddings for at least 60% of tests, using the mark `real_tokenization`
- ✅ Add direct integration with vector similarity math to verify scoring works correctly
- ✅ Test pagination and result ordering correctness with large result sets
- ✅ Validate error handling when ChromaDB returns unusual or empty data structures
- ✅ Test performance with realistic-sized knowledge bases (100+ entries)
- ✅ Add specific tests for the relevance categorization thresholds

## 6. Improve Tool Proposals Tests
- ✅ Refactor tests to use temporary directories instead of mocking file operations
- ✅ Add test cases that validate frontmatter parsing with malformed or edge-case YAML
- ✅ Test with real file synchronization instead of bypassing the KnowledgeSync mechanism
- ✅ Validate file permissions handling and proper error messaging
- ✅ Add file content validation to ensure proposals are written with correct format
- ✅ Create tests that verify proper handling of duplicate entry IDs
- ✅ Test category directory creation logic with various tag combinations

## 7. Enhance Tool Integration Tests
- Create proper end-to-end tests using the actual `KnowledgeSync` process instead of manual data insertion
- Implement a complete workflow test: create entry → embed → search → update → re-embed → delete
- Add API error simulation tests that don't rely on timeouts
- Test with varying ChromaDB collection sizes to validate search performance
- Add real filesystem event tests with proper cleanup
- Implement concurrent operation tests to verify tool thread safety
- Add end-to-end proposal workflow including approval and rejection paths

## 8. Establish Better Test Isolation
- Use unique collection names for each test run
- Implement proper teardown procedures for all fixtures
- Move from hardcoded timeouts to dynamic timeouts based on operation complexity
- Verify tests can run in parallel without interference
- Add CI pipeline stages that verify test isolation
- Create deterministic test environments with reproducible random seeds
- Implement proper database state verification between test steps

## 9. Advanced Error Case Testing
- Test network failure scenarios during embedding generation
- Simulate ChromaDB outages and verify graceful degradation
- Test with corrupted knowledge entries and validate error handling
- Add permissions failure simulation for file operations
- Test quota and rate limit handling for external APIs
- Verify proper logging of errors with contextual information
- Test with malicious input patterns to verify security
