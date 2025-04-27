# MCP Integration Tests

This directory contains tests that verify the integration between the Mind Palace application and the MCP (Multi-Context Processing) server.

## Test Files

- `test_server.py`: Tests tool integration with the MCP server
- `test_tool_registration.py`: Tests proper registration of MCP tools
- `test_knowledge_sync.py`: Tests knowledge synchronization with MCP integration
- `test_error_handling.py`: Tests error handling and edge cases
- `test_file_interactions.py`: Tests file system operations and event handling
- `test_server_lifecycle.py`: Tests server initialization and shutdown processes

## Running the Tests

Run all MCP tests:

```bash
uv run pytest tests/mcp/
```

Run a specific test file:

```bash
uv run pytest tests/mcp/test_server.py
```

Run with coverage:

```bash
uv run pytest tests/mcp/ --cov=app
```

## Test Coverage

These tests focus on the following aspects of MCP integration:

1. **Tool Registration**: Verifying that the Mind Palace tools are correctly registered with the MCP server
2. **Tool Functionality**: Testing that each tool functions correctly when invoked through MCP
3. **Error Handling**: Ensuring that tools handle errors gracefully
4. **Knowledge Synchronization**: Testing that the knowledge synchronization system works with MCP
5. **File System Integration**: Verifying that file creation, modification, and deletion are properly synchronized
6. **Server Lifecycle**: Testing server initialization, startup, and shutdown processes

## Test Design Principles

1. **Mock External Dependencies**: ChromaDB and embedding services are mocked to isolate tests
2. **Timeout Protection**: All tests use `@pytest.mark.timeout` to prevent test hangs
3. **Isolation**: Tests use temporary directories and fixtures to avoid interfering with actual knowledge base
4. **Error Testing**: Tests include both positive paths and error handling scenarios
5. **Comprehensive Coverage**: Tests cover all aspects of MCP integration

## Fixtures

The tests rely on fixtures defined in `tests/conftest.py`, including:

- `setup_knowledge_dirs`: Creates a temporary knowledge base directory structure
- `patched_knowledge_paths`: Patches app configuration to use temporary directories
- `mock_ctx`: Creates a mock MCP context
- `mock_mcp`: Creates a mock MCP server

## Adding New Tests

When adding new tests, follow these guidelines:

1. Use pytest fixtures for setup and teardown
2. Mock external services like ChromaDB and embedding API
3. Include both success and error cases
4. Add appropriate timeouts to prevent test hangs
5. Verify both synchronous and asynchronous operations
6. Test complete workflows from start to finish 

## Excluded Tests

Some test files in this directory are currently not functioning due to missing modules in the codebase:

1. `test_file_interactions.py` - This test depends on `mcp.file_utils` module which is not present in the current codebase. 
   These tests verify file system operations like getting file contents, metadata, writing files, etc.

2. `test_server_lifecycle.py` - This test depends on `app.server` module which is not implemented in the current codebase.
   These tests verify server initialization, startup, and shutdown processes.

A placeholder test file `test_excluded_modules.py` has been created to document these exclusions. See the main `tests/README.md` 
for more information about excluded tests. 