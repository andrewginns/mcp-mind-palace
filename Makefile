debug:
	npx @modelcontextprotocol/inspector \
  	uv --directory /Users/aginns/projects/mcp-mind-palace \
  	run run_server.py

force-update-embeddings:
	@echo "Force updating all embeddings in the knowledge base..."
	uv run scripts/force_update_embeddings.py