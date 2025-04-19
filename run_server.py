import sys
import os
import logging
import uvicorn

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.main import app

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting MCP server on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
