import sys
import os
import logging
import socket
import uvicorn
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.main import app

load_dotenv()

logger = logging.getLogger(__name__)

def find_available_port(start_port, max_attempts=10):
    """
    Find an available port starting from start_port.
    If start_port is available, return it. Otherwise try incremental ports.
    
    Args:
        start_port: Port to start checking from
        max_attempts: Maximum number of ports to check
        
    Returns:
        Available port number or None if no ports are available
    """
    for port_offset in range(max_attempts):
        port = start_port + port_offset
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            logger.warning(f"Port {port} is already in use, trying next port...")
        finally:
            sock.close()
    
    return None

if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    preferred_port = int(os.getenv("MCP_SERVER_PORT", "8080"))
    
    max_attempts = 10
    port = find_available_port(preferred_port, max_attempts)
    
    if port is None:
        logger.error(f"Could not find available port after trying {preferred_port} and {max_attempts-1} subsequent ports")
        sys.exit(1)
    
    if port != preferred_port:
        logger.warning(f"Preferred port {preferred_port} was not available. Using port {port} instead.")
    
    logger.info(f"Starting MCP server on {host}:{port}...")
    
    try:
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)
