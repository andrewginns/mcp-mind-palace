FROM python:3.12-slim

ARG PORT=8050

WORKDIR /app

# Install uv
RUN pip install uv

# Copy the project files
COPY . .

# Install dependencies
RUN python -m venv .venv
RUN uv pip install -e .

# Set environment variables
ENV PORT=${PORT}
ENV HOST=0.0.0.0
ENV TRANSPORT=sse

EXPOSE ${PORT}

# Command to run the server
CMD ["uv", "run", "run_server.py"] 