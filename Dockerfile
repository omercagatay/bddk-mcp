FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY *.py ./
COPY tests/ tests/

# Create persistent data directory for SQLite document store
RUN mkdir -p /app/data
ENV BDDK_DB_PATH=/app/data/bddk_docs.db

# Auto-sync documents on first deploy if store is empty
ENV BDDK_AUTO_SYNC=true

# Default to streamable-http transport for remote deployment
ENV MCP_TRANSPORT=streamable-http
ENV PORT=8000

EXPOSE 8000

CMD ["uv", "run", "python", "server.py"]
