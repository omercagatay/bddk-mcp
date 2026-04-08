FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY *.py ./

# Pre-download the embedding model at build time so runtime is fully offline.
ENV HF_HOME=/app/model_cache
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# PostgreSQL connection (override at runtime)
ENV BDDK_DATABASE_URL=postgresql://bddk:bddk@db:5432/bddk

# Auto-sync documents on first deploy if store is empty
ENV BDDK_AUTO_SYNC=true

# Default to streamable-http transport for remote deployment
ENV MCP_TRANSPORT=streamable-http
ENV PORT=8000

EXPOSE 8000

CMD ["uv", "run", "python", "server.py"]
