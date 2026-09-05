FROM python:3.13.14-slim-trixie@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.12.3@sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy

# Copy install metadata and package source before syncing so the packaged
# bddk-mcp / bddk-seed console entry points are installed in the image.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY bddk_mcp/ ./bddk_mcp/
RUN uv sync --frozen --no-dev && rm -rf /root/.cache/uv

# Bundle pre-populated seed data (run `python seed.py export` locally first)
COPY seed_data/ ./seed_data/
# The corpus manifest is Ed25519-signed; the trust anchor must live outside the
# corpus root, so bootstrap in this image passes /app/trust explicitly.
COPY deploy/trust/corpus-signing-public-key.pem ./trust/corpus-signing-public-key.pem

# Pre-download the embedding model at build time so runtime is fully offline.
ENV HF_HOME=/app/model_cache
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base', revision='d13f1b27baf31030b7fd040960d60d909913633f').save('/app/embedding_model')" \
    && rm -rf /app/model_cache
ENV BDDK_EMBEDDING_MODEL_PATH=/app/embedding_model
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV HOME=/tmp
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

# PostgreSQL connection is required and must be injected at runtime.
ENV BDDK_DATABASE_URL=""

# Serving is read-only with respect to corpus/schema lifecycle. Run
# `bddk-mcp migrate` with the schema-owner identity, apply reviewed runtime
# grants, and then run `bddk-mcp bootstrap` with the ingestion identity before
# starting this process.
ENV BDDK_AUTO_SYNC=false

# Default to streamable-http transport for remote deployment
ENV MCP_TRANSPORT=streamable-http
ENV MCP_HOST=0.0.0.0
ENV PORT=8000

# OpenShift may replace this UID with one from the namespace range.  Group 0
# ownership plus group-equals-owner permissions supports either case without
# granting a writable application root at runtime.
RUN chgrp -R 0 /app && chmod -R g=u /app
USER 10001:0

EXPOSE 8000

CMD [".venv/bin/bddk-mcp", "serve"]
