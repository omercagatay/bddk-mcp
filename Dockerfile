FROM python:3.13.16-slim-trixie@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.12.3@sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy

# Copy install metadata and package source before syncing so the packaged
# bddk-mcp console entry point is installed in the image.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY bddk_mcp/ ./bddk_mcp/
# Group-0 ownership and group-equals-owner permissions are applied inside the
# layers that create the files: a separate chgrp/chmod layer would copy the
# whole tree again (overlayfs copy-up) and double the image size.
RUN uv sync --frozen --no-dev && rm -rf /root/.cache/uv && chgrp -R 0 /app && chmod -R g=u /app

# Pre-download the embedding model at build time so runtime is fully offline.
ENV HF_HOME=/app/model_cache
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base', revision='d13f1b27baf31030b7fd040960d60d909913633f').save('/app/embedding_model')" \
    && rm -rf /app/model_cache && chgrp -R 0 /app/embedding_model && chmod -R g=u /app/embedding_model \
    && find /app/.venv -name __pycache__ -exec chmod -R g=u {} +
ENV BDDK_EMBEDDING_MODEL_PATH=/app/embedding_model
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV HOME=/tmp
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

# Bundle the reviewed seed data (documents.json, decision_cache.json and the
# embeddings/ vector cache). It comes after the model download, so a seed
# change does not invalidate the model layer.
COPY seed_data/ ./seed_data/

# Build the corpus file (sections, chunks, embeddings) into the image. The
# serving process opens it read-only, so the image is the versioned corpus
# artifact: no database, no migration and no pre-deploy step at runtime.
ENV BDDK_CORPUS_PATH=/app/corpus/bddk.sqlite
RUN .venv/bin/bddk-mcp build && chgrp -R 0 /app/corpus && chmod -R g=u /app/corpus

# Default to streamable-http transport for remote deployment
ENV MCP_TRANSPORT=streamable-http
ENV MCP_HOST=0.0.0.0
ENV PORT=8000

# Group 0 ownership plus group-equals-owner permissions lets a platform replace
# the UID without granting a writable application root at runtime.
RUN chgrp -R 0 /app/seed_data && chmod -R g=u /app/seed_data && chmod g=u /app
# Hardening: the serving process never runs the terminal, login/mount or perl
# tooling of the base image, and their Debian 13 CVEs have no fix.
# Nothing else in the image depends on them;
# dash, coreutils and findutils stay because Railway wraps start commands in
# `sh -c`.
RUN dpkg --purge --force-remove-essential --force-remove-protected --force-depends \
        util-linux mount login bsdutils liblastlog2-2 libsmartcols1 libmount1 libblkid1 libuuid1 \
        ncurses-bin ncurses-base libncursesw6 libtinfo6 libreadline8t64 bash perl-base
USER 10001:0

EXPOSE 8000

CMD [".venv/bin/bddk-mcp", "serve"]
