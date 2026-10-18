FROM python:3.13.16-slim-trixie@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.12.3@sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy

# Copy install metadata and package source before syncing so the packaged
# bddk-mcp / bddk-seed console entry points are installed in the image.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY bddk_mcp/ ./bddk_mcp/
# Group-0 ownership and group-equals-owner permissions are applied inside the
# layers that create the files: a separate chgrp/chmod layer would copy the
# whole tree again (overlayfs copy-up) and double the image size.
RUN uv sync --frozen --no-dev && rm -rf /root/.cache/uv && chgrp -R 0 /app && chmod -R g=u /app

# Bundle pre-populated seed data (run `python seed.py export` locally first)
COPY seed_data/ ./seed_data/
# The corpus manifest is Ed25519-signed; the trust anchor must live outside the
# corpus root, so bootstrap in this image passes /app/trust explicitly.
COPY deploy/trust/corpus-signing-public-key.pem ./trust/corpus-signing-public-key.pem

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
RUN chgrp -R 0 /app/seed_data /app/trust && chmod -R g=u /app/seed_data /app/trust && chmod g=u /app
# Persistent-volume mount path (admin editorial drafts). Railway mounts volumes
# root-owned regardless of image ownership, so the admin preview service runs
# with RAILWAY_RUN_UID=0 (Railway's documented remedy for non-root images);
# OpenShift keeps the non-root USER below. The process creates the 0600 SQLite
# file itself (drafts.py refuses anything less private).
RUN mkdir -p /data && chgrp 0 /data && chmod g+rwX /data
# Hardening: the serving process never runs the terminal, login/mount or perl
# tooling of the base image, and their Debian 13 CVEs have no fix
# (supply-chain/policy.json).  Nothing else in the image depends on them;
# dash, coreutils and findutils stay because Railway wraps start commands in
# `sh -c`.
RUN dpkg --purge --force-remove-essential --force-remove-protected --force-depends \
        util-linux mount login bsdutils liblastlog2-2 libsmartcols1 libmount1 libblkid1 libuuid1 \
        ncurses-bin ncurses-base libncursesw6 libtinfo6 libreadline8t64 bash perl-base
USER 10001:0

EXPOSE 8000

CMD [".venv/bin/bddk-mcp", "serve"]
