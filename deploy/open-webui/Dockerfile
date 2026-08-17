FROM ghcr.io/open-webui/open-webui@sha256:6a773e5c3a246b65cbe74ce942b294292c0e5f81c138f703d111bc162f7d7c3d

LABEL org.opencontainers.image.title="Open WebUI with refreshing MCP OAuth transport"
LABEL org.opencontainers.image.description="Pinned Open WebUI 0.11.0 with per-request MCP OAuth refresh"
LABEL org.opencontainers.image.version="0.11.0-mcp-oauth-refresh.1"
LABEL io.bankreg.openwebui.patch.sha256="ef6a9cc47167519da9d52ada67fe56374c5c608f363e9f367acbf2b76560dd39"
LABEL io.bankreg.openwebui.base.revision="01f4282f1ffe0d6212f58d3afbeae21fffd0c4be"
LABEL io.bankreg.openwebui.base.digest="sha256:6a773e5c3a246b65cbe74ce942b294292c0e5f81c138f703d111bc162f7d7c3d"

COPY patches/0001-mcp-oauth-refresh.patch /tmp/0001-mcp-oauth-refresh.patch

RUN cd /app \
    && git apply --check --recount /tmp/0001-mcp-oauth-refresh.patch \
    && git apply --recount /tmp/0001-mcp-oauth-refresh.patch \
    && python -m compileall -q \
        /app/backend/open_webui/utils/mcp/client.py \
        /app/backend/open_webui/utils/middleware.py \
        /app/backend/open_webui/utils/oauth.py \
    && rm /tmp/0001-mcp-oauth-refresh.patch
