# BDDK MCP Server

[Türkçe](README.md) | [Current status](docs/STATUS.md) | [Documentation index](docs/README.md) | [Security](SECURITY.md)

[![CI](https://github.com/omercagatay/bddk-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/omercagatay/bddk-mcp/actions/workflows/ci.yml)

This is an MCP server so a bank assistant can answer questions about BDDK decisions, regulations, and bulletins from published sources, not from the model's memory.

> [!IMPORTANT]
> This repository is an engineering beta. It is not legal advice, not a ruling on which article applies to a case, and not bank production acceptance. Counts and limits live in the [current status](docs/STATUS.md). Report vulnerabilities privately under the [security policy](SECURITY.md). Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md).

## What it is for

It is for inspection, compliance, internal audit, and the technical team that connects an assistant. It is not a public consumer app.

An assistant can ask it to:

- Find a decision or regulation record
- Search published text and open an article, principle, or annex
- Verify that a quotation occurs character for character in the stored text
- Read a bulletin figure, an announcement, or a short-period digest

The answer must follow the tool output. The tool does not invent a decision number, date, or conclusion. The model should not either.

## What it does not decide

- It does not give a legal opinion or pick the article that applies to a case. It does not say whether a provision is in force today; verify that against the official source.
- A technical check or a green test is not legal approval or bank acceptance.
- Freshness of every document against the live source is not measured. The corpus is not the whole of BDDK.
- SPK texts are not in this release.
- Some documents still carry formula or OCR warnings. A warning means the text is not clean. Details: [docs/DOCUMENT_QUALITY.md](docs/DOCUMENT_QUALITY.md).

Regulation and decision text comes from the locally published copy. Institution, announcement, and bulletin tools, including the announcement and bulletin half of the digest, can call BDDK live.

## Tools

- `search_bddk_regulations`: searches titles and catalog metadata. It does not open article text.
- `search_document_store`: searches inside published text.
- `search_bddk_institutions`: searches institutions and may call BDDK live.
- `search_bddk_announcements`: searches announcements and may call BDDK live.
- `get_bddk_document`: returns a stored document page. If extraction left an unsafe artifact, it refuses instead of silently rewriting the text.
- `get_document_history`: shows the stored document history.
- `get_document_section`: asks for an article (including inserted articles such as 66/A), a temporary or additional article, a principle, paragraph, annex, or lettered item (bent); with `quotation` it verifies the quote character for character against the stored text.
- `search_document_sections`: searches section headings.
- `get_bddk_bulletin`: weekly bulletin; may call live.
- `get_bddk_bulletin_snapshot`: one dated bulletin view; may call live.
- `get_bddk_monthly`: monthly bulletin; may call live.
- `analyze_bulletin_trends`: reads the direction of a bulletin series; may call live.
- `get_regulatory_digest`: recent decisions, announcements, and bulletin summary. The announcement and bulletin half may be live.
- `compare_bulletin_metrics`: compares bulletin figures across two periods; may call live.

The server only reads. Database preparation is separate: `bddk-mcp migrate` creates the schema, `bddk-seed import` loads the documents in `seed_data/`, and `bddk-mcp sync` downloads documents from the source. Details: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Connecting an assistant

The technical team makes the connection. The checked-in [`.mcp.json`](.mcp.json) is a portable local stdio example. Supply `BDDK_DATABASE_URL` from the platform secret store. Fixed local passwords are not the bank path.

<a id="codex-configuration"></a>

### Codex

For Codex, add this to `~/.codex/config.toml` or `.codex/config.toml` inside a trusted checkout, and set `cwd` to that checkout:

```toml
[mcp_servers.bddk]
command = "uv"
args = ["run", "--frozen", "bddk-mcp"]
cwd = "/absolute/path/to/bddk-mcp"
env_vars = ["BDDK_DATABASE_URL"]
startup_timeout_sec = 30
tool_timeout_sec = 60
```

This sketch is local stdio only. HTTP, identity, and network limits are in the [deployment guide](docs/DEPLOYMENT.md).

## Short record for the technical team

The package version is 5.0.1. There is no GitHub release tag for that version.

The tool list lives in `bddk_mcp/tools/registry.py`: 14 tools, one surface. Example: `uv run --frozen bddk-mcp serve`. Requires Python 3.12 or 3.13, `uv`, PostgreSQL 17, `pgvector`, and `unaccent`.

Setup order is `migrate`, then `bddk-seed import`, optionally `sync`. The signed release workflow, multi-identity database roles, operator job queue, admin console, telemetry and the legal-status tools that had no data path (`resolve_regulation_status`, `get_amendment_chain`, `get_cross_references`) were removed on 2026-11-09 and remain in Git history (tag `archive/pre-simplify-2026-11-09`).

## License

Versions after commit `aa62301` (2 October 2026) are distributed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Personal, research, educational, and noncommercial-organization use is permitted. **Any commercial use — including use by banks, financial institutions, consultancies, or other companies in the course of their business — is not permitted** without a separate written license from the copyright holder. The exact terms are in [LICENSE](LICENSE).

**Commercial licensing:** contact me [via GitHub](https://github.com/omercagatay).

Versions up to and including commit `aa62301` were published under the MIT License; that grant remains valid for those earlier copies only. Regulatory-source documents and other third-party data may have separate provenance or reuse conditions; the code license grants no extra rights over them. See [Licensing and Provenance](docs/LICENSING_AND_PROVENANCE.md).
