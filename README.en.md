# BDDK MCP Server

[Türkçe](README.md) | [Current status](docs/STATUS.md) | [Documentation index](docs/README.md) | [Security](SECURITY.md)

[![CI](https://github.com/omercagatay/bddk-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/omercagatay/bddk-mcp/actions/workflows/ci.yml)
[![Supply chain evidence](https://github.com/omercagatay/bddk-mcp/actions/workflows/supply-chain.yml/badge.svg)](https://github.com/omercagatay/bddk-mcp/actions/workflows/supply-chain.yml)

This is an MCP server so a bank assistant can answer questions about BDDK decisions, regulations, and bulletins from published sources, not from the model's memory.

> [!IMPORTANT]
> This repository is an engineering beta. It is not legal advice, not a ruling on which article applies to a case, and not bank production acceptance. Counts and limits live in the [current status](docs/STATUS.md). Report vulnerabilities privately under the [security policy](SECURITY.md). Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md).

## What it is for

It is for inspection, compliance, internal audit, and the technical team that connects an assistant. It is not a public consumer app.

An assistant can ask it to:

- Find a decision or regulation record
- Search published text and open an article, principle, or annex
- Show the recorded amendment chain and cross-references
- Ask for the recorded currency signal
- Read a bulletin figure, an announcement, or a short-period digest

The answer must follow the tool output. The tool does not invent a decision number, date, or conclusion. The model should not either.

## What it does not decide

- It does not give a legal opinion or pick the article that applies to a case. The currency tool abstains when it is not sure.
- A signature, a technical check, or a green test is not legal approval or bank acceptance.
- Freshness of every document against the live source is not measured. The corpus is not the whole of BDDK.
- SPK texts are not in this release.
- Some documents still carry formula or OCR warnings. A warning means the text is not clean. Details: [docs/DOCUMENT_QUALITY.md](docs/DOCUMENT_QUALITY.md).
- An admin-console draft does not publish text into the corpus. See the [draft and signature boundary](docs/CORPUS_GOVERNANCE.md#admin-editorial-drafts-and-document-signatures).

Regulation and decision text comes from the locally published copy. Institution, announcement, and bulletin tools, including the announcement and bulletin half of the digest, can call BDDK live.

## Two doors

One process does not open both the reading door and the operator door.

| Door | Selection | Database | Tools |
|---|---|---|---|
| Reading | `BDDK_TOOL_PROFILE=public` | `BDDK_DATABASE_URL` | 17 tools |
| Operator | `BDDK_TOOL_PROFILE=operator` | `BDDK_OPERATOR_DATABASE_URL` | 17 reading tools plus 14 operator tools |

The operator door does not fall back to the public database. A write does not finish inline; it returns a job receipt. Watch it with `get_operator_job`, `list_operator_jobs`, and `cancel_operator_job`. This is not a bank-accepted workflow queue.

Bank setup, identity separation, and publication order stay in the [deployment guide](docs/DEPLOYMENT.md). This page does not repeat that procedure.

## Tools

Reading tools:

- `search_bddk_regulations`: searches titles and catalog metadata. It does not open article text.
- `search_document_store`: searches inside published text.
- `search_bddk_institutions`: searches institutions and may call BDDK live.
- `search_bddk_announcements`: searches announcements and may call BDDK live.
- `get_bddk_document`: returns a stored document page. If extraction left an unsafe artifact, it refuses instead of silently rewriting the text.
- `get_document_history`: shows the stored document history.
- `get_document_section`: asks for an article, principle, paragraph, or annex.
- `search_document_sections`: searches section headings.
- `resolve_regulation_status`: asks for the recorded currency signal. It is not a legal opinion.
- `get_amendment_chain`: shows the amendment chain.
- `get_cross_references`: shows cross-references between documents.
- `get_bddk_bulletin`: weekly bulletin; may call live.
- `get_bddk_bulletin_snapshot`: one dated bulletin view; may call live.
- `get_bddk_monthly`: monthly bulletin; may call live.
- `analyze_bulletin_trends`: reads the direction of a bulletin series; may call live.
- `get_regulatory_digest`: recent decisions, announcements, and bulletin summary. The announcement and bulletin half may be live.
- `compare_bulletin_metrics`: compares bulletin figures across two periods; may call live.

Operator-only additions:

- `check_bddk_updates`: looks for new decisions or announcements.
- `document_store_stats`: reads store counts.
- `bddk_cache_status`: reads catalog-cache status.
- `refresh_bddk_cache`: refreshes the catalog cache.
- `sync_bddk_documents`: starts a document download job.
- `trigger_startup_sync`: starts the startup sync.
- `get_operator_job`: reads one job.
- `list_operator_jobs`: lists jobs.
- `cancel_operator_job`: cancels one job.
- `document_health`: reads document health.
- `health_check`: reads server and publication-gate health.
- `bddk_metrics`: reads counters.
- `backfill_degraded_documents`: reprocesses a degraded extraction.
- `document_quality_report`: reads the quality report.

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

This sketch is local stdio only. Bank HTTP, identity, and network limits are in the [deployment guide](docs/DEPLOYMENT.md).

## Short record for the technical team

The package version is 5.0.1. There is no GitHub release tag for that version.

The canonical operator registry contains 17 public tools plus 14 operator tools, 31 tools total.

Reading: `BDDK_TOOL_PROFILE=public`. Operator: `BDDK_TOOL_PROFILE=operator` and a separate `BDDK_OPERATOR_DATABASE_URL`. Example: `uv run --frozen bddk-mcp serve --profile public`. Requires Python 3.12 or 3.13, `uv`, PostgreSQL 17, `pgvector`, and `unaccent`.

Publication order is `migrate`, then `bootstrap`, then `verify-and-stage-corpus-release`, then `activate-corpus-release`. The old `publish-corpus-release` command is disabled. It is not a fallback for a failed publication. Detail and the bank path: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Corpus boundary: [docs/CORPUS_GOVERNANCE.md](docs/CORPUS_GOVERNANCE.md).

## License

Source code is under the [MIT License](LICENSE). That license does not grant extra rights over BDDK, Official Gazette, or other third-party texts. Boundary: [licensing and provenance](docs/LICENSING_AND_PROVENANCE.md).
