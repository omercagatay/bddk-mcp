# Chandra2 Primary OCR Backend — Design Spec

**Date:** 2026-04-18
**Goal:** Add Chandra2 as the primary PDF OCR backend ahead of LightOCR, re-backfill 14 mevzuat PDFs, and prove (via a before/after validation report) that formula and image fidelity improves vs. the current LightOCR-primary chain.

**Motivation:** The 2026-04-17 backfill verification found that LightOCR dropped 3 formula regions in `mevzuat_21194` as empty `<form></form>` placeholders. Chandra2 (datalab-to/chandra-ocr-2, 5B params, March 2026 release) reports state-of-the-art on the olmOCR benchmark with markedly better handling of formulas, tables, and multilingual text. LightOCR is retained as the fallback when Chandra2 is unavailable or emits the same failure signature.

**Scope constraint:** Mevzuat PDFs only. `bddk_*` docs are cleanly HTML-parsed and are out of scope.

---

## 1. Architecture

**OCR chain (new, ordered):**

```
Chandra2  →  LightOCR  →  PPStructure  →  markitdown
```

Chandra2 is added in front of the existing 3-tier chain. No existing backend is removed.

**Deployment model — vLLM, backfill-only, on-demand:**

Chandra2 runs under a `chandra_vllm` subprocess launched by `scripts/backfill_mevzuat.py` when `--use-chandra` is passed. vLLM is **never** resident during normal MCP server operation; only LightOCR/PPStructure/markitdown are available to the always-on server. This preserves VRAM (the RTX 5080 Laptop has 16 GB, tight for Chandra2's ~10–14 GB bf16 footprint) and scopes license exposure to explicit backfill runs.

The lifecycle is:

1. `backfill_mevzuat.py --use-chandra` spawns `chandra_vllm --port 8001 --gpu-memory-utilization 0.85 --max-model-len 4096`
2. Script polls `http://localhost:8001/v1/models` up to `CHANDRA_VLLM_HEALTH_TIMEOUT` seconds
3. Backfill loop runs; each doc's PDF is fed to the chain, which calls `ChandraBackend` first
4. On script exit (normal or error), vLLM is sent `SIGTERM`, then `SIGKILL` after a grace period

**Chain-level quality guard (applies to every backend):**

`run_extraction_chain` already requires `len(output) ≥ OCR_MIN_CONTENT_LEN (500)`. This spec extends the success condition: output qualifies if *and only if*

1. `len(output) ≥ 500`, **and**
2. `output` contains no empty `<form></form>` blocks (regex `<form>\s*</form>`)

If either check fails, the chain falls through to the next backend. This guard targets the concrete LightOCR failure mode observed on 2026-04-17 and applies uniformly — if Chandra2 also emits empty `<form>` blocks, the chain will fall through to LightOCR instead of silently accepting degraded output.

**Startup failure policy:**

If vLLM fails to start (OOM, port conflict, model download failure), the script hard-aborts with a clear error and does not touch the database. There is no silent fall-through to a LightOCR-only chain — the whole point of the branch is Chandra2, and a silent fallback would mask the problem.

---

## 2. Components & file layout

**New files:**

- `ocr_backends_chandra.py` — `ChandraBackend` implementing the `OCRBackend` Protocol from `ocr_backends.py`.
  - `is_available()` → HTTP GET `http://localhost:{CHANDRA_VLLM_PORT}/v1/models` with a short timeout, returns True on 200 OK.
  - `extract(pdf_bytes)` → converts PDF to page images via `pdf2image.convert_from_bytes`, iterates pages, sends each as an image content block to `POST /v1/chat/completions` with an OCR prompt, concatenates per-page markdown.
  - Separate module because `ocr_backends.py` is already 313 lines; Chandra adds ~100 more with vLLM HTTP handling.

- `vllm_manager.py` — `VLLMManager` sync context manager.
  - `__enter__` spawns `chandra_vllm` with configured args via `subprocess.Popen`, polls the health endpoint, raises `RuntimeError` on timeout or non-zero exit.
  - `__exit__` sends `SIGTERM`, waits up to a grace period, sends `SIGKILL` if the process is still alive. Handles `KeyboardInterrupt` cleanly.

- `scripts/snapshot_mevzuat_markdown.py` — one-off helper. Reads a list of `document_id`s from `--ids-file`, dumps `{document_id, extraction_method, markdown_content, chars}` to `--out <json>`. Produces the baseline snapshot before the Chandra2 re-run.

- `scripts/compare_ocr_backfill.py` — validation report generator. See Section 3.

- `tests/test_ocr_backends_chandra.py`, `tests/test_vllm_manager.py`, `tests/test_compare_ocr_backfill.py` — unit tests. See Section 4.

- `tests/test_chandra_smoke.py` — GPU-gated integration test.

**Changed files:**

- `ocr_backends.py`
  - Add `_content_ok(text, min_len)` helper: returns True iff `len(text) ≥ min_len` and no empty `<form></form>` matches.
  - `run_extraction_chain` replaces the inline `len(result) < min_len` check with `_content_ok(result, min_len)`, with the error string adjusted to distinguish "too short" from "empty form block".
  - `get_default_backends(include_chandra: bool = False) -> list[OCRBackend]` — when `include_chandra` is True, prepend `ChandraBackend()` to the existing `[LightOCR, PPStructure, markitdown]` list.

- `scripts/backfill_mevzuat.py`
  - New CLI flag `--use-chandra` (default False).
  - When set, wraps the existing `_run` body in `with VLLMManager(...):`, and passes `include_chandra=True` when constructing the backend chain for `DocumentSyncer`.
  - On `VLLMManager.__enter__` raising, logs the error and exits with non-zero status before opening the asyncpg pool.

- `config.py` — add env-var-overridable constants:
  - `CHANDRA_MODEL_NAME` (default `"datalab-to/chandra-ocr-2"`, env `BDDK_CHANDRA_MODEL`)
  - `CHANDRA_VLLM_PORT` (default `8001`, env `BDDK_CHANDRA_VLLM_PORT`) — chosen to avoid the MCP HTTP transport default of 8000
  - `CHANDRA_VLLM_HEALTH_TIMEOUT` (default `300` seconds, env `BDDK_CHANDRA_VLLM_HEALTH_TIMEOUT`)
  - `CHANDRA_GPU_MEMORY_UTILIZATION` (default `0.85`, env `BDDK_CHANDRA_GPU_MEM_UTIL`)
  - `CHANDRA_MAX_MODEL_LEN` (default `4096`, env `BDDK_CHANDRA_MAX_MODEL_LEN`)

- `pyproject.toml` — add `chandra-ocr` to the `[dependency-groups].gpu` list. Pulls in vLLM transitively.

**Unchanged in this spec:**

- `doc_sync.py` — no changes; it already accepts an injectable backend list via `DocumentSyncer`.
- `deps.py` / `server.py` — no changes; Chandra2 is not wired into the always-on server.
- `vector_store.py`, `doc_store.py` — no schema changes (no image embedding).
- `PPStructureBackend` — unchanged; moves from position 2 to position 3 in the chain.

---

## 3. Validation script

`scripts/compare_ocr_backfill.py` answers *"did Chandra2 actually improve things, and what did it miss?"*

**Inputs:**

- `--baseline <path>` — the JSON snapshot produced by `scripts/snapshot_mevzuat_markdown.py` before the Chandra2 run.
- Live database — the current (post-Chandra2) markdown content.
- PDFs on disk — cached from the backfill via `get_pdf_bytes` or re-downloaded lazily.

**Per-doc metrics:**

| Metric | Definition | Detection |
|---|---|---|
| `chars` | `LENGTH(markdown_content)` | SQL |
| `form_drops` | count of empty `<form></form>` blocks | regex `<form>\s*</form>` |
| `latex_markers` | count of `_{…}`, `^{…}`, `\frac`, `\sum`, `\Delta`, `\cdot`, `\sqrt`, `\min`, `\max` | regex |
| `md_image_refs` | count of `![…](…)` + non-empty `<img [^>]+>` | regex |
| `pdf_image_count` | image regions in the source PDF | `pdfplumber.images` summed per page |
| `regression_flag` | True iff `form_drops_after > form_drops_before` OR `latex_markers_after < latex_markers_before` | derived |

**Outputs:**

- `logs/compare_{stamp}.md` — per-doc table, summary totals, regression list, and a "silent drop candidates" list (docs where `pdf_image_count > md_image_refs + latex_markers` — an image in the PDF has no equivalent in the markdown).
- `logs/compare_{stamp}.csv` — same data for spreadsheet analysis.

**Success criteria (decision gate):**

1. `mevzuat_21194`: `form_drops` goes from 3 → 0.
2. No doc has `regression_flag` set.
3. At least one doc shows a measurable improvement — either `latex_markers` or `md_image_refs` strictly higher than baseline, or a silent-drop candidate from the baseline now has a matching markdown reference.

If all three pass → Chandra2 is validated and kept in the backfill chain. If any fail → keep the baseline snapshot, restore the affected rows, file a follow-up.

---

## 4. Testing strategy

**Unit tests (no GPU, no network):**

- `tests/test_ocr_backends_chandra.py`
  - `ChandraBackend.is_available()` returns False when the health URL is unreachable (mock `httpx` raising `ConnectError`).
  - Returns True on 200 OK.
  - `extract()` returns None on HTTP error; returns concatenated per-page markdown on success (mock the chat completions JSON).
  - `extract()` with empty `pdf_bytes` returns None without any HTTP call.

- `tests/test_vllm_manager.py`
  - `VLLMManager.__enter__` raises `RuntimeError` if health probe never succeeds within the timeout (mock `Popen` + `httpx.get` returning 503).
  - `__exit__` sends `SIGTERM`, then `SIGKILL` after the grace period (mock `Popen` with controllable `wait()`).
  - `__exit__` runs even under `KeyboardInterrupt` during the body.

- `tests/test_ocr_backends_chain.py` (extends existing chain tests)
  - Output with `<form>\n</form>` falls through to the next backend even when length ≥ 500.
  - Existing length-only rejection still works (regression guard).

- `tests/test_compare_ocr_backfill.py`
  - Regex metrics correct on fixture markdown (empty `<form>`, `![](..)`, `_{…}`, `<img>`, nested cases).
  - `pdf_image_count` correct on a 2-page fixture PDF with known image count.
  - `regression_flag` True when `form_drops_after > before`; False when equal; True when `latex_markers_after < before`.

**Integration test (GPU-gated, `@pytest.mark.gpu`):**

- `tests/test_chandra_smoke.py` — `VLLMManager` spin-up, `ChandraBackend.extract()` on a 1-page mevzuat fixture PDF, assert output is non-empty and contains expected Turkish tokens. Skipped when `torch.cuda.is_available()` is False. Long-running; not part of the default CI run.

**No model-output mocking:** never mock Chandra2's text output. Mock at the HTTP boundary (for unit tests) or run the real model (for integration).

**Pre-merge gate:** `uv run pytest tests/ -k "ocr or sync or vllm or compare" -v` must be green before the chain change is accepted, to catch any regression in the `<form>` guard against existing fixtures.

---

## 5. Runbook

**Pre-flight:**
```bash
uv sync --group gpu     # picks up chandra-ocr + vllm
nvidia-smi              # confirm ≥14 GB VRAM free
```

**Step 1 — Baseline snapshot (before Chandra2 touches the DB):**
```bash
uv run python scripts/snapshot_mevzuat_markdown.py \
    --ids-file scripts/backfill_ids_20260418.txt \
    --out logs/lightocr_baseline_20260418.json
```

**Step 2 — Chandra2 re-backfill:**
```bash
uv run python scripts/backfill_mevzuat.py \
    --use-chandra --yes \
    --ids-file scripts/backfill_ids_20260418.txt
```
Expected: ~1–2 hours on RTX 5080 at ~0.5–1 pages/sec. Structured log at `logs/backfill_YYYY-MM-DD-HHMM.log`.

**Step 3 — Validation report:**
```bash
uv run python scripts/compare_ocr_backfill.py \
    --baseline logs/lightocr_baseline_20260418.json
```
Produces `logs/compare_{stamp}.md` + `.csv`.

**Step 4 — Decision gate:**
Review the report against the three success criteria in Section 3.

**Step 5 — Outcome:**
- All three pass → Chandra2 kept as primary. Runbook documented in project `CLAUDE.md`. Branch closed.
- Any fail → keep the baseline, restore affected rows from the snapshot via a one-off SQL UPDATE, file a follow-up with the failing metrics attached.

**Out of scope for this plan:**
- Making Chandra2 part of the always-on MCP server.
- Image embedding (deferred — detection only).
- Non-mevzuat docs.
- PPStructure changes.

---

## 6. Open risks

1. **VRAM pressure** — 16 GB is tight for Chandra2 bf16. `CHANDRA_GPU_MEMORY_UTILIZATION=0.85` caps vLLM's allocator; `CHANDRA_MAX_MODEL_LEN=4096` caps KV-cache growth. If OOM is still observed in practice, drop `max-model-len` to 2048 and retry.
2. **License** — Chandra2 is OpenRAIL-M with commercial restrictions. Internal audit use qualifies as research/personal use under the stated exceptions; confirm with legal before any production-path deployment beyond this batch run.
3. **Throughput** — 0.5–1 pages/sec on RTX 5080 makes a full 14-doc run ~1–2 hours. Acceptable for a one-off validation run; if future plans need broader coverage, revisit vLLM deployment mode.
4. **Silent drops** — a backend could emit no placeholder at all for a missed image. The `pdf_image_count` vs `md_image_refs + latex_markers` diff in the validation report is the only guard against this. Not perfect; the report flags candidates but human spot-check is still required for high-stakes docs.
