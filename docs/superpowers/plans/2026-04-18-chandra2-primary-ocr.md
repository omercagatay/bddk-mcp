# Chandra2 Primary OCR Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Chandra2 (datalab-to/chandra-ocr-2) as the primary PDF OCR backend ahead of LightOCR, re-backfill 14 mevzuat PDFs under the new chain, and ship a validation report that proves formula and image fidelity vs. the 2026-04-17 LightOCR baseline.

**Architecture:** New `ChandraBackend` implements the existing `OCRBackend` Protocol in a separate module `ocr_backends_chandra.py`. It talks HTTP to a vLLM OpenAI-compatible server managed by a new `VLLMManager` context manager. The chain-level quality guard is extended with an empty-`<form>` rejector that applies uniformly to every backend. Chandra2 runs only during explicit `backfill_mevzuat.py --use-chandra` invocations; the always-on MCP server keeps its existing 3-tier chain and is not touched.

**Tech Stack:** Python 3.11+, httpx (already in deps), subprocess for vLLM lifecycle, pdf2image (already in deps), pdfplumber (new, dev group), chandra-ocr (new, gpu group — pulls in vLLM). Tests: pytest + `unittest.mock`.

**Project conventions:**
- Per user memory ("no git for internal work"): skip git add/commit/branch steps. Verification = tests pass + manual smoke check after the runbook.
- Ruff line length 120. Existing Protocol-based backend contract MUST NOT be broken — `tests/test_doc_sync.py` and `tests/test_ocr_backends.py` must stay green throughout.

**Spec:** `docs/superpowers/specs/2026-04-18-chandra2-primary-ocr-design.md`.

---

## File Structure

**New files:**
- `ocr_backends_chandra.py` — `ChandraBackend` class (HTTP client against the vLLM server). Separate module because `ocr_backends.py` is already 313 lines.
- `vllm_manager.py` — `VLLMManager` context manager owning the `chandra_vllm` subprocess lifecycle.
- `scripts/snapshot_mevzuat_markdown.py` — one-off helper that dumps current markdown for a set of doc IDs as JSON. Produces the baseline snapshot.
- `scripts/compare_ocr_backfill.py` — validation report generator (md + csv), reads baseline JSON + live DB + cached PDFs.
- `tests/test_ocr_backends_chandra.py` — unit tests for `ChandraBackend`.
- `tests/test_vllm_manager.py` — unit tests for `VLLMManager`.
- `tests/test_compare_ocr_backfill.py` — unit tests for the compare script's metrics.
- `tests/test_chandra_smoke.py` — GPU-gated integration test (marked `gpu`, skipped in CI).

**Modified files:**
- `ocr_backends.py` — add `_content_ok(text, min_len)` helper; `run_extraction_chain` uses it; `get_default_backends(include_chandra: bool = False)` prepends Chandra when requested.
- `tests/test_ocr_backends.py` — add chain test for the `<form>` guard; add `include_chandra` test for the factory.
- `scripts/backfill_mevzuat.py` — add `--use-chandra` flag that wraps the run in `VLLMManager` and passes `include_chandra=True`.
- `config.py` — add `CHANDRA_*` constants (model name, port, health timeout, GPU util, max model len).
- `pyproject.toml` — add `chandra-ocr` to `[dependency-groups].gpu`; add `pdfplumber` to `[dependency-groups].dev`.

**Unchanged:**
- `doc_sync.py`, `deps.py`, `server.py`, `doc_store.py`, `vector_store.py` — Chandra2 is not wired into the always-on server; no schema changes (no image embedding, per spec scope).

---

## Task 1: Add Chandra2 config constants

**Files:**
- Modify: `config.py` (append to the `# -- OCR extraction backends --` section, before `# -- pgvector --`)
- Test: `tests/test_config.py` (add a new test class at the end)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
class TestChandraConfig:
    def test_chandra_model_name_default(self):
        import importlib

        import config

        importlib.reload(config)
        assert config.CHANDRA_MODEL_NAME == "datalab-to/chandra-ocr-2"

    def test_chandra_vllm_port_default(self, monkeypatch):
        monkeypatch.delenv("BDDK_CHANDRA_VLLM_PORT", raising=False)
        import importlib

        import config

        importlib.reload(config)
        assert config.CHANDRA_VLLM_PORT == 8001

    def test_chandra_env_override(self, monkeypatch):
        monkeypatch.setenv("BDDK_CHANDRA_VLLM_PORT", "9001")
        monkeypatch.setenv("BDDK_CHANDRA_GPU_MEM_UTIL", "0.7")
        monkeypatch.setenv("BDDK_CHANDRA_MAX_MODEL_LEN", "2048")
        monkeypatch.setenv("BDDK_CHANDRA_VLLM_HEALTH_TIMEOUT", "60")
        import importlib

        import config

        importlib.reload(config)
        assert config.CHANDRA_VLLM_PORT == 9001
        assert config.CHANDRA_GPU_MEMORY_UTILIZATION == 0.7
        assert config.CHANDRA_MAX_MODEL_LEN == 2048
        assert config.CHANDRA_VLLM_HEALTH_TIMEOUT == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::TestChandraConfig -v`
Expected: FAIL with `AttributeError: module 'config' has no attribute 'CHANDRA_MODEL_NAME'`.

- [ ] **Step 3: Add constants to `config.py`**

Append this block inside `config.py` right after the `OCR_MIN_CONTENT_LEN` line and before `# -- pgvector --`:

```python
# -- Chandra2 (primary OCR, backfill-only, vLLM-served) ----------------------

CHANDRA_MODEL_NAME = os.environ.get("BDDK_CHANDRA_MODEL", "datalab-to/chandra-ocr-2")
CHANDRA_VLLM_PORT = int(os.environ.get("BDDK_CHANDRA_VLLM_PORT", "8001"))
CHANDRA_VLLM_HEALTH_TIMEOUT = int(os.environ.get("BDDK_CHANDRA_VLLM_HEALTH_TIMEOUT", "300"))
CHANDRA_GPU_MEMORY_UTILIZATION = float(os.environ.get("BDDK_CHANDRA_GPU_MEM_UTIL", "0.85"))
CHANDRA_MAX_MODEL_LEN = int(os.environ.get("BDDK_CHANDRA_MAX_MODEL_LEN", "4096"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py::TestChandraConfig -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify full config tests still pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all pass.

---

## Task 2: Add `chandra-ocr` and `pdfplumber` dependencies

**Files:**
- Modify: `pyproject.toml:24-36`

- [ ] **Step 1: Edit `pyproject.toml`**

Replace the `[dependency-groups]` block:

```toml
[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-asyncio>=0.24",
    "pytest-httpx>=0.30",
    "ruff>=0.8",
    "pdfplumber>=0.11",
]
gpu = [
    "torch>=2.0",
    "torchvision>=0.15",
    "paddleocr>=2.10",
    "paddlepaddle-gpu>=2.6; platform_system!='Darwin' and python_version<'3.13'",
    "chandra-ocr>=0.1",
]
```

- [ ] **Step 2: Resolve dependencies**

Run: `uv sync --dev`
Expected: lock file updates, `pdfplumber` installed. The `gpu` group is NOT pulled in by `--dev`; that's intentional for local test iteration.

- [ ] **Step 3: Smoke-import pdfplumber**

Run: `uv run python -c "import pdfplumber; print(pdfplumber.__version__)"`
Expected: prints a version string with no error.

- [ ] **Step 4: Confirm existing tests still green**

Run: `uv run pytest tests/ -m 'not gpu' -q`
Expected: pre-existing pass count, no new failures.

---

## Task 3: Chain-level `<form>` guard in `run_extraction_chain`

**Files:**
- Modify: `ocr_backends.py:272-307`
- Test: `tests/test_ocr_backends.py` (append new chain test at the end of `TestExtractionChain`)

- [ ] **Step 1: Write the failing test**

Append to the `TestExtractionChain` class in `tests/test_ocr_backends.py`:

```python
    def test_empty_form_block_triggers_fallback(self):
        degraded = "lorem " * 200 + "\n<form>\n</form>\n" + "ipsum " * 200
        assert len(degraded) >= 500  # sanity: would pass length check alone
        b1 = _FakeBackend("first", degraded)
        b2 = _FakeBackend("second", "y" * 600)
        attempt = run_extraction_chain(b"pdf", [b1, b2], min_len=500)
        assert attempt.backend == "second"

    def test_form_guard_accepts_non_empty_form(self):
        good = "lorem " * 200 + "\n<form>E = mc^2</form>\n" + "ipsum " * 200
        b1 = _FakeBackend("first", good)
        b2 = _FakeBackend("second", "y" * 600)
        attempt = run_extraction_chain(b"pdf", [b1, b2], min_len=500)
        assert attempt.backend == "first"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ocr_backends.py::TestExtractionChain::test_empty_form_block_triggers_fallback -v`
Expected: FAIL — current chain accepts the degraded output because length ≥ 500.

- [ ] **Step 3: Implement `_content_ok` helper + use in chain**

In `ocr_backends.py`, add above `run_extraction_chain`:

```python
import re

_EMPTY_FORM_RE = re.compile(r"<form>\s*</form>", re.IGNORECASE)


def _content_ok(text: str, min_len: int) -> tuple[bool, str]:
    """Chain-level quality guard.

    Returns (passes, reason). reason is '' when passes is True.
    """
    if len(text) < min_len:
        return False, f"output too short ({len(text)} < {min_len})"
    if _EMPTY_FORM_RE.search(text):
        return False, "contains empty <form></form> block"
    return True, ""
```

Then replace the length check inside `run_extraction_chain`:

```python
        if result is None:
            errors.append(f"{backend.name}: returned None")
            continue
        ok, reason = _content_ok(result, min_len)
        if not ok:
            errors.append(f"{backend.name}: {reason}")
            continue
```

(Remove the old `if len(result) < min_len:` block — `_content_ok` subsumes it.)

- [ ] **Step 4: Run both new tests to verify they pass**

Run: `uv run pytest tests/test_ocr_backends.py::TestExtractionChain -v`
Expected: all class tests PASS, including the new two.

- [ ] **Step 5: Full `ocr_backends` tests green**

Run: `uv run pytest tests/test_ocr_backends.py -v`
Expected: all PASS — the existing `test_short_output_triggers_fallback` still exercises the length path.

---

## Task 4: `VLLMManager` context manager

**Files:**
- Create: `vllm_manager.py`
- Test: `tests/test_vllm_manager.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_vllm_manager.py`:

```python
"""Unit tests for VLLMManager subprocess lifecycle."""

import signal
from unittest.mock import MagicMock, patch

import httpx
import pytest


class TestVLLMManagerHealthProbe:
    def test_enter_raises_when_health_never_succeeds(self):
        from vllm_manager import VLLMManager

        proc = MagicMock()
        proc.poll.return_value = None
        with patch("vllm_manager.subprocess.Popen", return_value=proc):
            with patch("vllm_manager.httpx.get", side_effect=httpx.ConnectError("nope")):
                with patch("vllm_manager.time.monotonic", side_effect=[0.0, 0.5, 1.0, 2.1]):
                    with patch("vllm_manager.time.sleep"):
                        mgr = VLLMManager(
                            model="datalab-to/chandra-ocr-2",
                            port=8001,
                            health_timeout=2,
                            gpu_memory_utilization=0.85,
                            max_model_len=4096,
                        )
                        with pytest.raises(RuntimeError, match="health"):
                            mgr.__enter__()
        proc.terminate.assert_called()

    def test_enter_raises_when_process_exits_early(self):
        from vllm_manager import VLLMManager

        proc = MagicMock()
        proc.poll.return_value = 1  # exited with code 1
        proc.stderr.read.return_value = b"OOM"
        with patch("vllm_manager.subprocess.Popen", return_value=proc):
            mgr = VLLMManager(
                model="x",
                port=8001,
                health_timeout=2,
                gpu_memory_utilization=0.85,
                max_model_len=4096,
            )
            with pytest.raises(RuntimeError, match="exited"):
                mgr.__enter__()

    def test_enter_returns_self_when_health_ok(self):
        from vllm_manager import VLLMManager

        proc = MagicMock()
        proc.poll.return_value = None
        response = MagicMock()
        response.status_code = 200
        with patch("vllm_manager.subprocess.Popen", return_value=proc):
            with patch("vllm_manager.httpx.get", return_value=response):
                with patch("vllm_manager.time.sleep"):
                    mgr = VLLMManager(
                        model="x",
                        port=8001,
                        health_timeout=60,
                        gpu_memory_utilization=0.85,
                        max_model_len=4096,
                    )
                    got = mgr.__enter__()
                    assert got is mgr


class TestVLLMManagerShutdown:
    def test_exit_sends_sigterm_then_sigkill(self):
        from vllm_manager import VLLMManager

        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.side_effect = [Exception("timeout"), 0]  # first wait times out
        with patch("vllm_manager.subprocess.Popen", return_value=proc):
            with patch("vllm_manager.httpx.get", return_value=MagicMock(status_code=200)):
                with patch("vllm_manager.time.sleep"):
                    mgr = VLLMManager(
                        model="x", port=8001, health_timeout=60,
                        gpu_memory_utilization=0.85, max_model_len=4096,
                    )
                    mgr.__enter__()
                    mgr.__exit__(None, None, None)
        proc.send_signal.assert_any_call(signal.SIGTERM)
        proc.kill.assert_called_once()

    def test_exit_noop_if_process_already_exited(self):
        from vllm_manager import VLLMManager

        proc = MagicMock()
        # first poll() during enter: None (alive); later polls after exit: 0
        poll_vals = iter([None, 0, 0])
        proc.poll.side_effect = lambda: next(poll_vals)
        with patch("vllm_manager.subprocess.Popen", return_value=proc):
            with patch("vllm_manager.httpx.get", return_value=MagicMock(status_code=200)):
                with patch("vllm_manager.time.sleep"):
                    mgr = VLLMManager(
                        model="x", port=8001, health_timeout=60,
                        gpu_memory_utilization=0.85, max_model_len=4096,
                    )
                    mgr.__enter__()
                    mgr.__exit__(None, None, None)
        proc.send_signal.assert_not_called()
        proc.kill.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vllm_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vllm_manager'`.

- [ ] **Step 3: Implement `vllm_manager.py`**

Create `vllm_manager.py`:

```python
"""Subprocess lifecycle manager for the Chandra2 vLLM server.

Spawns `chandra_vllm --port ...` as a child process, polls the OpenAI-compatible
`/v1/models` endpoint until it returns 200, and on exit sends SIGTERM (then
SIGKILL after a grace period).

Usage:
    with VLLMManager(model=..., port=8001, health_timeout=300, ...):
        # chandra_vllm is up; HTTP calls to http://localhost:8001 will work
        ...
    # on exit: process is terminated
"""

from __future__ import annotations

import logging
import signal
import subprocess
import time

import httpx

logger = logging.getLogger(__name__)

_SHUTDOWN_GRACE_SECONDS = 15.0
_HEALTH_POLL_INTERVAL = 2.0


class VLLMManager:
    """Sync context manager owning a chandra_vllm subprocess."""

    def __init__(
        self,
        model: str,
        port: int,
        health_timeout: int,
        gpu_memory_utilization: float,
        max_model_len: int,
    ) -> None:
        self._model = model
        self._port = port
        self._health_timeout = health_timeout
        self._gpu_util = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "VLLMManager":
        cmd = [
            "chandra_vllm",
            "--model", self._model,
            "--port", str(self._port),
            "--gpu-memory-utilization", str(self._gpu_util),
            "--max-model-len", str(self._max_model_len),
        ]
        logger.info("Launching vLLM: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._await_health()
        return self

    def _await_health(self) -> None:
        assert self._proc is not None
        url = f"http://localhost:{self._port}/v1/models"
        deadline = time.monotonic() + self._health_timeout
        while time.monotonic() < deadline:
            rc = self._proc.poll()
            if rc is not None:
                stderr_tail = b""
                if self._proc.stderr is not None:
                    try:
                        stderr_tail = self._proc.stderr.read()[-2000:]
                    except Exception:
                        pass
                raise RuntimeError(
                    f"chandra_vllm exited early with code {rc}. stderr tail: {stderr_tail!r}"
                )
            try:
                resp = httpx.get(url, timeout=3.0)
                if resp.status_code == 200:
                    logger.info("vLLM healthy on port %d", self._port)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(_HEALTH_POLL_INTERVAL)
        self._terminate()
        raise RuntimeError(
            f"chandra_vllm health probe never succeeded within {self._health_timeout}s"
        )

    def __exit__(self, exc_type, exc, tb) -> None:
        self._terminate()

    def _terminate(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            return
        logger.info("Sending SIGTERM to vLLM (pid=%d)", self._proc.pid)
        self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
            return
        except Exception:
            pass
        logger.warning("vLLM did not exit after SIGTERM; sending SIGKILL")
        self._proc.kill()
        try:
            self._proc.wait(timeout=5.0)
        except Exception:
            logger.error("vLLM process did not die after SIGKILL")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vllm_manager.py -v`
Expected: all 4 tests PASS.

---

## Task 5: `ChandraBackend` (HTTP client)

**Files:**
- Create: `ocr_backends_chandra.py`
- Test: `tests/test_ocr_backends_chandra.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ocr_backends_chandra.py`:

```python
"""Unit tests for ChandraBackend (HTTP client against vLLM)."""

from unittest.mock import MagicMock, patch

import httpx
import pytest


class TestChandraBackend:
    def test_name(self):
        from ocr_backends_chandra import ChandraBackend

        assert ChandraBackend().name == "chandra2"

    def test_is_available_false_on_connection_error(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        with patch("ocr_backends_chandra.httpx.get", side_effect=httpx.ConnectError("down")):
            assert backend.is_available() is False

    def test_is_available_false_on_non_200(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        resp = MagicMock(status_code=503)
        with patch("ocr_backends_chandra.httpx.get", return_value=resp):
            assert backend.is_available() is False

    def test_is_available_true_on_200(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        resp = MagicMock(status_code=200)
        with patch("ocr_backends_chandra.httpx.get", return_value=resp):
            assert backend.is_available() is True

    def test_extract_empty_returns_none(self):
        from ocr_backends_chandra import ChandraBackend

        assert ChandraBackend().extract(b"") is None

    def test_extract_concatenates_per_page_markdown(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        fake_images = [MagicMock(), MagicMock()]  # 2 pages
        completions = [
            MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "# Page 1"}}]},
            ),
            MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "# Page 2"}}]},
            ),
        ]
        with patch("ocr_backends_chandra.convert_from_bytes", return_value=fake_images):
            with patch.object(backend, "_post_chat", side_effect=completions):
                out = backend.extract(b"%PDF-fake")
        assert out == "# Page 1\n\n# Page 2"

    def test_extract_returns_none_when_all_pages_blank(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        fake_images = [MagicMock()]
        blank = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "  "}}]},
        )
        with patch("ocr_backends_chandra.convert_from_bytes", return_value=fake_images):
            with patch.object(backend, "_post_chat", return_value=blank):
                out = backend.extract(b"%PDF-fake")
        assert out is None

    def test_extract_returns_none_on_http_error(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        fake_images = [MagicMock()]
        with patch("ocr_backends_chandra.convert_from_bytes", return_value=fake_images):
            with patch.object(backend, "_post_chat", side_effect=httpx.HTTPError("boom")):
                out = backend.extract(b"%PDF-fake")
        assert out is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ocr_backends_chandra.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ocr_backends_chandra'`.

- [ ] **Step 3: Implement `ocr_backends_chandra.py`**

Create `ocr_backends_chandra.py`:

```python
"""Chandra2 OCR backend — HTTP client against a vLLM OpenAI-compatible server.

The server lifecycle is owned by `vllm_manager.VLLMManager`; this module only
talks HTTP. It is safe to import even when chandra-ocr / vLLM are not installed.
"""

from __future__ import annotations

import base64
import io
import logging

import httpx
from pdf2image import convert_from_bytes

from config import CHANDRA_MODEL_NAME, CHANDRA_VLLM_PORT

logger = logging.getLogger(__name__)

_OCR_PROMPT = (
    "Extract the document as clean markdown. Preserve tables as GFM pipe tables. "
    "Preserve math formulas as LaTeX inside $...$ or $$...$$. Transcribe Turkish "
    "text faithfully including diacritics. Describe each figure in one line. "
    "Do not invent content."
)

_HTTP_TIMEOUT_SECONDS = 180.0
_HEALTH_TIMEOUT_SECONDS = 3.0
_MAX_OUTPUT_TOKENS = 4096


def _image_to_data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class ChandraBackend:
    """Primary OCR backend. Talks to chandra_vllm over HTTP."""

    name = "chandra2"

    def __init__(self, port: int = 0, model: str = "") -> None:
        self._port = port or CHANDRA_VLLM_PORT
        self._model = model or CHANDRA_MODEL_NAME
        self._base_url = f"http://localhost:{self._port}"

    def is_available(self) -> bool:
        try:
            resp = httpx.get(f"{self._base_url}/v1/models", timeout=_HEALTH_TIMEOUT_SECONDS)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def _post_chat(self, data_url: str) -> httpx.Response:
        payload = {
            "model": self._model,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }
        return httpx.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            timeout=_HTTP_TIMEOUT_SECONDS,
        )

    def extract(self, pdf_bytes: bytes) -> str | None:
        if not pdf_bytes:
            return None
        try:
            images = convert_from_bytes(pdf_bytes, dpi=200)
        except Exception as e:
            logger.warning("chandra2: pdf2image failed: %s", e)
            return None
        if not images:
            return None
        pages: list[str] = []
        for idx, image in enumerate(images, 1):
            data_url = _image_to_data_url(image)
            try:
                resp = self._post_chat(data_url)
            except httpx.HTTPError as e:
                logger.warning("chandra2: chat call failed on page %d: %s", idx, e)
                return None
            if resp.status_code != 200:
                logger.warning("chandra2: non-200 (%d) on page %d", resp.status_code, idx)
                return None
            try:
                text = resp.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, ValueError) as e:
                logger.warning("chandra2: malformed response on page %d: %s", idx, e)
                return None
            text = (text or "").strip()
            if text:
                pages.append(text)
        return "\n\n".join(pages) if pages else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ocr_backends_chandra.py -v`
Expected: all 8 tests PASS.

---

## Task 6: `get_default_backends(include_chandra=True)`

**Files:**
- Modify: `ocr_backends.py:310-313`
- Test: `tests/test_ocr_backends.py` (extend `TestDefaultBackends`)

- [ ] **Step 1: Write the failing test**

Append to `TestDefaultBackends` in `tests/test_ocr_backends.py`:

```python
    def test_include_chandra_prepends_chandra2(self):
        backends = get_default_backends(include_chandra=True)
        names = [b.name for b in backends]
        assert names == ["chandra2", "lightocr", "pp_structure", "markitdown_degraded"]

    def test_default_without_chandra_unchanged(self):
        backends = get_default_backends(include_chandra=False)
        names = [b.name for b in backends]
        assert names == ["lightocr", "pp_structure", "markitdown_degraded"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ocr_backends.py::TestDefaultBackends -v`
Expected: FAIL — `get_default_backends()` doesn't accept `include_chandra`.

- [ ] **Step 3: Modify `get_default_backends`**

Replace the function at the bottom of `ocr_backends.py`:

```python
def get_default_backends(include_chandra: bool = False) -> list[OCRBackend]:
    """Return backend chain in preference order.

    Default: [lightocr, pp_structure, markitdown_degraded].
    With include_chandra=True: [chandra2, lightocr, pp_structure, markitdown_degraded].
    """
    chain: list[OCRBackend] = [LightOCRBackend(), PPStructureBackend(), MarkitdownBackend()]
    if include_chandra:
        from ocr_backends_chandra import ChandraBackend

        chain.insert(0, ChandraBackend())
    return chain
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ocr_backends.py::TestDefaultBackends -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Full OCR test suite green**

Run: `uv run pytest tests/test_ocr_backends.py tests/test_ocr_backends_chandra.py -v`
Expected: all PASS.

---

## Task 7: `scripts/backfill_mevzuat.py` — `--use-chandra` flag

**Files:**
- Modify: `scripts/backfill_mevzuat.py`
- Test: none (integration script — exercised manually in Task 12)

- [ ] **Step 1: Add the CLI flag and config plumbing**

Open `scripts/backfill_mevzuat.py`. In `main()`, add below the existing `--ids-file` argparse:

```python
    parser.add_argument(
        "--use-chandra",
        action="store_true",
        help="Run Chandra2 as primary OCR backend (spawns vLLM subprocess).",
    )
```

- [ ] **Step 2: Wrap the run in `VLLMManager` when `--use-chandra` is set**

Replace the body of `_run` so that when `args.use_chandra` is True, the asyncpg pool is created *inside* a `VLLMManager` block and the `DocumentSyncer` receives a backend chain that includes Chandra2.

Near the top of `scripts/backfill_mevzuat.py`, add imports:

```python
from contextlib import nullcontext

from ocr_backends import get_default_backends
```

In `_run`, replace the current `async with DocumentSyncer(store, vector_store=vs) as syncer:` line with the Chandra-aware construction. Insert this helper right above `_run`:

```python
def _make_vllm_ctx(use_chandra: bool):
    if not use_chandra:
        return nullcontext()
    from config import (
        CHANDRA_GPU_MEMORY_UTILIZATION,
        CHANDRA_MAX_MODEL_LEN,
        CHANDRA_MODEL_NAME,
        CHANDRA_VLLM_HEALTH_TIMEOUT,
        CHANDRA_VLLM_PORT,
    )
    from vllm_manager import VLLMManager

    return VLLMManager(
        model=CHANDRA_MODEL_NAME,
        port=CHANDRA_VLLM_PORT,
        health_timeout=CHANDRA_VLLM_HEALTH_TIMEOUT,
        gpu_memory_utilization=CHANDRA_GPU_MEMORY_UTILIZATION,
        max_model_len=CHANDRA_MAX_MODEL_LEN,
    )
```

Modify the asyncpg block in `_run` so the vLLM manager wraps the whole database work. After the `ids_file` parsing / `_scan` block, just before the `async with DocumentSyncer(...)` line, replace with:

```python
        backends = get_default_backends(include_chandra=args.use_chandra)

        async with DocumentSyncer(store, vector_store=vs, ocr_backends=backends) as syncer:
            ...  # existing loop body unchanged
```

And wrap the call to `_run` in `main()` with the vLLM context. Replace `return asyncio.run(_run(args))` with:

```python
    with _make_vllm_ctx(args.use_chandra):
        return asyncio.run(_run(args))
```

Keep the rest of `_run` (scan, confirm, iterate, report) unchanged.

- [ ] **Step 3: Syntax and import check**

Run: `uv run python -c "import ast; ast.parse(open('scripts/backfill_mevzuat.py').read())"`
Expected: no output (parses clean).

Run: `uv run python scripts/backfill_mevzuat.py --help`
Expected: help text includes `--use-chandra`.

- [ ] **Step 4: Dry-run smoke without Chandra (regression guard)**

Run: `uv run python scripts/backfill_mevzuat.py --limit 0 --yes --ids-file scripts/backfill_ids_20260418.txt` and interrupt with Ctrl-C when the DB pool opens.
Expected: the run reaches `Found N candidate docs` without crashing; the old path is unchanged.

---

## Task 8: `scripts/snapshot_mevzuat_markdown.py`

**Files:**
- Create: `scripts/snapshot_mevzuat_markdown.py`
- Test: none (one-off utility; exercised in runbook)

- [ ] **Step 1: Create the script**

```python
"""Dump current markdown_content for a set of document_ids as JSON.

Used to take a pre-Chandra2 baseline before the backfill overwrites rows.

Usage:
    uv run python scripts/snapshot_mevzuat_markdown.py \
        --ids-file scripts/backfill_ids_20260418.txt \
        --out logs/lightocr_baseline_20260418.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SNAPSHOT_SQL = """
SELECT document_id, extraction_method, markdown_content,
       LENGTH(markdown_content) AS chars
FROM documents
WHERE document_id = ANY($1::text[])
ORDER BY document_id
"""


async def _run(ids_file: Path, out_path: Path) -> int:
    import asyncpg

    from config import DATABASE_URL

    ids = [
        line.strip()
        for line in ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not ids:
        print(f"No IDs found in {ids_file}", file=sys.stderr)
        return 2

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    try:
        rows = await pool.fetch(SNAPSHOT_SQL, ids)
    finally:
        await pool.close()

    payload = [
        {
            "document_id": r["document_id"],
            "extraction_method": r["extraction_method"],
            "markdown_content": r["markdown_content"] or "",
            "chars": int(r["chars"] or 0),
        }
        for r in rows
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    missing = set(ids) - {r["document_id"] for r in rows}
    if missing:
        print(f"WARNING: {len(missing)} IDs not found: {sorted(missing)}", file=sys.stderr)
    print(f"Wrote {len(payload)} rows to {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Snapshot markdown_content for a set of doc_ids")
    p.add_argument("--ids-file", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    return asyncio.run(_run(args.ids_file, args.out))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Syntax check**

Run: `uv run python -c "import ast; ast.parse(open('scripts/snapshot_mevzuat_markdown.py').read())"`
Expected: no output.

Run: `uv run python scripts/snapshot_mevzuat_markdown.py --help`
Expected: help text with `--ids-file` and `--out`.

---

## Task 9: `scripts/compare_ocr_backfill.py` + metrics tests

**Files:**
- Create: `scripts/compare_ocr_backfill.py`
- Create: `tests/test_compare_ocr_backfill.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compare_ocr_backfill.py`:

```python
"""Unit tests for the compare_ocr_backfill metrics."""

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_ocr_backfill.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_ocr_backfill", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_ocr_backfill"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestTextMetrics:
    def test_form_drops_counts_empty_form_blocks(self):
        m = _load_module()
        text = "foo <form></form> bar <form>\n  \n</form> baz <form>not empty</form>"
        assert m.count_form_drops(text) == 2

    def test_form_drops_zero_when_absent(self):
        m = _load_module()
        assert m.count_form_drops("no forms here") == 0

    def test_latex_markers_counts_all_patterns(self):
        m = _load_module()
        text = r"$x_{i}$ and $y^{2}$ with \frac{a}{b}, \sum, \Delta, \cdot, \sqrt{2}, \min, \max"
        assert m.count_latex_markers(text) == 9

    def test_md_image_refs_counts_markdown_and_html(self):
        m = _load_module()
        text = "![alt](x.png) and ![](y.png) and <img src='z.png'/> and <img   src='w.png' >"
        assert m.count_md_image_refs(text) == 4

    def test_md_image_refs_ignores_empty_img_tag(self):
        m = _load_module()
        text = "<img> and <img  >"
        assert m.count_md_image_refs(text) == 0


class TestRegressionFlag:
    def test_flag_true_when_form_drops_worsen(self):
        m = _load_module()
        before = {"form_drops": 1, "latex_markers": 5}
        after = {"form_drops": 2, "latex_markers": 5}
        assert m.regression_flag(before, after) is True

    def test_flag_true_when_latex_markers_regress(self):
        m = _load_module()
        before = {"form_drops": 0, "latex_markers": 10}
        after = {"form_drops": 0, "latex_markers": 8}
        assert m.regression_flag(before, after) is True

    def test_flag_false_when_improved(self):
        m = _load_module()
        before = {"form_drops": 3, "latex_markers": 5}
        after = {"form_drops": 0, "latex_markers": 7}
        assert m.regression_flag(before, after) is False

    def test_flag_false_when_equal(self):
        m = _load_module()
        before = {"form_drops": 0, "latex_markers": 5}
        after = {"form_drops": 0, "latex_markers": 5}
        assert m.regression_flag(before, after) is False


class TestSilentDropCandidate:
    def test_candidate_when_pdf_has_more_images_than_md(self):
        m = _load_module()
        row = {"pdf_image_count": 3, "md_image_refs": 1, "latex_markers": 0}
        assert m.is_silent_drop_candidate(row) is True

    def test_not_candidate_when_latex_offsets_images(self):
        m = _load_module()
        row = {"pdf_image_count": 3, "md_image_refs": 0, "latex_markers": 5}
        assert m.is_silent_drop_candidate(row) is False

    def test_not_candidate_when_no_pdf_images(self):
        m = _load_module()
        row = {"pdf_image_count": 0, "md_image_refs": 0, "latex_markers": 0}
        assert m.is_silent_drop_candidate(row) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compare_ocr_backfill.py -v`
Expected: FAIL — module/functions don't exist yet.

- [ ] **Step 3: Create `scripts/compare_ocr_backfill.py`**

```python
"""Generate a before/after validation report for the Chandra2 backfill.

Reads the baseline JSON produced by scripts/snapshot_mevzuat_markdown.py,
queries the live DB for current markdown, fetches cached PDF bytes for each
doc, and emits a per-doc metrics report (markdown + csv).

Usage:
    uv run python scripts/compare_ocr_backfill.py \
        --baseline logs/lightocr_baseline_20260418.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"

_EMPTY_FORM_RE = re.compile(r"<form>\s*</form>", re.IGNORECASE)
_LATEX_PATTERNS = [
    r"_\{[^}]+\}",
    r"\^\{[^}]+\}",
    r"\\frac",
    r"\\sum",
    r"\\Delta",
    r"\\cdot",
    r"\\sqrt",
    r"\\min",
    r"\\max",
]
_LATEX_RE = re.compile("|".join(_LATEX_PATTERNS))
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_HTML_IMG_RE = re.compile(r"<img\s+[^>]*src=[^>]+>", re.IGNORECASE)


def count_form_drops(text: str) -> int:
    return len(_EMPTY_FORM_RE.findall(text or ""))


def count_latex_markers(text: str) -> int:
    return len(_LATEX_RE.findall(text or ""))


def count_md_image_refs(text: str) -> int:
    return len(_MD_IMG_RE.findall(text or "")) + len(_HTML_IMG_RE.findall(text or ""))


def regression_flag(before: dict, after: dict) -> bool:
    if after["form_drops"] > before["form_drops"]:
        return True
    if after["latex_markers"] < before["latex_markers"]:
        return True
    return False


def is_silent_drop_candidate(row: dict) -> bool:
    return row.get("pdf_image_count", 0) > row.get("md_image_refs", 0) + row.get("latex_markers", 0)


def _pdf_image_count(pdf_bytes: bytes) -> int:
    if not pdf_bytes:
        return 0
    import pdfplumber

    total = 0
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            total += len(page.images or [])
    return total


CURRENT_SQL = """
SELECT document_id, extraction_method, markdown_content,
       LENGTH(markdown_content) AS chars
FROM documents
WHERE document_id = ANY($1::text[])
ORDER BY document_id
"""


async def _build_rows(baseline: list[dict]) -> list[dict]:
    import asyncpg

    from config import DATABASE_URL
    from doc_store import DocumentStore

    ids = [b["document_id"] for b in baseline]
    by_id_before = {b["document_id"]: b for b in baseline}

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    try:
        current = await pool.fetch(CURRENT_SQL, ids)
        store = DocumentStore(pool)
        out: list[dict] = []
        for row in current:
            doc_id = row["document_id"]
            before = by_id_before.get(doc_id, {"markdown_content": "", "chars": 0})
            after_md = row["markdown_content"] or ""
            before_md = before.get("markdown_content", "") or ""
            pdf_bytes = await store.get_pdf_bytes(doc_id)
            pdf_imgs = _pdf_image_count(pdf_bytes) if pdf_bytes else 0
            before_metrics = {
                "chars": before.get("chars", 0),
                "form_drops": count_form_drops(before_md),
                "latex_markers": count_latex_markers(before_md),
                "md_image_refs": count_md_image_refs(before_md),
            }
            after_metrics = {
                "chars": int(row["chars"] or 0),
                "form_drops": count_form_drops(after_md),
                "latex_markers": count_latex_markers(after_md),
                "md_image_refs": count_md_image_refs(after_md),
            }
            entry = {
                "document_id": doc_id,
                "extraction_method_before": before.get("extraction_method"),
                "extraction_method_after": row["extraction_method"],
                "pdf_image_count": pdf_imgs,
                "before": before_metrics,
                "after": after_metrics,
                "regression": regression_flag(before_metrics, after_metrics),
                "silent_drop_candidate": is_silent_drop_candidate(
                    {
                        "pdf_image_count": pdf_imgs,
                        "md_image_refs": after_metrics["md_image_refs"],
                        "latex_markers": after_metrics["latex_markers"],
                    }
                ),
            }
            out.append(entry)
        return out
    finally:
        await pool.close()


def _render_markdown(rows: list[dict]) -> str:
    lines = ["# OCR backfill comparison report", ""]
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Per-doc metrics")
    lines.append("")
    lines.append(
        "| doc_id | method_before → after | chars Δ | form_drops Δ | "
        "latex Δ | md_imgs Δ | pdf_imgs | regression | silent? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        b, a = r["before"], r["after"]
        lines.append(
            f"| {r['document_id']} | {r['extraction_method_before']} → {r['extraction_method_after']} "
            f"| {b['chars']} → {a['chars']} "
            f"| {b['form_drops']} → {a['form_drops']} "
            f"| {b['latex_markers']} → {a['latex_markers']} "
            f"| {b['md_image_refs']} → {a['md_image_refs']} "
            f"| {r['pdf_image_count']} "
            f"| {'REGRESSION' if r['regression'] else 'ok'} "
            f"| {'YES' if r['silent_drop_candidate'] else '-'} |"
        )
    regressions = [r["document_id"] for r in rows if r["regression"]]
    silent = [r["document_id"] for r in rows if r["silent_drop_candidate"]]
    lines.append("")
    lines.append(f"**Regressions:** {regressions or 'none'}")
    lines.append(f"**Silent drop candidates:** {silent or 'none'}")
    return "\n".join(lines) + "\n"


def _render_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "document_id", "extraction_method_before", "extraction_method_after",
        "chars_before", "chars_after",
        "form_drops_before", "form_drops_after",
        "latex_markers_before", "latex_markers_after",
        "md_image_refs_before", "md_image_refs_after",
        "pdf_image_count", "regression", "silent_drop_candidate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "document_id": r["document_id"],
                "extraction_method_before": r["extraction_method_before"],
                "extraction_method_after": r["extraction_method_after"],
                "chars_before": r["before"]["chars"],
                "chars_after": r["after"]["chars"],
                "form_drops_before": r["before"]["form_drops"],
                "form_drops_after": r["after"]["form_drops"],
                "latex_markers_before": r["before"]["latex_markers"],
                "latex_markers_after": r["after"]["latex_markers"],
                "md_image_refs_before": r["before"]["md_image_refs"],
                "md_image_refs_after": r["after"]["md_image_refs"],
                "pdf_image_count": r["pdf_image_count"],
                "regression": r["regression"],
                "silent_drop_candidate": r["silent_drop_candidate"],
            })


async def _run(baseline_path: Path) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    rows = await _build_rows(baseline)

    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    md_path = LOG_DIR / f"compare_{stamp}.md"
    csv_path = LOG_DIR / f"compare_{stamp}.csv"
    md_path.write_text(_render_markdown(rows), encoding="utf-8")
    _render_csv(rows, csv_path)
    print(f"Wrote {md_path} and {csv_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Compare OCR backfill before/after")
    p.add_argument("--baseline", required=True, type=Path)
    args = p.parse_args()
    return asyncio.run(_run(args.baseline))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compare_ocr_backfill.py -v`
Expected: all 11 tests PASS.

- [ ] **Step 5: Syntax check and help**

Run: `uv run python scripts/compare_ocr_backfill.py --help`
Expected: help text with `--baseline`.

---

## Task 10: GPU-gated smoke test

**Files:**
- Create: `tests/test_chandra_smoke.py`

- [ ] **Step 1: Create the GPU-marked integration test**

```python
"""GPU-gated integration test for Chandra2 end-to-end.

Skipped unless CUDA is available. Spins up a real vLLM subprocess, sends a
1-page PDF, asserts non-empty Turkish output. Runtime: ~2 minutes on first
run (model download + vLLM startup).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


FIXTURE_PDF = Path(__file__).parent / "fixtures" / "mevzuat_42628_sample.pdf"


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda(), reason="CUDA not available")
def test_chandra_end_to_end_on_real_fixture():
    if not FIXTURE_PDF.exists():
        pytest.skip(f"fixture missing: {FIXTURE_PDF}")

    from config import (
        CHANDRA_GPU_MEMORY_UTILIZATION,
        CHANDRA_MAX_MODEL_LEN,
        CHANDRA_MODEL_NAME,
        CHANDRA_VLLM_HEALTH_TIMEOUT,
        CHANDRA_VLLM_PORT,
    )
    from ocr_backends_chandra import ChandraBackend
    from vllm_manager import VLLMManager

    pdf_bytes = FIXTURE_PDF.read_bytes()

    with VLLMManager(
        model=CHANDRA_MODEL_NAME,
        port=CHANDRA_VLLM_PORT,
        health_timeout=CHANDRA_VLLM_HEALTH_TIMEOUT,
        gpu_memory_utilization=CHANDRA_GPU_MEMORY_UTILIZATION,
        max_model_len=CHANDRA_MAX_MODEL_LEN,
    ):
        backend = ChandraBackend()
        assert backend.is_available() is True
        output = backend.extract(pdf_bytes)

    assert output is not None
    assert len(output) > 100
```

- [ ] **Step 2: Confirm the test is skipped when `-m 'not gpu'`**

Run: `uv run pytest tests/test_chandra_smoke.py -v`
Expected: the test is deselected (project default is `-m 'not gpu'`) — output shows `1 deselected` or similar.

---

## Task 11: Baseline snapshot (runbook step 1)

**Files:**
- No file changes — execution only.

- [ ] **Step 1: Ensure DB is up**

Run: `docker compose ps db`
Expected: `bddk-mcp-db-1` is `healthy` / `running`. If not, run `docker compose up -d db`.

- [ ] **Step 2: Confirm the ID list matches the intended 14 docs**

Run: `uv run python -c "p='scripts/backfill_ids_20260418.txt'; print(sum(1 for l in open(p) if l.strip() and not l.startswith('#')))"`
Expected: `14`.

- [ ] **Step 3: Take the baseline snapshot**

Run:
```bash
mkdir -p logs
uv run python scripts/snapshot_mevzuat_markdown.py \
    --ids-file scripts/backfill_ids_20260418.txt \
    --out logs/lightocr_baseline_20260418.json
```
Expected stdout: `Wrote 14 rows to logs/lightocr_baseline_20260418.json`.

- [ ] **Step 4: Spot-check the snapshot**

Run: `uv run python -c "import json; d=json.load(open('logs/lightocr_baseline_20260418.json')); print(len(d)); print({r['extraction_method'] for r in d}); print(any('<form></form>' in r['markdown_content'] for r in d))"`
Expected: `14`, a set including `"lightocr"`, and `True` (confirms the known `mevzuat_21194` empty-`<form>` issue is present in the baseline).

---

## Task 12: Chandra2 backfill execution (runbook step 2)

**Files:**
- No file changes — execution only.

- [ ] **Step 1: Pre-flight checks**

Run: `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`
Expected: ≥ 14000 (MiB). If less, close other GPU processes before proceeding.

Run: `uv sync --group gpu`
Expected: installs `chandra-ocr` and transitive `vllm`.

Run: `uv run python -c "from chandra_ocr import __version__; print(__version__)"`
Expected: prints a version with no error.

- [ ] **Step 2: Launch the backfill**

Run (in a terminal you can keep open — expect 1–2 hours):
```bash
uv run python scripts/backfill_mevzuat.py \
    --use-chandra --yes \
    --ids-file scripts/backfill_ids_20260418.txt
```
Expected timeline:
1. `Launching vLLM: chandra_vllm ...` log line.
2. Within ~5 minutes: `vLLM healthy on port 8001`.
3. `Found 14 candidate docs` and iteration begins.
4. Each doc logs `[i/14] mevzuat_XXXXX OK in NNs (method=chandra2, size=NNNB)`.
5. On completion: `Backfill complete: 14 ok, 0 failed.`

Log file: `logs/backfill_YYYY-MM-DD-HHMM.log` — tail this in a second terminal for progress.

- [ ] **Step 3: Confirm Chandra2 produced every extraction**

Run:
```bash
uv run python -c "
import asyncio, asyncpg
from config import DATABASE_URL

async def main():
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        rows = await pool.fetch(
            '''SELECT document_id, extraction_method, LENGTH(markdown_content) AS chars
               FROM documents WHERE document_id = ANY(\$1::text[]) ORDER BY document_id''',
            [line.strip() for line in open('scripts/backfill_ids_20260418.txt')
             if line.strip() and not line.startswith('#')],
        )
    finally:
        await pool.close()
    for r in rows:
        print(r['document_id'], r['extraction_method'], r['chars'])

asyncio.run(main())
"
```
Expected: 14 rows, every `extraction_method == 'chandra2'`, `chars` ≥ 500 on each.

If any row shows `extraction_method == 'lightocr'`, it means the chain fell through — check the backfill log for the Chandra2 error and decide whether to retry.

---

## Task 13: Validation report + decision gate (runbook steps 3-5)

**Files:**
- No file changes — execution + decision only.

- [ ] **Step 1: Generate the comparison report**

Run:
```bash
uv run python scripts/compare_ocr_backfill.py \
    --baseline logs/lightocr_baseline_20260418.json
```
Expected: `Wrote logs/compare_YYYY-MM-DD-HHMM.md and logs/compare_YYYY-MM-DD-HHMM.csv`.

- [ ] **Step 2: Review the report against the three success criteria**

Open the generated `logs/compare_*.md` and verify:

1. Row `mevzuat_21194`: `form_drops` shows `3 → 0`.
2. The `**Regressions:**` line reads `none`.
3. At least one row has either `latex_markers` strictly higher than baseline or `md_image_refs` strictly higher. (This is the "measurable improvement" gate.)

- [ ] **Step 3: Decision gate**

- **All three criteria pass** → Chandra2 validated. Move to Step 4.
- **Any criterion fails** → open the CSV, identify the failing doc(s), and either:
  a. Restore from baseline (`UPDATE documents SET markdown_content=$1, extraction_method='lightocr' WHERE document_id=$2` driven by the JSON), or
  b. Re-run only the failing IDs via a temporary IDs file and `--use-chandra`.
  Re-run Step 1 after any restore/retry.

- [ ] **Step 4: Document the outcome**

Append a short results block to the top of `docs/superpowers/specs/2026-04-18-chandra2-primary-ocr-design.md` (under the existing "Motivation" line):

```markdown
**Outcome (YYYY-MM-DD):** Chandra2 validated on 14 mevzuat PDFs. mevzuat_21194 form_drops 3 → 0. Report: logs/compare_YYYY-MM-DD-HHMM.md.
```

- [ ] **Step 5: Update project CLAUDE.md runbook pointer**

Append to the `## Commands` section of `CLAUDE.md`:

```bash
# Chandra2 backfill (one-off):
uv run python scripts/snapshot_mevzuat_markdown.py --ids-file <ids> --out <baseline.json>
uv run python scripts/backfill_mevzuat.py --use-chandra --yes --ids-file <ids>
uv run python scripts/compare_ocr_backfill.py --baseline <baseline.json>
```

- [ ] **Step 6: Full test suite green (final regression check)**

Run: `uv run pytest tests/ -m 'not gpu' -q`
Expected: all pre-existing tests plus the new ones pass; no regressions in `test_doc_sync`, `test_doc_sync_reindex`, `test_ocr_backends`.

---

## Self-review notes

Spec coverage:
- §1 Architecture → Tasks 3 (`<form>` guard), 5 (Chandra backend), 4 (vLLM lifecycle), 7 (backfill-only deployment), 6 (chain assembly).
- §2 Components & file layout → Tasks 1 (config), 2 (deps), 4, 5, 6, 7, 8 (snapshot), 9 (compare).
- §3 Validation script → Task 9 + Tasks 11–13 (execution).
- §4 Testing strategy → Tasks 3, 4, 5, 6, 9 (unit), 10 (GPU smoke).
- §5 Runbook → Tasks 11, 12, 13.
- §6 Open risks → addressed via: VRAM pressure (config defaults, pre-flight in Task 12), hard-abort on vLLM failure (Task 4 `_await_health` raises, bubbling up), silent drops (Task 9 `silent_drop_candidate`).

Naming consistency: `ChandraBackend.name == "chandra2"` used in Tasks 5, 6, 12. `get_default_backends(include_chandra=...)` signature consistent across Tasks 6 and 7. Config constants `CHANDRA_*` identical across Tasks 1, 7, 10, 12.

No placeholders: every code step contains complete code; every run step has an exact command and expected output.
