"""Empirical smoke test: chandra HF InferenceManager on real mevzuat PDF.

Validates Path A (in-process HF) works before rewriting code. First run
downloads ~10GB of model weights.

Usage: uv run python scripts/smoke_chandra_hf.py
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

assert torch.cuda.is_available(), "CUDA not available"

# Imports below are deferred until after the CUDA check so the script fails
# fast on CPU-only machines without paying the heavy chandra import cost.
from chandra.input import load_file  # noqa: E402
from chandra.model import InferenceManager  # noqa: E402
from chandra.model.schema import BatchInputItem  # noqa: E402

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "mevzuat_42628_sample.pdf"


def main() -> None:
    print(f"[{time.strftime('%H:%M:%S')}] fixture: {FIXTURE} ({FIXTURE.stat().st_size} bytes)")

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading InferenceManager(method='hf')...")
    mgr = InferenceManager(method="hf")
    print(f"[{time.strftime('%H:%M:%S')}] model loaded in {time.time() - t0:.1f}s")

    print(f"[{time.strftime('%H:%M:%S')}] rendering PDF pages...")
    images = load_file(str(FIXTURE), {})
    print(f"[{time.strftime('%H:%M:%S')}] {len(images)} page(s) rendered")

    # Process first page only for smoke test
    print(f"[{time.strftime('%H:%M:%S')}] running inference on page 1...")
    t1 = time.time()
    batch = [BatchInputItem(image=images[0], prompt_type="ocr_layout")]
    outputs = mgr.generate(batch)
    print(f"[{time.strftime('%H:%M:%S')}] inference took {time.time() - t1:.1f}s")

    result = outputs[0]
    md = result.markdown or ""
    print(f"[{time.strftime('%H:%M:%S')}] markdown chars: {len(md)}")
    print(f"[{time.strftime('%H:%M:%S')}] token_count: {result.token_count}")
    print(f"[{time.strftime('%H:%M:%S')}] num_chunks: {len(result.chunks or [])}")
    print(f"[{time.strftime('%H:%M:%S')}] num_images: {len(result.images or {})}")
    print(f"[{time.strftime('%H:%M:%S')}] error: {getattr(result, 'error', 'n/a')}")

    print("\n--- markdown preview (first 800 chars) ---")
    print(md[:800])
    print("--- end preview ---\n")

    # Smoke assertions
    assert len(md) > 100, f"markdown too short: {len(md)} chars"
    turkish_chars = sum(1 for c in md if c in "çğışöüÇĞİŞÖÜ")
    print(f"Turkish diacritic count: {turkish_chars}")
    assert turkish_chars > 0, "no Turkish diacritics — likely garbage output"

    print(f"\n[{time.strftime('%H:%M:%S')}] SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
