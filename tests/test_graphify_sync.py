from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import graphify_sync  # noqa: E402


def test_offline_env_drops_llm_keys_and_keeps_the_rest():
    env = graphify_sync.offline_env({"ANTHROPIC_API_KEY": "x", "GEMINI_API_KEY": "y", "PATH": "/bin"})
    assert "ANTHROPIC_API_KEY" not in env
    assert "GEMINI_API_KEY" not in env
    assert env["PATH"] == "/bin"
    assert env["GRAPHIFY_NO_TIPS"] == "1"


def test_install_is_idempotent_and_preserves_existing_hooks(tmp_path):
    existing = tmp_path / "post-commit"
    existing.write_text("#!/bin/sh\necho owner-hook\n", encoding="utf-8")

    assert graphify_sync.install_hooks(tmp_path) == list(graphify_sync.HOOK_NAMES)
    assert graphify_sync.install_hooks(tmp_path) == []

    text = existing.read_text(encoding="utf-8")
    assert "echo owner-hook" in text
    assert text.count(graphify_sync.BLOCK_START) == 1
    assert (tmp_path / "post-merge").stat().st_mode & 0o111


def test_uninstall_removes_only_the_sync_block(tmp_path):
    (tmp_path / "post-commit").write_text("#!/bin/sh\necho owner-hook\n", encoding="utf-8")
    graphify_sync.install_hooks(tmp_path)

    assert graphify_sync.uninstall_hooks(tmp_path) == list(graphify_sync.HOOK_NAMES)

    assert (tmp_path / "post-commit").read_text(encoding="utf-8") == "#!/bin/sh\necho owner-hook\n"
    assert not (tmp_path / "post-merge").exists()
    assert not (tmp_path / "post-checkout").exists()
    assert graphify_sync.uninstall_hooks(tmp_path) == []
