#!/usr/bin/env python3
"""Refresh the local graphify code graph and its Obsidian vault.

Local developer aid only (see AGENTS.md, "Code knowledge graph"). It runs the
AST-only ``graphify update`` path and then ``graphify export obsidian`` into the
git-ignored ``graphify-out/obsidian`` vault. LLM provider keys are removed from
the child environment so no step can call a model backend.

Usage:
    uv run --group graph python scripts/graphify_sync.py                   # refresh graph + vault
    uv run --group graph python scripts/graphify_sync.py --install-hooks   # refresh after commit/merge/checkout
    uv run --group graph python scripts/graphify_sync.py --uninstall-hooks
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH_OUT = ROOT / "graphify-out"
VAULT_DIR = GRAPH_OUT / "obsidian"

HOOK_NAMES = ("post-commit", "post-merge", "post-checkout")
BLOCK_START = "# >>> bddk-mcp graphify-sync >>>"
BLOCK_END = "# <<< bddk-mcp graphify-sync <<<"
HOOK_BLOCK = f"""{BLOCK_START}
# Refreshes graphify-out/ and its Obsidian vault in the background (local only, no LLM).
# Remove with: uv run --group graph python scripts/graphify_sync.py --uninstall-hooks
(
  cd "$(git rev-parse --show-toplevel)" || exit 0
  mkdir -p graphify-out
  uv run --group graph python scripts/graphify_sync.py > graphify-out/sync.log 2>&1
) >/dev/null 2>&1 &
{BLOCK_END}
"""

# Keys graphify would use to reach an LLM backend; never forwarded.
LLM_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "MOONSHOT_API_KEY",
    "DEEPSEEK_API_KEY",
)


def offline_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for key in LLM_ENV_KEYS:
        env.pop(key, None)
    env["GRAPHIFY_NO_TIPS"] = "1"
    return env


def _graphify(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "graphify", *args], cwd=ROOT, env=offline_env()).returncode


def sync() -> int:
    update_code = _graphify("update", ".")
    graph_json = GRAPH_OUT / "graph.json"
    if not graph_json.is_file():
        print("graphify-sync: graph.json was not produced; vault not refreshed.", file=sys.stderr)
        return update_code or 1
    if update_code != 0:
        # `graphify update` also exits 1 when nothing changed; the existing graph is still exported.
        print("graphify-sync: update reported no rebuild; exporting the existing graph.", file=sys.stderr)
    export_code = _graphify("export", "obsidian", "--graph", str(graph_json), "--dir", str(VAULT_DIR))
    if export_code == 0:
        print(f"graphify-sync: Obsidian vault refreshed at {VAULT_DIR}")
    return export_code


def _hooks_dir() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    path = Path(out)
    return path if path.is_absolute() else ROOT / path


def _strip_block(text: str) -> str:
    start = text.find(BLOCK_START)
    if start < 0:
        return text
    end = text.find(BLOCK_END, start)
    end = len(text) if end < 0 else end + len(BLOCK_END)
    return (text[:start].rstrip("\n") + "\n" + text[end:].lstrip("\n")).rstrip("\n") + "\n"


def install_hooks(hooks_dir: Path) -> list[str]:
    """Add the sync block to each hook, preserving any existing hook content."""
    hooks_dir.mkdir(parents=True, exist_ok=True)
    changed = []
    for name in HOOK_NAMES:
        hook = hooks_dir / name
        current = hook.read_text(encoding="utf-8") if hook.exists() else "#!/bin/sh\n"
        if BLOCK_START in current:
            continue
        hook.write_text(current.rstrip("\n") + "\n\n" + HOOK_BLOCK, encoding="utf-8", newline="\n")
        hook.chmod(hook.stat().st_mode | 0o111)
        changed.append(name)
    return changed


def uninstall_hooks(hooks_dir: Path) -> list[str]:
    """Remove only the sync block; delete a hook file that held nothing else."""
    changed = []
    for name in HOOK_NAMES:
        hook = hooks_dir / name
        if not hook.exists():
            continue
        current = hook.read_text(encoding="utf-8")
        if BLOCK_START not in current:
            continue
        remaining = _strip_block(current)
        if remaining.strip() in ("", "#!/bin/sh"):
            hook.unlink()
        else:
            hook.write_text(remaining, encoding="utf-8", newline="\n")
        changed.append(name)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--install-hooks", action="store_true", help="refresh after commit, merge and checkout")
    group.add_argument("--uninstall-hooks", action="store_true", help="remove the refresh hooks")
    args = parser.parse_args(argv)

    if args.install_hooks:
        changed = install_hooks(_hooks_dir())
        print(f"graphify-sync hooks installed: {', '.join(changed) or 'already present'}")
        return 0
    if args.uninstall_hooks:
        changed = uninstall_hooks(_hooks_dir())
        print(f"graphify-sync hooks removed: {', '.join(changed) or 'none found'}")
        return 0
    return sync()


if __name__ == "__main__":
    raise SystemExit(main())
