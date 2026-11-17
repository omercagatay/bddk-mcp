"""The Claude Code hooks in .claude/: the after-edit ruff script and the shape of settings.json."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".claude" / "hooks" / "ruff_after_edit.py"
SETTINGS = ROOT / ".claude" / "settings.json"


def run_hook(stdin: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(HOOK)], input=stdin, capture_output=True, text=True, timeout=60)


def edit_event(path: Path) -> str:
    return json.dumps({"tool_name": "Edit", "tool_input": {"file_path": str(path)}, "cwd": str(path.parent)})


def test_lint_finding_blocks_with_the_finding_on_stderr(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("import os\n", encoding="utf-8")
    result = run_hook(edit_event(sample))
    assert result.returncode == 2
    assert "F401" in result.stderr
    assert sample.read_text(encoding="utf-8") == "import os\n"  # reported, never auto-fixed


def test_clean_file_passes(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("VALUE = 1\n", encoding="utf-8")
    assert run_hook(edit_event(sample)).returncode == 0


def test_non_python_file_is_ignored(tmp_path):
    sample = tmp_path / "notes.md"
    sample.write_text("import os\n", encoding="utf-8")
    assert run_hook(edit_event(sample)).returncode == 0


def test_a_ruff_module_beside_the_edited_file_is_not_executed(tmp_path):
    marker = tmp_path / "executed"
    (tmp_path / "ruff.py").write_text(f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8")
    sample = tmp_path / "sample.py"
    sample.write_text("VALUE = 1\n", encoding="utf-8")
    assert run_hook(edit_event(sample)).returncode == 0
    assert not marker.exists()


def test_permission_rules_stay_narrow():
    permissions = json.loads(SETTINGS.read_text(encoding="utf-8"))["permissions"]
    # The exact reviewed list: a new or wider entry is a reviewed change of this test, not only of the settings file.
    # Shell rules run the repository's own checks and read Railway's state; deploying, `railway run` and applying
    # configuration from a shell always prompt, and ruff is allowed only in the forms that write nothing.
    checks = [
        "uv run pytest tests/*",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run python scripts/check_repository_hygiene.py",
        "uv lock --check",
    ]
    railway = ["deployment list", "logs", "status", "whoami", "config plan"]
    assert sorted(permissions["allow"]) == sorted(
        [f"Bash(railway {command} *)" for command in railway]
        + ["mcp__railway"]
        + [f"{tool}({command})" for command in checks for tool in ("Bash", "PowerShell")]
    )
    for flag in ("--basetemp", "--pastebin", "--junitxml", "@"):
        for tool in ("Bash", "PowerShell"):
            assert f"{tool}(uv run pytest *{flag}*)" in permissions["deny"], (tool, flag)


@pytest.mark.parametrize("stdin", ["", "not json", "[]", '{"tool_input": {}}', '{"tool_input": {"file_path": null}}'])
def test_empty_or_malformed_input_exits_silently(stdin):
    result = run_hook(stdin)
    assert (result.returncode, result.stderr) == (0, "")


def test_settings_hooks_are_well_formed():
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    commands = [
        handler["command"]
        for groups in settings["hooks"].values()
        for group in groups
        for handler in group["hooks"]
        if handler["type"] == "command"
    ]
    assert commands
    for command in commands:
        assert not re.search(r"\bjq\b", command)
        assert "|| true" not in command
        for script in re.findall(r"\.claude/hooks/[\w.]+", command):
            assert (ROOT / script).is_file(), script
    assert {"Stop", "SubagentStop"}.isdisjoint(settings["hooks"])
