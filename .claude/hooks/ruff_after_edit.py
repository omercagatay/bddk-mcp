"""PostToolUse hook for Edit|Write: lint the edited Python file with ruff and show the findings to Claude.

Reads the hook event JSON on stdin. Exit 2 with the findings on stderr when ruff reports something; exit 0
otherwise, also for malformed input or a path that is not an existing .py file. No --fix: an automatic fix can
delete an import that the next edit was about to use.
"""

import json
import subprocess
import sys
from pathlib import Path

try:
    path = Path(json.load(sys.stdin)["tool_input"]["file_path"])
except (ValueError, KeyError, TypeError):
    sys.exit(0)
if path.suffix != ".py" or not path.is_file():
    sys.exit(0)

# The file may live in a worktree, not in the session's project directory: run from the repository that holds it.
top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path.parent, capture_output=True, text=True)
cwd = top.stdout.strip() if top.returncode == 0 else path.parent
result = subprocess.run(
    # -P keeps the working directory off sys.path, so a ruff.py beside the edited file is linted, never executed.
    [sys.executable, "-P", "-m", "ruff", "check", "--force-exclude", "--output-format", "concise", str(path.resolve())],
    cwd=cwd,
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    sys.stderr.write(result.stdout + result.stderr)
    sys.exit(2)
