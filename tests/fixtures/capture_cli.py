"""Run the current SCAR trace command from an isolated workload directory."""
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def run_trace(entry, output, *options, environment=None):
    command_env = {name: value for name, value in os.environ.items() if not name.startswith("SCAR_")}
    command_env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    command_env.update(environment or {})
    return subprocess.run(
        [sys.executable, "-m", "scar", "trace", "--project-root", str(entry.parent),
         "--out", str(output), *map(str, options), "--", sys.executable, str(entry)],
        cwd=entry.parent, env=command_env, capture_output=True, text=True, timeout=90,
    )
