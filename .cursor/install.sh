#!/usr/bin/env bash
# Idempotent bootstrap for the CS336 assignments (uv-managed Python projects).
# Safe to run repeatedly: it only does work when something is missing or stale.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Ensure the user-local bin (where uv installs) is on PATH for this script.
export PATH="$HOME/.local/bin:$PATH"

# 1. Ensure uv is available (managed Python + dependency resolver).
if ! command -v uv >/dev/null 2>&1; then
  echo "== installing uv =="
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# 2. Ensure CPython development headers are present. torch.compile / TorchInductor
#    generates and compiles C++ at runtime and needs Python.h (assignment 2's
#    flash-attention tests exercise this path).
if ! ls /usr/include/python3.*/Python.h >/dev/null 2>&1; then
  echo "== installing python3-dev (Python.h for TorchInductor) =="
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends python3-dev
fi

# 3. Sync each assignment's virtual environment from its lockfile.
for d in assignment1 assignment2 assignment5; do
  if [ -f "$ROOT/$d/pyproject.toml" ]; then
    echo "== uv sync: $d =="
    (cd "$ROOT/$d" && uv sync)
  fi
done

echo "== environment ready =="
