"""Lightweight IO helpers for prompting_baselines (naming aligned with reference solutions)."""

from __future__ import annotations

import json
from pathlib import Path


def load_data(filepath: str | Path) -> list[dict]:
    examples = []
    with Path(filepath).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_prompt(prompt_path: str | Path) -> str:
    return Path(prompt_path).read_text(encoding="utf-8")
