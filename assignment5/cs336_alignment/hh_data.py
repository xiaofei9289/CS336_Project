"""Load Anthropic HH preference data for DPO.

Keeps only single-turn dialogues. A later human message can diverge between
the chosen and rejected transcripts, so those rows are dropped.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


HH_TRAIN_FILES = (
    "harmless-base.jsonl.gz",
    "helpful-online.jsonl.gz",
    "helpful-base.jsonl.gz",
    "helpful-rejection-sampled.jsonl.gz",
)


def _turns(transcript: str) -> tuple[list[str], list[str]]:
    humans: list[str] = []
    assistants: list[str] = []
    chunks = transcript.split("\n\nHuman:")
    for chunk in chunks[1:]:
        if "\n\nAssistant:" not in chunk:
            humans.append(chunk.strip())
            assistants.append("")
            continue
        human, assistant = chunk.split("\n\nAssistant:", 1)
        humans.append(human.strip())
        assistants.append(assistant.strip())
    return humans, assistants


def load_hh_train(data_dir: str | Path) -> list[dict[str, str]]:
    """Merge the four HH training files into single-turn preference rows."""
    root = Path(data_dir)
    rows: list[dict[str, str]] = []
    for name in HH_TRAIN_FILES:
        path = root / name
        with gzip.open(path, "rt") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                chosen_humans, chosen_assistants = _turns(raw["chosen"])
                rejected_humans, rejected_assistants = _turns(raw["rejected"])
                if len(chosen_humans) != 1 or len(rejected_humans) != 1:
                    continue
                if chosen_humans[0] != rejected_humans[0]:
                    continue
                if not chosen_assistants[0] or not rejected_assistants[0]:
                    continue
                rows.append(
                    {
                        "source": name.removesuffix(".jsonl.gz"),
                        "instruction": chosen_humans[0],
                        "chosen": chosen_assistants[0],
                        "rejected": rejected_assistants[0],
                    }
                )
    return rows
