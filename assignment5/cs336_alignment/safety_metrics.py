"""Answer parsers for the safety-supplement benchmarks."""

from __future__ import annotations

import re


def parse_mmlu_response(model_output: str) -> str | None:
    match = re.search(r"The correct answer is ([A-D])\b", model_output)
    if match is None:
        return None
    return match.group(1)


def parse_gsm8k_response(model_output: str) -> str | None:
    # "1,200" is one number. The comma is not a second value.
    normalized = re.sub(r"(?<=\d),(?=\d)", "", model_output)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", normalized)
    if not numbers:
        return None
    return numbers[-1]


def gsm8k_answers_match(parsed: str | None, gold: str) -> bool:
    if parsed is None:
        return False
    gold_text = gold.replace(",", "").strip()
    try:
        return float(parsed) == float(gold_text)
    except ValueError:
        return parsed == gold_text
