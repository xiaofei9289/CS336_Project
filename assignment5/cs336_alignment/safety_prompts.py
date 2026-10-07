"""Prompt formatting for the optional safety / RLHF supplement.

Zero-shot benchmarks share one system prompt. After SFT, the same task text
is placed in the Alpaca template instead. GRPO prompts are not used here.
"""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts_safety"


def _read(name: str) -> str:
    return (PROMPT_DIR / name).read_text()


def render_zero_shot(instruction: str) -> str:
    system = _read("zero_shot_system_prompt.prompt")
    if "{instruction}" not in system:
        raise ValueError("zero_shot_system_prompt.prompt is missing {instruction}")
    return system.replace("{instruction}", instruction)


def render_alpaca(instruction: str, response: str = "") -> str:
    template = _read("alpaca_sft.prompt").rstrip("\n")
    return template.format(instruction=instruction, response=response)


def mmlu_instruction(subject: str, question: str, options: list[str]) -> str:
    text = _read("mmlu_zero_shot.prompt")
    replacements = {
        "{subject}": subject.replace("_", " "),
        "{question}": question,
        "{options[0]}": options[0],
        "{options[1]}": options[1],
        "{options[2]}": options[2],
        "{options[3]}": options[3],
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def gsm8k_instruction(question: str) -> str:
    return _read("gsm8k_zero_shot.prompt").replace("{question}", question)


def format_benchmark_prompt(instruction: str, style: str) -> str:
    if style == "zero_shot":
        return render_zero_shot(instruction)
    if style == "alpaca":
        return render_alpaca(instruction, response="")
    raise ValueError(f"unknown prompt style {style}")
