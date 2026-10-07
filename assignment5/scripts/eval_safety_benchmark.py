"""Greedy zero-shot or Alpaca-template eval for the safety supplement.

Does not change GRPO sampling defaults. Throughput counts only vLLM generate time.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from cs336_alignment.safety_metrics import gsm8k_answers_match, parse_gsm8k_response, parse_mmlu_response
from cs336_alignment.safety_prompts import (
    format_benchmark_prompt,
    gsm8k_instruction,
    mmlu_instruction,
)
from cs336_alignment.vllm_utils import VLLMServer

ROOT = Path(__file__).resolve().parents[1]


def load_mmlu(split_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(split_dir.glob("*_test.csv")):
        subject = path.name[: -len("_test.csv")]
        with path.open(newline="") as handle:
            for record in csv.reader(handle):
                if len(record) < 6:
                    continue
                question, *options, answer = record[:6]
                rows.append(
                    {
                        "subject": subject,
                        "question": question,
                        "options": options,
                        "answer": answer.strip(),
                    }
                )
    return rows


def load_gsm8k(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append(
                {
                    "question": raw["question"],
                    "answer": raw["answer"].split("####")[-1].strip().replace(",", ""),
                }
            )
    return rows


def load_alpaca(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def load_sst(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def examples_for(task: str, limit: int | None) -> list[dict]:
    if task == "mmlu":
        rows = load_mmlu(ROOT / "data/mmlu/test")
    elif task == "gsm8k":
        rows = load_gsm8k(ROOT / "data/gsm8k/test.jsonl")
    elif task == "alpaca":
        rows = load_alpaca(ROOT / "data/alpaca_eval/alpaca_eval_gpt4_turbo.json")
    elif task == "sst":
        rows = load_sst(ROOT / "data/simple_safety_tests/simple_safety_tests.csv")
    else:
        raise ValueError(task)
    return rows if limit is None else rows[:limit]


def instruction_for(task: str, row: dict) -> str:
    if task == "mmlu":
        return mmlu_instruction(row["subject"], row["question"], row["options"])
    if task == "gsm8k":
        return gsm8k_instruction(row["question"])
    if task == "alpaca":
        return row["instruction"]
    if task == "sst":
        return row["prompts_final"]
    raise ValueError(task)


def score_row(task: str, row: dict, generation: str) -> dict:
    if task == "mmlu":
        parsed = parse_mmlu_response(generation)
        return {"parsed": parsed, "correct": parsed == row["answer"]}
    if task == "gsm8k":
        parsed = parse_gsm8k_response(generation)
        return {"parsed": parsed, "correct": gsm8k_answers_match(parsed, row["answer"])}
    return {}


def write_outputs(task: str, records: list[dict], output: Path, generator: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if task == "alpaca":
        payload = [
            {
                "instruction": record["instruction"],
                "output": record["generation"],
                "generator": generator,
                "dataset": record["dataset"],
            }
            for record in records
        ]
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if task == "sst":
        with output.open("w") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {"prompts_final": record["prompts_final"], "output": record["generation"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mmlu", "gsm8k", "alpaca", "sst"], required=True)
    parser.add_argument("--prompt-style", choices=["zero_shot", "alpaca"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator", default="llama-3.1-8b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    rows = examples_for(args.task, args.limit)
    prompts = [format_benchmark_prompt(instruction_for(args.task, row), args.prompt_style) for row in rows]
    sampling = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "n": 1,
        "seed": 0,
    }
    if args.prompt_style == "zero_shot":
        sampling["stop"] = ["# Query:"]
        sampling["include_stop_str_in_output"] = False

    server = VLLMServer(model_id=args.model, port=args.port, gpu=args.gpu, seed=0)
    server.start()
    started = time.perf_counter()
    completions = server.generate_completions(prompts, sampling, batch_size=args.batch_size)
    elapsed = time.perf_counter() - started
    if len(completions) != len(rows):
        raise RuntimeError(f"expected {len(rows)} completions, got {len(completions)}")

    records = []
    n_correct = 0
    n_scored = 0
    n_unparsed = 0
    for row, prompt, completion in zip(rows, prompts, completions, strict=True):
        record = dict(row)
        record["prompt"] = prompt
        record["generation"] = completion.text
        scored = score_row(args.task, row, completion.text)
        record.update(scored)
        if "correct" in scored:
            n_scored += 1
            n_correct += int(scored["correct"])
            if scored["parsed"] is None:
                n_unparsed += 1
        records.append(record)

    write_outputs(args.task, records, args.output, args.generator)
    summary = {
        "task": args.task,
        "prompt_style": args.prompt_style,
        "n": len(records),
        "elapsed_generate_seconds": elapsed,
        "examples_per_second": len(records) / elapsed if elapsed else None,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_scored) if n_scored else None,
        "n_unparsed": n_unparsed,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
