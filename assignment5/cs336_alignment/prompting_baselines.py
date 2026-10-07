"""Prompting baselines: question_only / r1_zero / r1_zero_three_shot.

Sampling: temperature=1.0, top_p=1.0, max_tokens=512.
r1_zero and three-shot stop at </answer> with include_stop_str_in_output=True.
Do not set that stop for question_only.

Three outcome buckets:
1. format=1 and answer=1
2. format=1 and answer=0
3. format=0 and answer=0

Default ``--prompt all``: one vLLM server session; question_only generated alone (no stop);
both R1 prompts share stop params in one batched request (tunable batch_size).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cs336_alignment.prompting_data_construction import (
    examples_from_rows,
    reward_fn_for_prompt,
    vllm_sampling_params,
)
from cs336_alignment.utils import load_data, load_prompt
from cs336_alignment.vllm_utils import VLLMServer


ROOT = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent
PROMPT_NAMES = ("question_only", "r1_zero", "r1_zero_three_shot")
PROMPTS = {
    "question_only": PKG / "prompts/question_only.prompt",
    "r1_zero": PKG / "prompts/r1_zero.prompt",
    "r1_zero_three_shot": PKG / "prompts/r1_zero_three_shot_gsm8k.prompt",
}
DEFAULT_BATCH_SIZE = 32

# ------------------------------------------------------------
# Load dataset rows
# ------------------------------------------------------------
def load_dataset_rows(dataset_path: Path, limit: int | None) -> list[dict]:
    rows = load_data(dataset_path)
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_examples_for_prompt(rows: list[dict], prompt_name: str) -> list[dict]:
    return examples_from_rows(rows, load_prompt(PROMPTS[prompt_name]))


def build_all_examples(dataset_path: Path, limit: int | None) -> dict[str, list[dict]]:
    rows = load_dataset_rows(dataset_path, limit)
    return {name: build_examples_for_prompt(rows, name) for name in PROMPT_NAMES}

# ------------------------------------------------------------
# Reward functions
# ------------------------------------------------------------
def summarize(prompt_name: str, examples: list[dict], responses: list[str]) -> dict[str, int]:
    reward_fn = reward_fn_for_prompt(prompt_name)
    counts = {"both_correct": 0, "format_only": 0, "neither": 0}
    for example, response in zip(examples, responses):
        scores = reward_fn(response, example["ground_truth"])
        if scores["format_reward"] == 1.0 and scores["answer_reward"] == 1.0:
            counts["both_correct"] += 1
        elif scores["format_reward"] == 1.0:
            counts["format_only"] += 1
        else:
            counts["neither"] += 1
    return counts

def print_counts(prompt_name: str, counts: dict[str, int]) -> None:
    total = counts["both_correct"] + counts["format_only"] + counts["neither"]
    print(f"prompt={prompt_name} n={total}")
    print(f"format=1 answer=1: {counts['both_correct']} ({counts['both_correct'] / total:.3f})")
    print(f"format=1 answer=0: {counts['format_only']} ({counts['format_only'] / total:.3f})")
    print(f"format=0 answer=0: {counts['neither']} ({counts['neither'] / total:.3f})")

# ------------------------------------------------------------
# Generate texts and evaluate on server
# ------------------------------------------------------------

def generate_texts(
    server: VLLMServer,
    prompts: list[str],
    sampling_key: str,
    seed: int,
    batch_size: int,
) -> list[str]:
    completions = server.generate_completions(
        prompts,
        vllm_sampling_params(sampling_key, seed),
        batch_size=batch_size,
    )
    return [item.text for item in completions]


def evaluate_on_server(
    server: VLLMServer,
    prompt_name: str,
    examples: list[dict],
    *,
    seed: int,
    batch_size: int,
) -> dict[str, int]:
    texts = generate_texts(
        server,
        [example["prompt"] for example in examples],
        prompt_name,
        seed,
        batch_size,
    )
    counts = summarize(prompt_name, examples, texts)
    print_counts(prompt_name, counts)
    return counts

def evaluate(
    prompt_name: str,
    examples: list[dict],
    model_id: str,
    *,
    seed: int = 0,
    gpu: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int]:
    server = VLLMServer(model_id=model_id, gpu=gpu, seed=seed)
    server.start()
    try:
        return evaluate_on_server(
            server,
            prompt_name,
            examples,
            seed=seed,
            batch_size=batch_size,
        )
    finally:
        server.stop()


def evaluate_all(
    examples_by_prompt: dict[str, list[dict]],
    model_id: str,
    *,
    seed: int,
    gpu: int,
    batch_size: int,
) -> dict[str, dict[str, int]]:
    server = VLLMServer(model_id=model_id, gpu=gpu, seed=seed)
    server.start()
    summaries: dict[str, dict[str, int]] = {}
    try:
        question_only_examples = examples_by_prompt["question_only"]
        summaries["question_only"] = summarize(
            "question_only",
            question_only_examples,
            generate_texts(
                server,
                [ex["prompt"] for ex in question_only_examples],
                "question_only",
                seed,
                batch_size,
            ),
        )

        r1_examples = examples_by_prompt["r1_zero"]
        r1_three_shot_examples = examples_by_prompt["r1_zero_three_shot"]
        n = len(r1_examples)
        assert n == len(r1_three_shot_examples)
        combined_prompts = [ex["prompt"] for ex in r1_examples] + [ex["prompt"] for ex in r1_three_shot_examples]
        combined_texts = generate_texts(
            server,
            combined_prompts,
            "r1_zero",
            seed,
            batch_size,
        )
        summaries["r1_zero"] = summarize("r1_zero", r1_examples, combined_texts[:n])
        summaries["r1_zero_three_shot"] = summarize(
            "r1_zero_three_shot",
            r1_three_shot_examples,
            combined_texts[n:],
        )
    finally:
        server.stop()

    for name in PROMPT_NAMES:
        print_counts(name, summaries[name])
        print()
    return summaries


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument(
        "--prompt",
        choices=(*PROMPT_NAMES, "all"),
        default="all",
        help="all: one vLLM session for all three prompts (recommended for §3.4)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    if args.prompt == "all":
        examples_by_prompt = build_all_examples(args.dataset, args.limit)
        evaluate_all(
            examples_by_prompt,
            args.model,
            seed=args.seed,
            gpu=args.gpu,
            batch_size=args.batch_size,
        )
    else:
        examples = build_examples_for_prompt(
            load_dataset_rows(args.dataset, args.limit),
            args.prompt,
        )
        evaluate(
            args.prompt,
            examples,
            args.model,
            seed=args.seed,
            gpu=args.gpu,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
