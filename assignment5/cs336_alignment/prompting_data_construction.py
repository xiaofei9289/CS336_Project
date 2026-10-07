"""GSM8K prompts, rewards, and vLLM sampling shared by baselines and GRPO."""

from __future__ import annotations

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn


def extract_gsm8k_answer(answer_field: str) -> str:
    """GSM8K answers are `{rationale} #### {final}`; rewards use only the part after ####."""
    return answer_field.split("####")[-1].strip()


def render_prompt(template: str, question: str) -> str:
    return template.replace("{question}", question)


def examples_from_rows(rows: list[dict], template: str) -> list[dict]:
    return [
        {
            "question": row["question"],
            "ground_truth": extract_gsm8k_answer(row["answer"]),
            "prompt": render_prompt(template, row["question"]),
        }
        for row in rows
    ]


def reward_fn_for_prompt(prompt_name: str):
    if prompt_name.startswith("question_only"):
        return question_only_reward_fn
    return r1_zero_reward_fn


def vllm_sampling_params(
    prompt_name: str,
    seed: int,
    *,
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> dict:
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": 1,
        "seed": seed,
    }
    if not prompt_name.startswith("question_only"):
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    return params
