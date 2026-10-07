"""Per-example DPO loss. The returned tensor lives on the policy's device."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def _ids_with_boundaries(tokenizer: PreTrainedTokenizerBase, text: str, *, add_eos: bool) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        token_ids = [tokenizer.bos_token_id] + token_ids
    if add_eos and tokenizer.eos_token_id is not None:
        token_ids = token_ids + [tokenizer.eos_token_id]
    return token_ids


def response_log_prob(model, tokenizer: PreTrainedTokenizerBase, prompt: str, response: str) -> torch.Tensor:
    template_path = Path(__file__).resolve().parent / "prompts_safety" / "alpaca_sft.prompt"
    template = template_path.read_text().rstrip("\n")
    prefix = template[: template.index("{response}")].format(instruction=prompt)
    prefix_ids = _ids_with_boundaries(tokenizer, prefix, add_eos=False)
    full_ids = _ids_with_boundaries(tokenizer, prefix + response, add_eos=True)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("response tokenization is not a continuation of the prompt")
    device = next(model.parameters()).device
    token_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = model(token_ids[:, :-1]).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    labels = token_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]
    return token_log_probs[len(prefix_ids) - 1 :].sum()


def per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    chosen = response_log_prob(lm, tokenizer, prompt, response_chosen)
    rejected = response_log_prob(lm, tokenizer, prompt, response_rejected)
    chosen_ref = response_log_prob(lm_ref, tokenizer, prompt, response_chosen).to(chosen.device)
    rejected_ref = response_log_prob(lm_ref, tokenizer, prompt, response_rejected).to(chosen.device)
    logits = beta * ((chosen - chosen_ref) - (rejected - rejected_ref))
    return -F.logsigmoid(logits)
