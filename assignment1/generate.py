"""Use a trained TinyStories model to generate text token by token.

Place this file in the repository root, alongside the ``cs336_basics`` folder.
This script performs inference only; it does not train or change the model.
"""

from __future__ import annotations

import argparse
import gc
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from cs336_basics.checkpoint import load_checkpoint
from cs336_basics.layers import softmax
from cs336_basics.optimizer import AdamW
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM


# These values must exactly match the training configuration.
# 词表大小在 main() 中从 tokenizer 自动读取。
# 以下结构超参数必须与训练配置一致。
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1_344
ROPE_THETA = 10_000.0


def load_tokenizer(
    tokenizer_path: Path,
) -> tuple[Tokenizer, dict[int, bytes], list[str]]:
    """Load vocab, merges, and special tokens from the tokenizer pickle."""
    with tokenizer_path.open("rb") as file:
        saved: Any = pickle.load(file)

    # Also accept a complete Tokenizer object, although the expected format is
    # the dictionary written by the tokenizer-training script.
    if callable(getattr(saved, "encode", None)) and callable(
        getattr(saved, "decode", None)
    ):
        vocab = dict(saved.vocab)
        special_tokens = getattr(saved, "special_tokens", [])
        if isinstance(special_tokens, Mapping):
            special_tokens = list(special_tokens.keys())
        return saved, vocab, list(special_tokens)

    if isinstance(saved, Mapping):
        try:
            vocab = dict(saved["vocab"])
            merges = list(saved["merges"])
        except KeyError as error:
            raise ValueError(
                "Tokenizer pickle must contain 'vocab' and 'merges'."
            ) from error
        special_tokens = saved.get("special_tokens", [])
    elif isinstance(saved, (tuple, list)) and len(saved) == 3:
        vocab = dict(saved[0])
        merges = list(saved[1])
        special_tokens = saved[2]
    else:
        raise ValueError(
            "Unsupported tokenizer pickle. Expected a dictionary containing "
            "vocab, merges, and special_tokens."
        )

    if isinstance(special_tokens, Mapping):
        special_tokens = list(special_tokens.keys())
    else:
        special_tokens = list(special_tokens)

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
    )
    return tokenizer, vocab, special_tokens


def find_token_id(vocab: Mapping[int, bytes], token: str) -> int | None:
    """Return the id whose vocabulary bytes exactly equal ``token``."""
    wanted = token.encode("utf-8")
    matches = [token_id for token_id, value in vocab.items() if value == wanted]

    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Token {token!r} has multiple ids: {matches}")
    return int(matches[0])


def choose_device(requested: str) -> torch.device:
    """Resolve the requested inference device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")

    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested, but MPS is not available.")

    return torch.device(requested)


def sample_next_token(
    logits: Tensor,
    temperature: float,
    top_p: float,
) -> int:
    """Apply temperature and top-p, then sample one token id.

    ``temperature=0`` performs greedy argmax decoding. ``top_p=1`` disables
    top-p filtering and samples from the full softmax distribution.
    """
    if logits.ndim != 1:
        raise ValueError(f"Expected 1-D logits, got {tuple(logits.shape)}")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    if temperature == 0:
        return int(torch.argmax(logits).item())

    # Compute probabilities in FP32 even if the model uses BF16/FP16.
    probabilities = softmax(logits.float() / temperature, dim=-1)

    if top_p < 1:
        sorted_probabilities, sorted_ids = torch.sort(
            probabilities,
            descending=True,
        )
        cumulative_probabilities = torch.cumsum(
            sorted_probabilities,
            dim=-1,
        )

        # Keep the smallest prefix whose cumulative probability reaches top_p.
        remove = cumulative_probabilities >= top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False

        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
        sorted_probabilities /= sorted_probabilities.sum()

        sampled_position = torch.multinomial(
            sorted_probabilities,
            num_samples=1,
        )
        return int(sorted_ids[sampled_position].item())

    return int(torch.multinomial(probabilities, num_samples=1).item())


def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    eos_id: int | None,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Return ``(prompt_plus_generation, newly_generated_ids)``."""
    token_ids = list(prompt_ids)
    hidden_seed = False

    # A forward pass cannot use an empty sequence. Treat EOS as a document
    # boundary for an empty prompt, then hide that artificial seed in output.
    if not token_ids:
        if eos_id is None:
            raise ValueError(
                "The prompt is empty and EOS is unavailable. Use a non-empty "
                "--prompt."
            )
        token_ids.append(eos_id)
        hidden_seed = True

    generated_ids: list[int] = []

    with torch.inference_mode():
        for _ in range(max_tokens):
            # The model was trained with context_length=256. After the sequence
            # grows beyond it, condition only on the most recent 256 ids.
            recent_ids = token_ids[-model.context_length :]
            model_input = torch.tensor(
                [recent_ids],
                dtype=torch.long,
                device=device,
            )

            all_logits = model(model_input)
            last_position_logits = all_logits[0, -1, :]
            next_id = sample_next_token(
                last_position_logits,
                temperature=temperature,
                top_p=top_p,
            )

            token_ids.append(next_id)
            generated_ids.append(next_id)

            if eos_id is not None and next_id == eos_id:
                break

    visible_ids = token_ids[1:] if hidden_seed else token_ids
    return visible_ids, generated_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a story from checkpoint_final.pt."
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="0 = argmax; 1 = original softmax distribution (default: 0.8)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="1 disables top-p filtering (default: 0.9)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="maximum number of new tokens (default: 256)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"D:\desktop\runs\tinystories\checkpoint_final.pt"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tinystories/tokenizer_train_10k.pkl"),
    )

    # 新增：指定后，将原始文本和生成设置保存到这个目录。
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generation.txt and metadata.txt.",
    )

    
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--show-token-ids", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")

    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer, vocab, special_tokens = load_tokenizer(args.tokenizer)

    # 自动适配 TinyStories 的 10k 或 OWT 的 32k 词表。
    vocab_size = len(vocab)

    # 模型要求 token ID 连续覆盖 0 到 vocab_size - 1。
    if vocab_size == 0:
        raise ValueError("Tokenizer vocabulary is empty.")

    if set(vocab.keys()) != set(range(vocab_size)):
        raise ValueError(
            "Tokenizer token IDs must be contiguous from 0 "
            f"to {vocab_size - 1}."
        )

    if args.eos_token not in special_tokens:
        print(
            f"Warning: {args.eos_token!r} is not listed as a special token."
        )

    eos_id = find_token_id(vocab, args.eos_token)
    if eos_id is None:
        print("Warning: EOS id not found; generation will stop at max-tokens.")

    # load_checkpoint requires an optimizer. Load everything on CPU first, then
    # delete the unused optimizer state before moving the model onto the GPU.
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
        device="cpu",
    )
    optimizer = AdamW(model.parameters())
    iteration = load_checkpoint(args.checkpoint, model, optimizer)
    del optimizer
    gc.collect()

    model = model.to(device)
    model.eval()

    prompt_ids = [int(token_id) for token_id in tokenizer.encode(args.prompt)]
    invalid_ids = [
        token_id
        for token_id in prompt_ids
        if not 0 <= token_id < vocab_size
    ]
    if invalid_ids:
        raise ValueError(f"Prompt produced invalid token ids: {invalid_ids[:10]}")

    all_ids, generated_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=eos_id,
        device=device,
    )

    stopped_by_eos = (
        eos_id is not None
        and bool(generated_ids)
        and generated_ids[-1] == eos_id
    )
    stop_reason = "EOS" if stopped_by_eos else "max_tokens"

    # 与原脚本的显示内容一致：prompt + 新生成文本。
    # 直接解码，不做润色、删改或 strip()。
    raw_text = tokenizer.decode(all_ids)

    metadata = "\n".join(
        [
            f"Checkpoint: {args.checkpoint}",
            f"Tokenizer: {args.tokenizer}",
            f"Vocabulary size: {vocab_size}",
            f"Loaded checkpoint iteration: {iteration}",
            f"Device: {device}",
            f"Prompt: {args.prompt!r}",
            f"Temperature: {args.temperature}",
            f"Top-p: {args.top_p}",
            f"Seed: {args.seed}",
            f"Maximum new tokens: {args.max_tokens}",
            f"Prompt tokens: {len(prompt_ids)}",
            f"Generated tokens: {len(generated_ids)}",
            f"EOS token: {args.eos_token!r}",
            f"EOS token ID: {eos_id}",
            f"Stop reason: {stop_reason}",
        ]
    )

    print(metadata)
    print("\n--- generated text ---\n")
    print(raw_text)

    if args.show_token_ids:
        print("\n--- generated token ids ---\n")
        print(generated_ids)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        text_path = args.output_dir / "generation.txt"
        metadata_path = args.output_dir / "metadata.txt"

        text_path.write_text(raw_text, encoding="utf-8")
        metadata_path.write_text(metadata + "\n", encoding="utf-8")

        print(f"\nSaved generated text: {text_path}")
        print(f"Saved generation settings: {metadata_path}")


if __name__ == "__main__":
    main()
