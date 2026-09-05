"""Create tiny uint16 memmap files for an end-to-end training smoke test.

The validation file intentionally contains the same token stream as the
training file.  This is only for the one-batch overfitting check, not for a
real experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/dummy"))
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_memmap(path: Path, tokens: np.ndarray) -> None:
    array = np.memmap(path, dtype=np.uint16, mode="w+", shape=tokens.shape)
    array[:] = tokens
    array.flush()
    del array


def main() -> None:
    args = parse_args()
    if not 2 <= args.vocab_size <= np.iinfo(np.uint16).max + 1:
        raise ValueError("vocab_size must be between 2 and 65536 for uint16")
    if args.num_tokens < 2:
        raise ValueError("num_tokens must be at least 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tokens = rng.integers(
        low=0,
        high=args.vocab_size,
        size=args.num_tokens,
        dtype=np.uint16,
    )

    train_path = args.output_dir / "train.bin"
    valid_path = args.output_dir / "valid.bin"
    write_memmap(train_path, tokens)
    write_memmap(valid_path, tokens)

    print(f"wrote {train_path} ({args.num_tokens:,} tokens)")
    print(f"wrote {valid_path} ({args.num_tokens:,} tokens)")
    print("dtype=uint16; train and valid are intentionally identical")


if __name__ == "__main__":
    main()
