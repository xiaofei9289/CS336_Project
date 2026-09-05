"""Train and save a 10k TinyStories BPE tokenizer.

Run this script from the Assignment01 repository root.  It uses the
run_train_bpe adapter that already passed your unit tests.
"""

from __future__ import annotations

import argparse
import cProfile
import pickle
import pstats
import sys
import time
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # resource is unavailable on Windows.
    resource = None

from tests.adapters import run_train_bpe


DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BPE on TinyStories and save vocab/merges as pickle."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-valid.txt"),
        help="Training corpus. Start with TinyStories valid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tinystories/tokenizer_valid_10k.pkl"),
        help="Output pickle containing vocab, merges, and run metadata.",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument(
        "--desired-num-chunks",
        type=int,
        default=32,
        help="How many special-token-aligned chunks to use when reading the corpus.",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        help="Repeat this option to add multiple special tokens.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Write a .prof file and print the 30 most expensive calls.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output pickle.",
    )
    return parser.parse_args()


def peak_rss_mib() -> float | None:
    """Return maximum resident set size for this process in MiB."""
    if resource is None:
        return None
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        # macOS reports bytes.
        return maximum_rss / (1024**2)
    # Linux and most Unix variants report KiB.
    return maximum_rss / 1024


def validate_result(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_size: int,
    special_tokens: list[str],
) -> None:
    if not isinstance(vocab, dict):
        raise TypeError(f"vocab must be dict, got {type(vocab).__name__}")
    if not isinstance(merges, list):
        raise TypeError(f"merges must be list, got {type(merges).__name__}")
    if len(vocab) != vocab_size:
        raise ValueError(f"expected {vocab_size} vocab entries, got {len(vocab)}")

    for token_id, token_bytes in vocab.items():
        if not isinstance(token_id, int) or not isinstance(token_bytes, bytes):
            raise TypeError(
                "vocab must have type dict[int, bytes]; "
                f"found {type(token_id).__name__} -> {type(token_bytes).__name__}"
            )

    for index, merge in enumerate(merges):
        if (
            not isinstance(merge, tuple)
            or len(merge) != 2
            or not isinstance(merge[0], bytes)
            or not isinstance(merge[1], bytes)
        ):
            raise TypeError(
                "merges must have type list[tuple[bytes, bytes]]; "
                f"bad item at index {index}: {merge!r}"
            )

    vocab_values = set(vocab.values())
    missing_special_tokens = [
        token for token in special_tokens if token.encode("utf-8") not in vocab_values
    ]
    if missing_special_tokens:
        raise ValueError(f"special tokens missing from vocab: {missing_special_tokens}")

    expected_ids = set(range(vocab_size))
    if set(vocab) != expected_ids:
        raise ValueError("vocab IDs are not contiguous from 0 to vocab_size - 1")


def describe_longest_token(
    vocab: dict[int, bytes],
    special_tokens: list[str],
) -> tuple[tuple[int, bytes], tuple[int, bytes]]:
    longest_overall = max(vocab.items(), key=lambda item: len(item[1]))
    special_token_bytes = {token.encode("utf-8") for token in special_tokens}
    ordinary_tokens = [
        item for item in vocab.items() if item[1] not in special_token_bytes
    ]
    longest_ordinary = max(ordinary_tokens, key=lambda item: len(item[1]))
    return longest_overall, longest_ordinary


def print_token(label: str, token: tuple[int, bytes]) -> None:
    token_id, token_bytes = token
    decoded = token_bytes.decode("utf-8", errors="replace")
    print(f"{label}_id={token_id}")
    print(f"{label}_bytes_length={len(token_bytes)}")
    print(f"{label}_bytes={token_bytes!r}")
    print(f"{label}_decoded={decoded!r}")


def save_and_verify(output: Path, artifact: dict[str, Any]) -> None:
    temporary_output = output.with_name(f".{output.name}.tmp")
    with temporary_output.open("wb") as file:
        pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_output.replace(output)

    with output.open("rb") as file:
        restored = pickle.load(file)
    if restored["vocab"] != artifact["vocab"]:
        raise RuntimeError("saved vocab did not round-trip exactly")
    if restored["merges"] != artifact["merges"]:
        raise RuntimeError("saved merges did not round-trip exactly")


def main() -> None:
    args = parse_args()
    special_tokens = args.special_tokens or [DEFAULT_SPECIAL_TOKEN]
    special_tokens = list(dict.fromkeys(special_tokens))

    if not args.input.is_file():
        raise FileNotFoundError(f"corpus not found: {args.input}")
    if args.vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small for byte tokens and special tokens")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; pass --overwrite to replace it"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_size_mib = args.input.stat().st_size / (1024**2)
    print(f"input={args.input}")
    print(f"input_size_mib={source_size_mib:.2f}")
    print(f"vocab_size={args.vocab_size}")
    print(f"desired_num_chunks={args.desired_num_chunks}")
    print(f"special_tokens={special_tokens}")

    profiler = cProfile.Profile() if args.profile else None
    profile_path = args.output.with_suffix(".prof")
    started_at = time.perf_counter()

    if profiler is not None:
        profiler.enable()
    try:
        vocab, merges = run_train_bpe(
            input_path=args.input,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            desired_num_chunks=args.desired_num_chunks,
        )
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(profile_path)
            print(f"profile={profile_path}")
            pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(30)

    elapsed_seconds = time.perf_counter() - started_at
    maximum_rss_mib = peak_rss_mib()
    validate_result(vocab, merges, args.vocab_size, special_tokens)

    unique_special_count = len(special_tokens)
    expected_merge_count = args.vocab_size - 256 - unique_special_count
    if len(merges) != expected_merge_count:
        print(
            "warning: merge count differs from the usual expectation: "
            f"expected={expected_merge_count}, actual={len(merges)}"
        )

    longest_overall, longest_ordinary = describe_longest_token(vocab, special_tokens)
    artifact: dict[str, Any] = {
        "vocab": vocab,
        "merges": merges,
        "special_tokens": special_tokens,
        "metadata": {
            "input_path": str(args.input),
            "input_size_bytes": args.input.stat().st_size,
            "vocab_size": args.vocab_size,
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_mib": maximum_rss_mib,
            "longest_token_id": longest_overall[0],
            "longest_token_bytes": longest_overall[1],
            "longest_ordinary_token_id": longest_ordinary[0],
            "longest_ordinary_token_bytes": longest_ordinary[1],
        },
    }
    save_and_verify(args.output, artifact)

    print("\n=== BPE training summary ===")
    print(f"vocab_entries={len(vocab)}")
    print(f"merges={len(merges)}")
    print(f"elapsed_seconds={elapsed_seconds:.3f}")
    if maximum_rss_mib is not None:
        print(f"peak_rss_mib={maximum_rss_mib:.2f}")
    else:
        print("peak_rss_mib=unavailable")
    print_token("longest_token", longest_overall)
    print_token("longest_ordinary_token", longest_ordinary)
    print(f"saved={args.output}")
    print("pickle_round_trip=ok")


if __name__ == "__main__":
    main()
