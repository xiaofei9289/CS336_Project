"""Measure tokenizer_experiments numbers using existing Tokenizer pickles.

Run from the repository root.  Does not train a tokenizer.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any

from tests.adapters import get_tokenizer

DELIMITER = "<|endoftext|>"
PILE_BYTES = 825 * (1024**3)


def load_tokenizer(path: Path):
    with path.open("rb") as file:
        artifact: dict[str, Any] = pickle.load(file)
    return get_tokenizer(
        artifact["vocab"],
        artifact["merges"],
        artifact["special_tokens"],
    )


def take_documents(path: Path, count: int) -> list[str]:
    documents: list[str] = []
    remainder = ""
    with path.open("r", encoding="utf-8", newline="") as file:
        while len(documents) < count:
            chunk = file.read(1 << 20)
            if not chunk:
                break
            buffer = remainder + chunk
            start = 0
            while len(documents) < count:
                found = buffer.find(DELIMITER, start)
                if found == -1:
                    break
                end = found + len(DELIMITER)
                documents.append(buffer[start:end])
                start = end
            remainder = buffer[start:]
    if remainder and len(documents) < count:
        documents.append(remainder)
    if len(documents) < count:
        raise RuntimeError(f"only found {len(documents)} documents in {path}")
    return documents


def encode_stats(tokenizer, text: str) -> dict[str, float | int]:
    utf8_bytes = len(text.encode("utf-8"))
    started = time.perf_counter()
    token_ids = tokenizer.encode(text)
    elapsed = time.perf_counter() - started
    n_tokens = len(token_ids)
    return {
        "utf8_bytes": utf8_bytes,
        "n_tokens": n_tokens,
        "bytes_per_token": utf8_bytes / n_tokens if n_tokens else float("nan"),
        "elapsed_seconds": elapsed,
        "tokens_per_second": n_tokens / elapsed if elapsed > 0 else 0.0,
        "bytes_per_second": utf8_bytes / elapsed if elapsed > 0 else 0.0,
    }


def corpus_stream_stats(tokenizer, path: Path) -> dict[str, float | int]:
    """Encode a full text file document-by-document with encode_iterable."""
    from encode_tinystories import iter_file_segments

    utf8_bytes = 0
    n_tokens = 0
    n_docs = 0
    started = time.perf_counter()
    for document in iter_file_segments(path, DELIMITER, 1 << 20):
        ids = list(tokenizer.encode_iterable([document]))
        n_tokens += len(ids)
        utf8_bytes += len(document.encode("utf-8"))
        n_docs += 1
        if n_docs % 2000 == 0:
            elapsed_so_far = time.perf_counter() - started
            print(
                f"  progress {path.name}: docs={n_docs:,} "
                f"bytes={utf8_bytes:,} tokens={n_tokens:,} "
                f"elapsed_s={elapsed_so_far:.1f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "utf8_bytes": utf8_bytes,
        "n_tokens": n_tokens,
        "n_docs": n_docs,
        "bytes_per_token": utf8_bytes / n_tokens if n_tokens else float("nan"),
        "elapsed_seconds": elapsed,
        "tokens_per_second": n_tokens / elapsed if elapsed > 0 else 0.0,
        "bytes_per_second": utf8_bytes / elapsed if elapsed > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ts-tokenizer",
        type=Path,
        default=Path("data/tinystories/tokenizer_train_10k.pkl"),
    )
    parser.add_argument(
        "--owt-tokenizer",
        type=Path,
        default=Path("data/owt/tokenizer_train_32k.pkl"),
    )
    parser.add_argument(
        "--ts-text",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.txt"),
    )
    parser.add_argument(
        "--owt-text",
        type=Path,
        default=Path("data/owt_train.txt"),
    )
    parser.add_argument(
        "--ts-stream-text",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-valid.txt"),
        help="Smaller TinyStories file for matched-corpus compression and throughput.",
    )
    parser.add_argument(
        "--owt-stream-text",
        type=Path,
        default=Path("data/owt_valid.txt"),
        help="Smaller OpenWebText file for matched-corpus compression and throughput.",
    )
    parser.add_argument("--n-docs", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/tokenizer_experiments/results.json"),
    )
    args = parser.parse_args()

    ts_tok = load_tokenizer(args.ts_tokenizer)
    owt_tok = load_tokenizer(args.owt_tokenizer)

    ts_docs = take_documents(args.ts_text, args.n_docs)
    owt_docs = take_documents(args.owt_text, args.n_docs)
    ts_joined = "".join(ts_docs)
    owt_joined = "".join(owt_docs)

    cross = {
        "ts_tokenizer_on_10_ts_docs": encode_stats(ts_tok, ts_joined),
        "owt_tokenizer_on_10_owt_docs": encode_stats(owt_tok, owt_joined),
        "ts_tokenizer_on_10_owt_docs": encode_stats(ts_tok, owt_joined),
        "owt_tokenizer_on_10_ts_docs": encode_stats(owt_tok, ts_joined),
    }

    print("=== 10-document cross encoding ===")
    for name, stats in cross.items():
        print(
            f"{name}: bytes={stats['utf8_bytes']:,} tokens={stats['n_tokens']:,} "
            f"bytes/token={stats['bytes_per_token']:.4f}"
        )

    print("\n=== Matched corpus stream (valid splits) ===")
    matched = {
        "ts_tokenizer_on_ts_valid": corpus_stream_stats(
            ts_tok, args.ts_stream_text
        ),
        "owt_tokenizer_on_owt_valid": corpus_stream_stats(
            owt_tok, args.owt_stream_text
        ),
    }
    for name, stats in matched.items():
        print(
            f"{name}: bytes={stats['utf8_bytes']:,} tokens={stats['n_tokens']:,} "
            f"bytes/token={stats['bytes_per_token']:.4f} "
            f"elapsed_s={stats['elapsed_seconds']:.2f} "
            f"bytes/s={stats['bytes_per_second']:,.0f} "
            f"tokens/s={stats['tokens_per_second']:,.0f}"
        )

    # Throughput: reuse the longer of the two matched runs (OWT valid).
    owt_valid = matched["owt_tokenizer_on_owt_valid"]
    bytes_per_second = float(owt_valid["bytes_per_second"])
    pile_seconds = PILE_BYTES / bytes_per_second if bytes_per_second else float("inf")
    pile = {
        "pile_bytes": PILE_BYTES,
        "pile_gib": 825,
        "reference_run": "owt_tokenizer_on_owt_valid",
        "bytes_per_second": bytes_per_second,
        "tokens_per_second": owt_valid["tokens_per_second"],
        "estimated_seconds": pile_seconds,
        "estimated_hours": pile_seconds / 3600,
        "estimated_days": pile_seconds / 86400,
    }
    print("\n=== Pile estimate (825 GiB) ===")
    print(
        f"rate={bytes_per_second:,.0f} bytes/s -> "
        f"{pile['estimated_hours']:.1f} hours "
        f"({pile['estimated_days']:.2f} days)"
    )

    payload = {
        "machine": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "tokenizers": {
            "tiny_stories": str(args.ts_tokenizer),
            "owt": str(args.owt_tokenizer),
        },
        "ten_documents": cross,
        "matched_corpus_valid": matched,
        "pile_estimate": pile,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved={args.output}")


if __name__ == "__main__":
    main()
