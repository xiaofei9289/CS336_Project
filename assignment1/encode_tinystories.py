"""Stream TinyStories text into a raw uint16 token-ID file.

Run from the Assignment01 repository root.  The script loads the pickle made by
train_bpe_tinystories.py and uses the Tokenizer already connected in
tests/adapters.py.
"""

from __future__ import annotations

import argparse
import itertools
import pickle
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from tests.adapters import get_tokenizer


UINT16_LIMIT = np.iinfo(np.uint16).max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode TinyStories into a raw uint16 memmap-compatible file."
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tinystories/tokenizer_valid_10k.pkl"),
        help="Pickle containing vocab, merges, and special_tokens.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-valid.txt"),
        help="TinyStories text file to encode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tinystories/valid.bin"),
        help="Raw uint16 output file.",
    )
    parser.add_argument(
        "--read-chars",
        type=int,
        default=1 << 20,
        help="Characters read from the text file at a time.",
    )
    parser.add_argument(
        "--write-buffer-tokens",
        type=int,
        default=1_000_000,
        help="Token IDs buffered before each binary write.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Print progress after approximately this many token IDs.",
    )
    parser.add_argument(
        "--sample-tokens",
        type=int,
        default=256,
        help="Maximum number of token IDs in the decoded sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output file.",
    )
    return parser.parse_args()


def load_tokenizer_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        artifact = pickle.load(file)

    if not isinstance(artifact, dict):
        raise TypeError("tokenizer pickle must contain a dictionary")
    required_keys = {"vocab", "merges", "special_tokens"}
    missing = required_keys - artifact.keys()
    if missing:
        raise KeyError(f"tokenizer pickle is missing keys: {sorted(missing)}")

    vocab = artifact["vocab"]
    merges = artifact["merges"]
    special_tokens = artifact["special_tokens"]
    if not isinstance(vocab, dict) or not all(
        isinstance(key, int) and isinstance(value, bytes)
        for key, value in vocab.items()
    ):
        raise TypeError("vocab must have type dict[int, bytes]")
    if not isinstance(merges, list) or not all(
        isinstance(pair, tuple)
        and len(pair) == 2
        and isinstance(pair[0], bytes)
        and isinstance(pair[1], bytes)
        for pair in merges
    ):
        raise TypeError("merges must have type list[tuple[bytes, bytes]]")
    if not isinstance(special_tokens, list) or not all(
        isinstance(token, str) for token in special_tokens
    ):
        raise TypeError("special_tokens must have type list[str]")
    if not special_tokens:
        raise ValueError("at least one special token is required")
    if len(vocab) > UINT16_LIMIT + 1:
        raise ValueError("vocabulary does not fit in uint16")
    return artifact


def iter_delimited_segments(
    raw_chunks: Iterable[str],
    delimiter: str,
) -> Iterator[str]:
    """Yield text ending at complete delimiters, even across raw chunk edges."""
    if not delimiter:
        raise ValueError("delimiter must not be empty")

    remainder = ""
    for raw_chunk in raw_chunks:
        if not raw_chunk:
            continue
        buffer = remainder + raw_chunk
        segment_start = 0

        while True:
            delimiter_start = buffer.find(delimiter, segment_start)
            if delimiter_start == -1:
                break
            segment_end = delimiter_start + len(delimiter)
            yield buffer[segment_start:segment_end]
            segment_start = segment_end

        remainder = buffer[segment_start:]

    if remainder:
        yield remainder


def iter_file_chunks(path: Path, read_chars: int) -> Iterator[str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        while True:
            chunk = file.read(read_chars)
            if not chunk:
                return
            yield chunk


def iter_file_segments(
    path: Path,
    delimiter: str,
    read_chars: int,
) -> Iterator[str]:
    return iter_delimited_segments(iter_file_chunks(path, read_chars), delimiter)


def find_special_token_id(vocab: dict[int, bytes], special_token: str) -> int:
    target = special_token.encode("utf-8")
    matches = [token_id for token_id, token_bytes in vocab.items() if token_bytes == target]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one ID for {special_token!r}, found {matches}"
        )
    return matches[0]


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def check_chunked_encoding(
    tokenizer: Any,
    input_path: Path,
    delimiter: str,
    read_chars: int,
) -> None:
    # Two delimiter-terminated segments are enough to test a real document
    # boundary.  This reads only a tiny prefix of the corpus.
    fragment = "".join(
        itertools.islice(
            iter_file_segments(input_path, delimiter, read_chars),
            2,
        )
    )
    if not fragment:
        raise ValueError("input corpus is empty")

    whole_ids = list(tokenizer.encode(fragment))

    # Seven-character raw chunks deliberately make it likely that the special
    # token spans multiple raw reads.  iter_delimited_segments must repair it.
    artificial_chunks = (
        fragment[start : start + 7] for start in range(0, len(fragment), 7)
    )
    safe_segments = iter_delimited_segments(artificial_chunks, delimiter)
    chunked_ids = list(tokenizer.encode_iterable(safe_segments))

    mismatch = first_difference(whole_ids, chunked_ids)
    if mismatch is not None:
        left = whole_ids[mismatch : mismatch + 10]
        right = chunked_ids[mismatch : mismatch + 10]
        raise AssertionError(
            "whole encode and chunked encode_iterable differ at token index "
            f"{mismatch}; whole={left}, chunked={right}, "
            f"lengths=({len(whole_ids)}, {len(chunked_ids)})"
        )
    print(
        "chunk_equivalence=ok "
        f"(characters={len(fragment):,}, tokens={len(whole_ids):,})"
    )


def flush_ids(output_file: Any, token_buffer: list[int]) -> None:
    if token_buffer:
        np.asarray(token_buffer, dtype=np.uint16).tofile(output_file)
        token_buffer.clear()


def encode_to_uint16(
    tokenizer: Any,
    input_path: Path,
    temporary_output: Path,
    delimiter: str,
    special_token_id: int,
    vocab_size: int,
    read_chars: int,
    write_buffer_tokens: int,
    progress_every: int,
) -> tuple[int, int, int, int]:
    token_buffer: list[int] = []
    total_tokens = 0
    minimum_id = vocab_size
    maximum_id = -1
    encoded_special_count = 0
    source_special_count = 0
    next_progress = progress_every

    def observed_segments() -> Iterator[str]:
        nonlocal source_special_count
        for segment in iter_file_segments(input_path, delimiter, read_chars):
            source_special_count += segment.count(delimiter)
            yield segment

    started_at = time.perf_counter()
    with temporary_output.open("wb") as output_file:
        for raw_token_id in tokenizer.encode_iterable(observed_segments()):
            token_id = int(raw_token_id)
            if not 0 <= token_id < vocab_size:
                raise ValueError(
                    f"out-of-range token ID {token_id}; expected 0 <= id < {vocab_size}"
                )
            if token_id > UINT16_LIMIT:
                raise ValueError(f"token ID {token_id} does not fit in uint16")

            token_buffer.append(token_id)
            total_tokens += 1
            minimum_id = min(minimum_id, token_id)
            maximum_id = max(maximum_id, token_id)
            if token_id == special_token_id:
                encoded_special_count += 1

            if len(token_buffer) >= write_buffer_tokens:
                flush_ids(output_file, token_buffer)
            if progress_every > 0 and total_tokens >= next_progress:
                elapsed = time.perf_counter() - started_at
                rate = total_tokens / elapsed if elapsed > 0 else 0.0
                print(f"encoded={total_tokens:,} tokens ({rate:,.0f} tokens/s)")
                next_progress += progress_every

        flush_ids(output_file, token_buffer)

    if total_tokens == 0:
        raise ValueError("tokenizer produced no token IDs")
    if source_special_count == 0:
        raise ValueError(f"no {delimiter!r} delimiter was found in the source file")
    if encoded_special_count != source_special_count:
        raise AssertionError(
            "special-token count mismatch: "
            f"source={source_special_count}, encoded={encoded_special_count}"
        )
    return total_tokens, minimum_id, maximum_id, encoded_special_count


def find_first_id(ids: np.memmap, target_id: int) -> int | None:
    scan_size = 100_000
    for start in range(0, len(ids), scan_size):
        stop = min(start + scan_size, len(ids))
        local_positions = np.flatnonzero(np.asarray(ids[start:stop]) == target_id)
        if len(local_positions):
            return start + int(local_positions[0])
    return None


def decode_sample(
    tokenizer: Any,
    ids: np.memmap,
    special_token_id: int,
    sample_tokens: int,
) -> str:
    special_position = find_first_id(ids, special_token_id)
    if special_position is None:
        raise AssertionError("encoded memmap contains no special-token ID")

    tokens_before_special = max(0, sample_tokens - 8)
    start = max(0, special_position - tokens_before_special)
    stop = min(len(ids), start + sample_tokens)
    return tokenizer.decode(np.asarray(ids[start:stop], dtype=np.int64).tolist())


def main() -> None:
    args = parse_args()
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer pickle not found: {args.tokenizer}")
    if not args.input.is_file():
        raise FileNotFoundError(f"input corpus not found: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; pass --overwrite to replace it"
        )
    if args.read_chars < 1 or args.write_buffer_tokens < 1:
        raise ValueError("read-chars and write-buffer-tokens must be positive")
    if args.sample_tokens < 1:
        raise ValueError("sample-tokens must be positive")

    artifact = load_tokenizer_artifact(args.tokenizer)
    vocab: dict[int, bytes] = artifact["vocab"]
    merges: list[tuple[bytes, bytes]] = artifact["merges"]
    special_tokens: list[str] = artifact["special_tokens"]
    if "<|endoftext|>" not in special_tokens:
        raise ValueError("tokenizer artifact does not contain <|endoftext|>")

    delimiter = "<|endoftext|>"
    special_token_id = find_special_token_id(vocab, delimiter)
    tokenizer = get_tokenizer(vocab, merges, special_tokens)

    print(f"tokenizer={args.tokenizer}")
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"vocab_size={len(vocab):,}")
    print(f"special_token_id={special_token_id}")

    check_chunked_encoding(
        tokenizer,
        args.input,
        delimiter,
        args.read_chars,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    started_at = time.perf_counter()
    total_tokens, tracked_min, tracked_max, special_count = encode_to_uint16(
        tokenizer=tokenizer,
        input_path=args.input,
        temporary_output=temporary_output,
        delimiter=delimiter,
        special_token_id=special_token_id,
        vocab_size=len(vocab),
        read_chars=args.read_chars,
        write_buffer_tokens=args.write_buffer_tokens,
        progress_every=args.progress_every,
    )
    elapsed_seconds = time.perf_counter() - started_at

    item_size = np.dtype(np.uint16).itemsize
    file_size = temporary_output.stat().st_size
    if file_size % item_size != 0:
        raise AssertionError(
            f"output byte length {file_size} is not divisible by {item_size}"
        )
    if file_size // item_size != total_tokens:
        raise AssertionError(
            "file length does not match the number of encoded tokens: "
            f"file={file_size // item_size}, encoded={total_tokens}"
        )

    ids = np.memmap(temporary_output, dtype=np.uint16, mode="r")
    memmap_min = int(ids.min())
    memmap_max = int(ids.max())
    if memmap_min != tracked_min or memmap_max != tracked_max:
        raise AssertionError(
            "memmap min/max differs from streaming counters: "
            f"memmap=({memmap_min}, {memmap_max}), "
            f"stream=({tracked_min}, {tracked_max})"
        )
    if memmap_min < 0 or memmap_max >= len(vocab):
        raise AssertionError(
            f"invalid ID range [{memmap_min}, {memmap_max}] for vocab {len(vocab)}"
        )

    sample = decode_sample(
        tokenizer,
        ids,
        special_token_id,
        args.sample_tokens,
    )
    if delimiter not in sample:
        raise AssertionError("decoded sample does not contain the complete special token")

    # Publish valid.bin only after every self-check has passed.
    del ids
    temporary_output.replace(args.output)

    rate = total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    print("\n=== Encoding summary ===")
    print(f"total_tokens={total_tokens:,}")
    print(f"id_min={memmap_min}")
    print(f"id_max={memmap_max}")
    print(f"special_token_count={special_count:,}")
    print(f"file_size_bytes={file_size:,}")
    print(f"file_size_mib={file_size / (1024**2):.2f}")
    print(f"elapsed_seconds={elapsed_seconds:.3f}")
    print(f"tokens_per_second={rate:,.0f}")
    print("memmap_open=ok")
    print("uint16_size_check=ok")
    print("id_range_check=ok")
    print("special_token_count_check=ok")
    print("decode_check=ok")
    print("\n=== Decode sample ===")
    print(sample)
    print("=== End decode sample ===")


if __name__ == "__main__":
    main()
