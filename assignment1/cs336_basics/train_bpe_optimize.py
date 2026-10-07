import os
import time
from collections import Counter
from multiprocessing import Pool

from .pretokenization_example import find_chunk_boundaries
# from .bpe import (
#     build_pair_heap,
#     build_pair_index,
#     initialize_vocab,
#     merge_incrementally,
#     pretokenize,
#     push_pair_heap_updates,
#     select_best_pair_from_heap,
# )
from .bpe_optimize import (
    build_pair_heap,
    build_pair_index,
    initialize_vocab,
    merge_incrementally,
    pretokenize,
    push_pair_heap_updates,
    select_best_pair_from_heap,
)

# Steps for training a byte-level BPE tokenizer

# Step 1: initialize the vocabulary

# Step 2: merge a pair

# Step 3: add the new token

# Step 4: repeat steps 2 and 3 until the vocabulary reaches the target size

# Step 5: save the vocabulary


ChunkTask = tuple[
    str | bytes,
    int,
    int,
    tuple[str, ...],
]


def _pretokenize_chunk(
    task: ChunkTask,
) -> Counter[tuple[int, ...]]:
    """Read and pre-tokenize one ``[start, end)`` file span in a worker process."""
    input_path, start, end, special_tokens = task

    # A worker receives only a path and integer offsets. The raw bytes and the
    # decoded str stay inside that worker and are not sent across processes.
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk_text = file.read(end - start).decode("utf-8")

    return pretokenize(
        text=chunk_text,
        special_tokens=list(special_tokens),
    )


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    """Train a byte-level BPE tokenizer on the given corpus.

    Returns:
        ``vocab``: map from token ID to token bytes.
        ``merges``: byte pairs in the order they were learned.

    ``desired_num_chunks`` sets the number of corpus chunks and defaults to 32.
    ``num_workers`` sets the number of pretok processes and defaults to at most 32.
    The two are independent. Merging still runs single-threaded in the main process.

    Windows uses spawn, so callers must invoke this function under
    ``if __name__ == "__main__":``.
    """
    # Stage 1: initialize the vocabulary and turn the corpus into a pre-token frequency table.
    # 1. Initialize the base vocabulary and special tokens
    vocab = initialize_vocab(
        vocab_size = vocab_size,
        special_tokens = special_tokens,
    )

    # 2. Stream and pre-tokenize so the whole corpus is not decoded into one huge str.
    # <|endoftext|> is a safe chunk boundary only when it is also a special token:
    # chunks then do not cut ordinary pre-tokens, and the boundary itself does not
    # enter later pair counts.
    split_special_token = "<|endoftext|>"
    desired_num_chunks = kwargs.get("desired_num_chunks", 32)
    num_workers = kwargs.get(
        "num_workers",
        min(32, os.cpu_count() or 1),
    )

    if not isinstance(desired_num_chunks, int) or desired_num_chunks < 1:
        raise ValueError("desired_num_chunks must be a positive integer")

    if not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")

    if split_special_token not in special_tokens:
        # Without a safe document boundary, do not split on arbitrary byte offsets.
        # That can change GPT-2 pre-tokenization.
        desired_num_chunks = 1

    # The main process only finds boundaries. It does not read or decode whole chunks.
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file=file,
            desired_num_chunks=desired_num_chunks,
            split_special_token=split_special_token.encode("utf-8"),
        )

    worker_input_path = os.path.abspath(os.fspath(input_path))
    worker_special_tokens = tuple(special_tokens)
    chunk_tasks: list[ChunkTask] = [
        (
            worker_input_path,
            start,
            end,
            worker_special_tokens,
        )
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    pretoken_counts: Counter[tuple[int, ...]] = Counter()
    effective_num_workers = min(num_workers, len(chunk_tasks))

    if effective_num_workers <= 1:
        # Do not create a process pool for one chunk or an explicit num_workers=1.
        # That keeps small-corpus tests simple.
        for task in chunk_tasks:
            pretoken_counts.update(_pretokenize_chunk(task))
    else:
        # Only read + decode + pretokenize run in parallel in workers.
        # The main process adds each Counter as soon as it arrives. Merging stays out of the pool.
        with Pool(processes=effective_num_workers) as pool:
            for chunk_pretoken_counts in pool.imap_unordered(
                _pretokenize_chunk,
                chunk_tasks,
                chunksize=1,
            ):
                pretoken_counts.update(chunk_pretoken_counts)
                del chunk_pretoken_counts

    print(
        f"pretok_done unique_pretokens={len(pretoken_counts)}",
        flush=True,
    )
    
    # Stage 2: repeatedly select the globally best pair and add it to the vocabulary.

    # 4. Save byte pairs in training order
    merges: list[tuple[bytes,bytes]] = []

    pair_counts, pair_to_pretokens = build_pair_index(
        pretoken_counts
    )
    pair_heap = build_pair_heap(
        pair_counts=pair_counts,
        vocab=vocab,
    )

    merge_started_at = time.perf_counter()
    remaining_merges = vocab_size - len(vocab)
    print(
        f"merge_start current_vocab={len(vocab)} "
        f"target_vocab={vocab_size} remaining_merges={remaining_merges}",
        flush=True,
    )

    # 6. Keep merging pairs until the vocabulary reaches the target size
    while len(vocab) < vocab_size and pair_counts:
        best_pair = select_best_pair_from_heap(
            pair_heap=pair_heap,
            pair_counts=pair_counts,
        )

        if best_pair is None:
            raise RuntimeError(
                "pair heap is empty while pair_counts is not empty"
            )

        # best_pair is two old token IDs
        left_token_bytes = vocab[best_pair[0]]
        right_token_bytes = vocab[best_pair[1]]

        # Vocabulary IDs are contiguous, so len(vocab) is the next ID
        new_token_id = max(vocab) + 1

        # Create the new merged token
        vocab[new_token_id] = (
            left_token_bytes
            + right_token_bytes
        )

        # merges stores the byte pair, not the token-ID pair
        merges.append(
            (left_token_bytes, right_token_bytes)
        )

        changed_pairs = merge_incrementally(
            pretoken_counts=pretoken_counts,
            pair_counts=pair_counts,
            pair_to_pretokens=pair_to_pretokens,
            pair_to_merge=best_pair,
            new_token_id=new_token_id,
        )

        push_pair_heap_updates(
            pair_heap=pair_heap,
            changed_pairs=changed_pairs,
            pair_counts=pair_counts,
            vocab=vocab,
        )

        # Lazy updates keep stale entries. When the heap grows past 4 times the
        # number of live pairs, rebuild it so old low-frequency entries do not
        # occupy memory for long.
        if (
            pair_counts
            and len(pair_heap)
            > max(4 * len(pair_counts), 100_000)
        ):
            pair_heap = build_pair_heap(
                pair_counts=pair_counts,
                vocab=vocab,
            )

        merge_step = len(merges)
        if merge_step == 1 or merge_step % 500 == 0:
            elapsed = time.perf_counter() - merge_started_at
            rate = merge_step / elapsed if elapsed > 0 else 0.0
            remaining = vocab_size - len(vocab)
            eta_s = remaining / rate if rate > 0 else -1.0
            print(
                f"merge_step={merge_step} vocab={len(vocab)}/{vocab_size} "
                f"elapsed_s={elapsed:.1f} merges_per_s={rate:.1f} eta_s={eta_s:.0f}",
                flush=True,
            )

    merge_elapsed = time.perf_counter() - merge_started_at
    print(
        f"merge_done merges={len(merges)} vocab={len(vocab)} "
        f"elapsed_s={merge_elapsed:.1f}",
        flush=True,
    )

    return vocab, merges
