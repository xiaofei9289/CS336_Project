import os
from collections import Counter
from .pretokenization_example import find_chunk_boundaries
from .bpe import (
    initialize_vocab,
    build_pair_index,
    merge_incrementally,
    pretokenize,
    select_best_pair,
)

# Steps for training a byte-level BPE tokenizer

# Step 1: initialize the vocabulary

# Step 2: merge a pair

# Step 3: add the new token

# Step 4: repeat steps 2 and 3 until the vocabulary reaches the target size

# Step 5: save the vocabulary

"""
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # Train a byte-level BPE tokenizer on the given corpus.

    # Returns:
    #    ``vocab``: map from token ID to token bytes.
    #    ``merges``: byte pairs in the order they were learned.

   # ``**kwargs`` is reserved for the adapter or later performance options.
   # The basic implementation does not use it.
    
    # Stage 1: initialize the vocabulary and turn the corpus into a pre-token frequency table.
    # 1. Initialize the base vocabulary and special tokens
    vocab = initialize_vocab(
        vocab_size = vocab_size,
        special_tokens = special_tokens,
    )
    
    # 2. Read the training corpus from disk
    with open(input_path, "r", encoding="utf-8") as file:
        input_text = file.read()
    
    # Stage 2: repeatedly select the globally best pair and add it to the vocabulary.

    # 4. Save byte pairs in training order
    merges: list[tuple[bytes,bytes]] = []


    pretoken_counts = pretokenize(
        text = input_text,
        special_tokens = special_tokens,
    )

    pair_counts, pair_to_pretokens = build_pair_index(
        pretoken_counts
    )

    # 6. Keep merging pairs until the vocabulary reaches the target size
    while len(vocab) < vocab_size and pair_counts:
        best_pair = select_best_pair(
            pair_counts,
            vocab,
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

        merge_incrementally(
            pretoken_counts=pretoken_counts,
            pair_counts=pair_counts,
            pair_to_pretokens=pair_to_pretokens,
            pair_to_merge=best_pair,
            new_token_id=new_token_id,
        )

    return vocab, merges

"""





# Steps for training a byte-level BPE tokenizer

# Step 1: initialize the vocabulary

# Step 2: merge a pair

# Step 3: add the new token

# Step 4: repeat steps 2 and 3 until the vocabulary reaches the target size

# Step 5: save the vocabulary


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

    ``desired_num_chunks`` may be passed through ``**kwargs``. It controls
    how many chunks the corpus is split into. Chunks are still processed
    sequentially; multiprocessing is not used.
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
    desired_num_chunks = kwargs.get("desired_num_chunks", 4)

    if not isinstance(desired_num_chunks, int) or desired_num_chunks < 1:
        raise ValueError("desired_num_chunks must be a positive integer")

    if split_special_token not in special_tokens:
        # Without a safe document boundary, do not split on arbitrary byte offsets.
        # That can change GPT-2 pre-tokenization.
        desired_num_chunks = 1

    pretoken_counts: Counter[tuple[int, ...]] = Counter()

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file=file,
            desired_num_chunks=desired_num_chunks,
            split_special_token=split_special_token.encode("utf-8"),
        )

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk_text = file.read(end - start).decode("utf-8")

            chunk_pretoken_counts = pretokenize(
                text=chunk_text,
                special_tokens=special_tokens,
            )
            pretoken_counts.update(chunk_pretoken_counts)

            # This chunk is already merged into the total Counter, so it can be released.
            del chunk_text, chunk_pretoken_counts
    
    # Stage 2: repeatedly select the globally best pair and add it to the vocabulary.

    # 4. Save byte pairs in training order
    merges: list[tuple[bytes,bytes]] = []

    pair_counts, pair_to_pretokens = build_pair_index(
        pretoken_counts
    )

    # 6. Keep merging pairs until the vocabulary reaches the target size
    while len(vocab) < vocab_size and pair_counts:
        best_pair = select_best_pair(
            pair_counts,
            vocab,
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

        merge_incrementally(
            pretoken_counts=pretoken_counts,
            pair_counts=pair_counts,
            pair_to_pretokens=pair_to_pretokens,
            pair_to_merge=best_pair,
            new_token_id=new_token_id,
        )

    return vocab, merges



