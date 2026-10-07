"""
Low-level helpers for byte-level BPE training.

This module only performs stateless primitive operations:
1. Initialize the vocabulary;
2. GPT-2 pre-tokenization;
3. Count and select adjacent token pairs;
4. Apply the selected pair to pre-tokens.

The full training loop lives in ``train_bpe.py``. Encoding and decoding
live in ``Tokenizer`` in ``tokenizer.py``.
This module does not own the training loop.
"""

from collections import Counter, defaultdict
import regex as regex

# Training and encoding must use the same GPT-2 pre-tokenization rules.
GPT2_PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

TokenIds = tuple[int, ...]
TokenPair = tuple[int, int]
PretokenCounts = dict[TokenIds, int]

# Function 01: initialize the base vocabulary
def initialize_vocab(
    vocab_size: int,
    special_tokens: list[str],
) -> dict[int, bytes]:
    """
    Create the initial vocabulary for byte-level BPE.

    The initial vocabulary contains:
    1. 256 base byte tokens;
    2. Deduplicated special tokens.

    vocab_size is the final target vocabulary size.
    This function does not create BPE merge tokens early; it only checks
    that the target size is large enough.
"""
    # 1. Create the 256 base byte tokens
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}

    # 2. Add special tokens, and keep their IDs unique
    # 2.1 Drop duplicate special tokens while preserving the original order
    unique_special_tokens = list(dict.fromkeys(special_tokens))
    
    # 3. The initial vocabulary size is 256 plus the number of special tokens
    # 3.1 Add the base vocabulary length and the special-token count
    initial_vocab_size = len(vocab) + len(unique_special_tokens)
    # 3.2 The requested vocab_size must be at least the initial vocabulary size
    if vocab_size < initial_vocab_size:
        raise ValueError(
            f"Vocab size must be greater than or equal to {initial_vocab_size}"
            )
    # 4. Merge the base vocabulary and the special tokens
    # 4.1 Append each special token's bytes to the initial vocabulary
    for special_token in unique_special_tokens:
        special_token_id = len(vocab)
        vocab[special_token_id] = special_token.encode('utf-8')

    return vocab

# Function 02: split text on special tokens

def split_on_special_tokens(
    text: str,
    special_tokens: list[str],
) -> list[str]:

    """ 
    Split training text on special tokens and drop those tokens from the result.

    This is training-time behavior: special tokens are only document boundaries
    and must not enter ordinary BPE pair counts. Encoding must keep special
    tokens, so ``Tokenizer._split_on_special_tokens`` handles that separately.
        
    """

    # 1. Deduplicate the special-token list while preserving the original order
    unique_special_tokens = list(dict.fromkeys(special_tokens))

    # 2. If there are no special tokens, return the text unsplit
    if not unique_special_tokens:
        return [text]

    # 3. Reject an empty string as a special token
    if any(token == "" for token in unique_special_tokens):
        raise ValueError("Special tokens list cannot contain empty strings")

    # 4. Split the text on special tokens
    # 4.1 Put longer special tokens first so a prefix does not hide a longer match
    unique_special_tokens.sort(key=len, reverse=True)
    # 4.2 Escape special tokens so the regex treats their symbols as literal characters
    # 4.2.1 Build the escaped-token list
    escaped_tokens = [regex.escape(token) for token in unique_special_tokens]
    # 4.2.2 Join the escaped tokens with |
    special_tokens_pattern = regex.compile("|".join(escaped_tokens))
    # 4.4 Split the text with the regex and return the pieces
    return special_tokens_pattern.split(text)

# Function 03: pre-tokenize text fragments
def pretokenize(
    text: str,
    special_tokens: list[str],
) -> PretokenCounts:
    """
    Turn training text into a frequency table of ``byte-ID tuple -> count``.
    """

    # 0. Create the counter
    # key: the byte-integer sequence of one pre-token
    # value: how many times that sequence occurs
    pretoken_counts: Counter[TokenIds] = Counter()


    # 1. Split the text on special tokens first
    text_fragments = split_on_special_tokens(text, special_tokens)
    
    # 2. Handle each ordinary text fragment separately
    for text_fragment in text_fragments:
        # 2.1 Skip empty strings
        if text_fragment == "":
            continue
        # 2.2 Store every regex match so we can check that nothing was dropped
        matched_text_parts = []
        
        # 3. Pre-tokenize with the GPT-2 regex
        for match in GPT2_PRETOKEN_PATTERN.finditer(text_fragment):
            pretoken_text = match.group(0)
            matched_text_parts.append(pretoken_text)
            
            # 4. Encode the matched text as UTF-8 bytes
            pretoken_bytes = pretoken_text.encode('utf-8')
            # 5. Turn each byte into an integer and store them in an immutable tuple
            pretoken_token_ids =tuple(pretoken_bytes)
            # 6. Count how often this byte-integer sequence occurs
            pretoken_counts[pretoken_token_ids] += 1

        # 7. Confirm the regex did not drop any characters
        assert "".join(matched_text_parts) == text_fragment
        
    return pretoken_counts


# Function 04: count adjacent pairs globally
def count_adjacent_pairs(
    pretoken_counts: dict[tuple[int, ...], int],
) -> dict[TokenPair, int]:
    """
    Count the global frequency of every adjacent token pair, weighted by pre-token frequency.
    """

    # 0. Create the counter
    # key: the integer pair of one adjacent pair
    # value: how many times that pair occurs
    adjacent_pair_counts: Counter[tuple[int, int]] = Counter()
    
    # 1. Walk every integer sequence in pretoken_counts
    for pretoken_token_ids, pretoken_frequency in pretoken_counts.items():
        # 1.1 Skip sequences shorter than 2
        if len(pretoken_token_ids) < 2:
            continue
        # 1.2 
        for pair in zip(pretoken_token_ids, pretoken_token_ids[1:]):
            adjacent_pair_counts[pair] += pretoken_frequency
    return dict(adjacent_pair_counts)   
#------
def build_pair_index(
    pretoken_counts: dict[tuple[int, ...], int],
):
    """
    Returns:
    1. pair_counts:
       Weighted count of each adjacent pair across the corpus

    2. pair_to_pretokens:
       Which unique pre-tokens contain each pair
    """

    pair_counts = Counter()
    pair_to_pretokens = defaultdict(set)

    for pretoken, frequency in pretoken_counts.items():

        # Count how often each pair occurs inside this pre-token
        local_pair_counts = Counter(
            zip(pretoken, pretoken[1:])
        )

        for pair, occurrences in local_pair_counts.items():
            # Occurrences times this pre-token's corpus frequency
            pair_counts[pair] += occurrences * frequency

            # The inverted index only needs to record this pre-token once
            pair_to_pretokens[pair].add(pretoken)

    return pair_counts, pair_to_pretokens
#--------------------------------




# Function 05: pick the most frequent pair; break ties toward the lexicographically larger byte pair
def select_best_pair(
    pair_counts: dict[tuple[int, int], int],
    vocab: dict[int, bytes],
) -> TokenPair | None:
    """
    Pick the globally most frequent pair. On a tie, pick the lexicographically larger byte pair.
    """

    # 1. Return None when pair_counts is empty
    if not pair_counts:
        return None
    # 2. Find the highest-frequency pair in pair_counts
    best_pair = max(
        pair_counts,
        key = lambda pair:(pair_counts[pair], vocab[pair[0]], vocab[pair[1]])
    )

    return best_pair

# Function 06: merge one non-overlapping run of the pair, left to right, inside a single pre-token
def merge_pairs(
    token_ids: tuple[int, ...],
    target_pair: tuple[int, int],
    new_token_id: int,
) -> TokenIds:
    """
    Merge one non-overlapping run of the pair, left to right, inside a single pre-token
    """

    # 1. Create a new token-id list
    merged_token_ids = []
    # 2. Pointer to the current position
    current_index = 0
    # 3. 
    while current_index < len(token_ids):
        # 3.1 If the current position matches both ids of target_pair, merge them
        if(
            current_index + 1 < len(token_ids) and
            token_ids[current_index] == target_pair[0] and
            token_ids[current_index + 1] == target_pair[1]
        ):
            merged_token_ids.append(new_token_id)
            current_index += 2
        else:
            # 3.2 Otherwise keep the current token id
            merged_token_ids.append(token_ids[current_index])
            current_index += 1
    return tuple(merged_token_ids)


# Function 07: apply the selected pair to every pre-token and build the new frequency table.
def merge_all_pretokens(
    pretoken_counts: PretokenCounts,
    pair_to_merge: TokenPair,
    new_token_id: int,
) -> PretokenCounts:
    """Apply one pair merge to every pre-token and aggregate the new frequency table."""
    new_pretoken_counts: Counter[TokenIds] = Counter()

    for token_ids, frequency in pretoken_counts.items():
        merged_token_ids = merge_pairs(
            token_ids=token_ids,
            target_pair=pair_to_merge,
            new_token_id=new_token_id,
        )
        new_pretoken_counts[merged_token_ids] += frequency

    return dict(new_pretoken_counts)


#--------------------------------

def merge_incrementally(
    pretoken_counts: dict[tuple[int, ...], int],
    pair_counts: Counter,
    pair_to_pretokens: dict[
        tuple[int, int],
        set[tuple[int, ...]],
    ],
    pair_to_merge: tuple[int, int],
    new_token_id: int,
) -> None:
    """
    Merge pair_to_merge into new_token_id.

    Update these structures in place:
    - pretoken_counts
    - pair_counts
    - pair_to_pretokens
    """

    # Copy first, because the inverted index is mutated below
    affected_pretokens = list(
        pair_to_pretokens.get(pair_to_merge, set())
    )

    if not affected_pretokens:
        return

    # Record every change first so we do not mutate the structure while iterating it
    changes = []

    for old_pretoken in affected_pretokens:
        frequency = pretoken_counts[old_pretoken]

        new_pretoken = merge_pairs(
            old_pretoken,
            pair_to_merge,
            new_token_id,
        )

        old_local_counts = Counter(
            zip(old_pretoken, old_pretoken[1:])
        )

        new_local_counts = Counter(
            zip(new_pretoken, new_pretoken[1:])
        )

        changes.append(
            (
                old_pretoken,
                new_pretoken,
                frequency,
                old_local_counts,
                new_local_counts,
            )
        )

    # Stage 1: remove the old pre-token and its pair contributions
    for (
        old_pretoken,
        new_pretoken,
        frequency,
        old_local_counts,
        new_local_counts,
    ) in changes:

        del pretoken_counts[old_pretoken]

        for old_pair, occurrences in old_local_counts.items():
            pair_counts[old_pair] -= occurrences * frequency

            # Drop the old sequence from the inverted index
            pretoken_set = pair_to_pretokens[old_pair]
            pretoken_set.discard(old_pretoken)

            # Drop empty entries
            if not pretoken_set:
                del pair_to_pretokens[old_pair]

            if pair_counts[old_pair] == 0:
                del pair_counts[old_pair]

    # Stage 2: add the new pre-token and its pair contributions
    for (
        old_pretoken,
        new_pretoken,
        frequency,
        old_local_counts,
        new_local_counts,
    ) in changes:

        # Collisions are rare, but adding is safer than assigning
        pretoken_counts[new_pretoken] = (
            pretoken_counts.get(new_pretoken, 0)
            + frequency
        )

        for new_pair, occurrences in new_local_counts.items():
            pair_counts[new_pair] += occurrences * frequency
            pair_to_pretokens[new_pair].add(new_pretoken)


