import regex as regex
from collections.abc import Iterable, Iterator

from .bpe import GPT2_PRETOKEN_PATTERN

class Tokenizer:

    # 1. Initialize the constructor
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes,bytes]],
        special_tokens: list[str] | None = None,
        ):
        # 1. Copy mutable inputs so changes outside the class do not affect the Tokenizer.
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or []) 

        # 2. Build the inverse vocabulary: token bytes -> token id
        self.inverse_vocab = self._build_inverse_vocab()

        # 3. Build merge ranks
        self.merge_ranks = self._build_merge_ranks()
        
        # 4. Store the token id of each special token
        self.special_token_ids: dict[str, int] = {}

        for special_token in self.special_tokens:
            # Convert the special-token string to UTF-8 bytes
            token_bytes = special_token.encode("utf-8")

            # Check that the special token exists in the vocab
            if token_bytes not in self.inverse_vocab:
                raise ValueError(
                    f"special token {special_token!r} is not in vocab"
                )

            # Store the special-token mapping
            self.special_token_ids[special_token] = self.inverse_vocab[token_bytes]
        
        # Compile once so encode does not rebuild the regex on every call.
        unique_special_tokens = sorted(
            set(self.special_tokens),
            key=len,
            reverse=True,
        )
        if "" in unique_special_tokens:
            raise ValueError("special_tokens cannot contain an empty string")

        self._special_token_pattern = (
            regex.compile(
                "|".join(
                    regex.escape(token) for token in unique_special_tokens
                )
            )
            if unique_special_tokens
            else None
        ) 




    # ---------- Public API ----------
    # 1. encode
    def encode(self, text: str) -> list[int]:
        """
        Encode arbitrary raw text into a list of token IDs.

        Steps:
        1. Split the raw text while keeping special tokens;
        2. Map each special token directly to a single ID;
        3. Pre-tokenize ordinary text with the GPT-2 pattern;
        4. Turn each pre-token into initial byte IDs;
        5. Apply BPE merges inside each pre-token;
        6. Concatenate all results.
        """

        final_token_ids: list[int] = []

        # 1. Separate ordinary text from special tokens
        segments = self._split_on_special_tokens(text)

        for segment, is_special in segments:

            # 2. A special token emits a single ID
            if is_special:
                final_token_ids.append(self.special_token_ids[segment])
            else:
                final_token_ids.extend(self._encode_ordinary_text(segment))

        return final_token_ids

    # 2. encode_iterable, encoding one chunk at a time
    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """Encode a string iterable chunk by chunk and yield token IDs."""
        for chunk in iterable:
            if chunk:
                yield from self.encode(chunk)

    # 3. decode
    def decode(self, tokens: list[int]) -> str:
        # 1.1 Create a byte buffer
        decode_bytes = bytearray()
        # 1.2 Walk every token id
        for token_id in tokens:
            # 1.21 Check that the token id exists
            if token_id not in self.vocab:
                raise ValueError(f"Unkown token id: {token_id}")
            # 1.22 Look up and append the token's bytes
            decode_bytes.extend(self.vocab[token_id])
        # 1.3 After the loop, turn the byte buffer into a string
        decoded_str = decode_bytes.decode("utf-8", errors="replace")
        return decoded_str

# ---------- Private initialization helpers ----------

    def _build_inverse_vocab(self) -> dict[bytes, int]:
        """Build the inverse vocabulary, ``token bytes -> token ID``."""
        inverse_vocab: dict[bytes, int] = {}

        for token_id, token_bytes in self.vocab.items():
            if token_bytes in inverse_vocab:
                existing_id = inverse_vocab[token_bytes]
                raise ValueError(
                    f"token bytes {token_bytes!r} already use ID {existing_id}"
                )
            inverse_vocab[token_bytes] = token_id

        return inverse_vocab

    def _build_merge_ranks(self) -> dict[tuple[bytes, bytes], int]:
        """Turn the merge list into a map from pair to rank."""
        merge_ranks: dict[tuple[bytes, bytes], int] = {}

        for rank, pair in enumerate(self.merges):
            if pair in merge_ranks:
                raise ValueError(
                    f"merge pair {pair!r} already has rank {merge_ranks[pair]}"
                )
            merge_ranks[pair] = rank

        return merge_ranks

# ---------- Private encoding helpers ----------
# 1. Split while keeping special tokens
    def _split_on_special_tokens(
        self, 
        text: str,
        ) -> list[tuple[str, bool]]:
        """
        Given a piece of raw text:
        1. Find the special tokens in it
        2. Separate ordinary text from special tokens
        3. Keep special tokens unchanged
        4. Return a list of typed segments
        """
        # 1. An empty string yields an empty list
        if not text:
            return []
        # 2. With no special tokens, return the whole string as ordinary text
        if not self.special_tokens:
            return [(text, False)]
        # 3. Empty result list that will hold the final segments
        segments: list[tuple[str, bool]] = []
        # 4. Pointer used to walk the text
        pointer = 0
        # 5. Walk the text and find every special token
        for match in self._special_token_pattern.finditer(text):
            # 5.1 A gap before the match is ordinary text
            if match.start() > pointer:
                normal_text = text[pointer:match.start()]
                segments.append((normal_text, False))
            # 5.2 A match at the pointer means the text starts with this special token
            special_token =match.group(0)
            segments.append((special_token, True))
            # 5.3 Advance the pointer
            pointer = match.end()
        # 6. Ordinary text after the last special token
        if pointer < len(text):
            normal_text = text[pointer:]
            segments.append((normal_text, False))
        return segments

    # 2. Encode ordinary text
    def _encode_ordinary_text(self, text: str) -> list[int]:
        """Pre-tokenize and apply BPE to text that contains no special tokens."""
        pre_tokens = [
            match.group(0)
            for match in GPT2_PRETOKEN_PATTERN.finditer(text)
        ]
        if "".join(pre_tokens) != text:
            raise ValueError(
                "GPT-2 pre-tokenization did not preserve the input text: "
                f"{text!r}"
            )

        token_ids: list[int] = []
        for pre_token in pre_tokens:
            initial_ids = self._encode_initial_bytes(pre_token)
            token_ids.extend(self._apply_bpe_merges(initial_ids))

        return token_ids

    # 3. Encode the initial single-byte token IDs
    def _encode_initial_bytes(self, pre_token: str) -> list[int]:
        """Turn one pre-token into its initial single-byte token IDs."""
        token_ids: list[int] = []

        for byte_value in pre_token.encode("utf-8"):
            byte_token = bytes([byte_value])
            if byte_token not in self.inverse_vocab:
                raise ValueError(
                    "single-byte token is missing from vocabulary: "
                    f"{byte_token!r} (byte value: {byte_value})"
                )
            token_ids.append(self.inverse_vocab[byte_token])

        return token_ids
    
    # 4. Apply BPE merges
    def _apply_bpe_merges(self, token_ids: list[int]) -> list[int]:
        """Apply every available merge to one pre-token, in training order."""
        current_ids = list(token_ids)

        for token_id in current_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Unknown token ID: {token_id}")

        while len(current_ids) >= 2:
            mergeable_pairs = []
            for left_id, right_id in zip(current_ids, current_ids[1:]):
                pair = (self.vocab[left_id], self.vocab[right_id])
                if pair in self.merge_ranks:
                    mergeable_pairs.append(pair)

            if not mergeable_pairs:
                break

            best_pair = min(
                mergeable_pairs,
                key=self.merge_ranks.__getitem__,
            )
            merged_bytes = best_pair[0] + best_pair[1]
            if merged_bytes not in self.inverse_vocab:
                raise ValueError(
                    "merged token is missing from vocabulary: "
                    f"{best_pair[0]!r} + {best_pair[1]!r} = {merged_bytes!r}"
                )

            current_ids = self._merge_pair_once(
                token_ids=current_ids,
                target_pair=best_pair,
                merged_id=self.inverse_vocab[merged_bytes],
            )

        return current_ids

    # 5. Apply one BPE merge
    def _merge_pair_once(
        self,
        token_ids: list[int],
        target_pair: tuple[bytes, bytes],
        merged_id: int,
    ) -> list[int]:
        """Merge every non-overlapping occurrence of the target pair, left to right."""
        merged_ids: list[int] = []
        index = 0

        while index < len(token_ids):
            if index + 1 < len(token_ids):
                current_pair = (
                    self.vocab[token_ids[index]],
                    self.vocab[token_ids[index + 1]],
                )
                if current_pair == target_pair:
                    merged_ids.append(merged_id)
                    index += 2
                    continue

            merged_ids.append(token_ids[index])
            index += 1

        return merged_ids
