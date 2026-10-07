"""Fuzzy document deduplication with MinHash and locality-sensitive hashing."""

import hashlib
import random
import re
import string
import unicodedata
from collections import defaultdict
from functools import cache
from pathlib import Path

_PRIME = (1 << 61) - 1
_PUNCTUATION = str.maketrans({character: " " for character in string.punctuation})


def normalize_text(text: str) -> str:
    """Lowercase, strip accents and punctuation, and collapse whitespace."""
    decomposed = unicodedata.normalize("NFD", text)
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    lowered = without_accents.lower().translate(_PUNCTUATION)
    return re.sub(r"\s+", " ", lowered).strip()


def word_ngrams(text: str, n: int) -> set[str]:
    words = text.split()
    if len(words) < n:
        return set()
    return {" ".join(words[index : index + n]) for index in range(len(words) - n + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)


@cache
def _hash_coefficients(num_hashes: int) -> tuple[tuple[int, int], ...]:
    generator = random.Random(336)
    return tuple(
        (generator.randrange(1, _PRIME), generator.randrange(0, _PRIME))
        for _ in range(num_hashes)
    )


def _gram_hash(ngram: str) -> int:
    digest = hashlib.blake2s(ngram.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % _PRIME


def minhash_signature(ngrams: set[str], num_hashes: int) -> tuple[int, ...]:
    coefficients = _hash_coefficients(num_hashes)
    if not ngrams:
        return tuple(_PRIME for _ in range(num_hashes))
    signature = [_PRIME] * num_hashes
    for ngram in ngrams:
        hashed = _gram_hash(ngram)
        for index, (slope, intercept) in enumerate(coefficients):
            value = (slope * hashed + intercept) % _PRIME
            if value < signature[index]:
                signature[index] = value
    return tuple(signature)


def minhash_deduplication(
    input_files: list[Path],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: Path,
) -> None:
    """Write one copy of each document, dropping fuzzy duplicates."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for path in input_files]
    originals = [path.read_bytes() for path in paths]
    ngram_sets = [
        word_ngrams(normalize_text(text.decode("utf-8", errors="replace")), ngrams)
        for text in originals
    ]
    signatures = [minhash_signature(ngram_set, num_hashes) for ngram_set in ngram_sets]

    rows_per_band = num_hashes // num_bands
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for doc_id, signature in enumerate(signatures):
        for band in range(num_bands):
            start = band * rows_per_band
            key = (band, signature[start : start + rows_per_band])
            buckets[key].append(doc_id)

    candidate_pairs: set[tuple[int, int]] = set()
    for doc_ids in buckets.values():
        for left in range(len(doc_ids)):
            for right in range(left + 1, len(doc_ids)):
                first, second = doc_ids[left], doc_ids[right]
                if first > second:
                    first, second = second, first
                candidate_pairs.add((first, second))

    parent = list(range(len(paths)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in candidate_pairs:
        if jaccard_similarity(ngram_sets[left], ngram_sets[right]) >= jaccard_threshold:
            union(left, right)

    clusters: dict[int, list[int]] = defaultdict(list)
    for doc_id in range(len(paths)):
        clusters[find(doc_id)].append(doc_id)

    drop: set[int] = set()
    chooser = random.Random(0)
    for members in clusters.values():
        if len(members) == 1:
            continue
        keep = chooser.choice(members)
        drop.update(member for member in members if member != keep)

    for doc_id, path in enumerate(paths):
        if doc_id in drop:
            continue
        (output_directory / path.name).write_bytes(originals[doc_id])
