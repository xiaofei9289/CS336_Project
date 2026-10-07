"""Filter Common Crawl WET records into language-model training text.

Each WET file is handled by one worker. Inside a file, a document is dropped
by the first check it fails. Kept documents then lose lines that occur more
than once in that file, and remaining emails, phone numbers, and IPv4
addresses are masked.
"""

import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.exact_line_deduplication import _line_key
from cs336_data.gopher_quality import gopher_quality_filter
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import is_english
from cs336_data.mask_personal_info import mask_emails, mask_ips, mask_phone_numbers

NSFW_THRESHOLD = 0.6
TOXIC_THRESHOLD = 0.9


def iter_conversion_records(path: Path):
    with open(path, "rb") as handle:
        for record in ArchiveIterator(handle, record_types=WarcRecordType.conversion):
            text = record.reader.read().decode("utf-8", errors="replace")
            if not text.strip():
                continue
            yield record.headers.get("WARC-Target-URI"), text


def mask_personal_info(text: str) -> tuple[str, int]:
    replacements = 0
    for mask in (mask_emails, mask_phone_numbers, mask_ips):
        text, count = mask(text)
        replacements += count
    return text, replacements


def deduplicate_lines(documents: list[dict]) -> tuple[list[dict], list[dict], int]:
    counts: dict[bytes, int] = {}
    split_lines: list[list[str]] = []
    for document in documents:
        lines = document["text"].splitlines()
        split_lines.append(lines)
        for line in lines:
            key = _line_key(line)
            counts[key] = counts.get(key, 0) + 1

    kept = []
    emptied = []
    removed_lines = 0
    for document, lines in zip(documents, split_lines):
        unique_lines = [line for line in lines if counts[_line_key(line)] == 1]
        removed_lines += len(lines) - len(unique_lines)
        text = "\n".join(unique_lines).strip()
        if not text:
            emptied.append(
                {
                    "url": document["url"],
                    "reason": "empty_after_line_dedup",
                    "text": " ".join(document["text"].split())[:240],
                }
            )
            continue
        document["text"] = text
        document["lines_removed"] = len(lines) - len(unique_lines)
        kept.append(document)
    return kept, emptied, removed_lines


def filter_wet_file(path: str) -> dict:
    dropped = Counter()
    discards = []
    candidates = []
    seen = 0
    for url, text in iter_conversion_records(Path(path)):
        seen += 1
        preview = " ".join(text.split())[:240]
        if not is_english(text):
            dropped["not_english"] += 1
            discards.append({"url": url, "reason": "not_english", "text": preview})
            continue
        nsfw_label, nsfw_score = classify_nsfw(text)
        if nsfw_label == "nsfw" and nsfw_score >= NSFW_THRESHOLD:
            dropped["nsfw"] += 1
            discards.append({"url": url, "reason": "nsfw", "score": nsfw_score, "text": preview})
            continue
        toxic_label, toxic_score = classify_toxic_speech(text)
        if toxic_label == "toxic" and toxic_score >= TOXIC_THRESHOLD:
            dropped["toxic"] += 1
            discards.append({"url": url, "reason": "toxic", "score": toxic_score, "text": preview})
            continue
        if not gopher_quality_filter(text):
            dropped["gopher"] += 1
            discards.append({"url": url, "reason": "gopher", "text": preview})
            continue
        candidates.append({"url": url, "text": text})

    return {
        "path": path,
        "seen": seen,
        "candidates": candidates,
        "dropped": dict(dropped),
        "discards": discards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wet_paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    workers = args.workers
    if workers <= 0:
        workers = len(os.sched_getaffinity(0))
    paths = [str(path) for path in args.wet_paths]
    if workers == 1 or len(paths) == 1:
        results = [filter_wet_file(path) for path in paths]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as executor:
            futures = [executor.submit(filter_wet_file, path) for path in paths]
            for future in as_completed(futures):
                results.append(future.result())

    candidates = []
    discards = []
    dropped = Counter()
    seen = 0
    for result in results:
        seen += result["seen"]
        dropped.update(result["dropped"])
        candidates.extend(result["candidates"])
        discards.extend(result["discards"])

    kept_documents, emptied, removed_lines = deduplicate_lines(candidates)
    dropped["empty_after_line_dedup"] += len(emptied)
    discards.extend(emptied)
    pii_replacements = 0
    docs_with_pii = 0
    kept_lines = []
    for document in kept_documents:
        masked, count = mask_personal_info(document["text"])
        pii_replacements += count
        docs_with_pii += count > 0
        kept_lines.append(" ".join(masked.split()))
        discards.append(
            {
                "url": document["url"],
                "reason": "kept",
                "lines_removed": document["lines_removed"],
                "pii_replacements": count,
                "text": " ".join(masked.split())[:240],
            }
        )

    text_path = args.output_dir / "filtered.txt"
    text_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    summary = {
        "inputs": paths,
        "seen": seen,
        "kept": len(kept_lines),
        "dropped": dict(dropped),
        "removed_duplicate_lines": removed_lines,
        "pii_replacements": pii_replacements,
        "docs_with_pii": docs_with_pii,
        "nsfw_threshold": NSFW_THRESHOLD,
        "toxic_threshold": TOXIC_THRESHOLD,
        "output": str(text_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    generator = random.Random(0)
    kept_examples = [item for item in discards if item["reason"] == "kept"]
    discarded_examples = [item for item in discards if item["reason"] != "kept"]
    sample = {
        "kept": generator.sample(kept_examples, min(5, len(kept_examples))),
        "discarded": generator.sample(discarded_examples, min(5, len(discarded_examples))),
    }
    (args.output_dir / "sample.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
