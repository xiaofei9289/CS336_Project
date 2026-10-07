"""Drop lines that occur more than once across a set of documents."""

import hashlib
from pathlib import Path


def _line_key(line: str) -> bytes:
    return hashlib.sha256(line.encode("utf-8")).digest()


def exact_line_deduplication(
    input_files: list[Path], output_directory: Path
) -> None:
    """Rewrite each file, keeping only lines that appear once in the corpus."""
    output_directory.mkdir(parents=True, exist_ok=True)
    counts: dict[bytes, int] = {}
    documents: list[tuple[Path, list[str]]] = []
    for path in input_files:
        path = Path(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        documents.append((path, lines))
        for line in lines:
            key = _line_key(line)
            counts[key] = counts.get(key, 0) + 1

    for path, lines in documents:
        kept = [line for line in lines if counts[_line_key(line)] == 1]
        text = "" if not kept else "\n".join(kept) + "\n"
        (output_directory / path.name).write_text(text, encoding="utf-8")
