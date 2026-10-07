"""Encode filtered documents with the GPT-2 tokenizer.

Each input line is one document. The GPT-2 end-of-sequence id is appended
after every document, and the ids are stored as uint16.
"""

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    documents = args.input_path.read_text(encoding="utf-8").splitlines()
    ids: list[int] = []
    for document in documents:
        ids.extend(tokenizer.encode(document))
        ids.append(tokenizer.eos_token_id)
    array = np.asarray(ids, dtype=np.uint16)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(args.output_path)
    print(f"documents {len(documents)}")
    print(f"tokens {len(array)}")
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
