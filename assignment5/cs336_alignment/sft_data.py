"""Packed instruction-tuning dataset. Reads plain JSONL and .jsonl.gz."""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PackedSFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length: int, shuffle: bool):
        template_path = Path(__file__).resolve().parent / "prompts_safety" / "alpaca_sft.prompt"
        template = template_path.read_text().rstrip("\n")
        documents = []
        path = Path(dataset_path)
        handle_cm = gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")
        with handle_cm as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = template.format(instruction=row["prompt"], response=row["response"])
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                document = []
                if tokenizer.bos_token_id is not None:
                    document.append(tokenizer.bos_token_id)
                document.extend(token_ids)
                if tokenizer.eos_token_id is not None:
                    document.append(tokenizer.eos_token_id)
                documents.append(document)
        if shuffle:
            random.shuffle(documents)
        stream: list[int] = []
        for document in documents:
            stream.extend(document)
        self.seq_length = seq_length
        self.stream = stream
        self.n_sequences = max(0, (len(stream) - 1) // seq_length)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.n_sequences:
            raise IndexError(index)
        start = index * self.seq_length
        input_ids = self.stream[start : start + self.seq_length]
        labels = self.stream[start + 1 : start + 1 + self.seq_length]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
