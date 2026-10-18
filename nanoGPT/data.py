"""Character-level data loading and batching (from gpt-dev.ipynb)."""

from __future__ import annotations

from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = DATA_DIR / "input.txt"


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_text(path: Path | str = DEFAULT_DATA_PATH) -> str:
    path = Path(path)
    return path.read_text(encoding="utf-8")


def prepare_tensors(
    text: str,
    train_frac: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    return data[:n], data[n:], tokenizer


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
