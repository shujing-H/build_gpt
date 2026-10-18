"""
Train the bigram model on Tiny Shakespeare (same loop as gpt-dev.ipynb).

Run from repo root:
  uv run python -m nanoGPT.train_bigram
"""

import argparse

import torch

from nanoGPT.bigram import BigramLanguageModel
from nanoGPT.data import DEFAULT_DATA_PATH, get_batch, load_text, prepare_tensors


def parse_args():
    p = argparse.ArgumentParser(description="Train bigram char LM")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--sample-len", type=int, default=300)
    return p.parse_args()


@torch.no_grad()
def sample(model, tokenizer, device, max_new_tokens: int) -> str:
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0].tolist())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    text = load_text(args.data)
    train_data, val_data, tok = prepare_tensors(text)
    print(f"chars: {tok.vocab_size:,}  train tokens: {len(train_data):,}")

    model = BigramLanguageModel(tok.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("--- before train ---")
    print(sample(model, tok, device, args.sample_len))

    model.train()
    for step in range(args.steps):
        xb, yb = get_batch(
            "train",
            train_data,
            val_data,
            args.batch_size,
            args.block_size,
            device,
        )
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    model.eval()
    print("--- after train ---")
    print(sample(model, tok, device, args.sample_len))


if __name__ == "__main__":
    main()
