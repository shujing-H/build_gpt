"""
Train a small character-level GPT (uses nanoGPT/model.py).

Notebook stops at single-head attention; this wires the full stack for the same dataset.

Run from repo root:
  uv run python -m nanoGPT.train_gpt
"""

import argparse

import torch

from nanoGPT.data import DEFAULT_DATA_PATH, get_batch, load_text, prepare_tensors
from nanoGPT.model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train small char-level GPT")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--sample-len", type=int, default=300)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


@torch.no_grad()
def sample(model: GPT, tokenizer, device, max_new_tokens: int) -> str:
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0].tolist())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    text = load_text(args.data)
    train_data, val_data, tok = prepare_tensors(text)

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=tok.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=False,
    )
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=args.lr,
        betas=(0.9, 0.95),
        device_type=device.type,
    )

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
        if step % 500 == 0 or step == args.steps - 1:
            with torch.no_grad():
                xb_v, yb_v = get_batch(
                    "val",
                    train_data,
                    val_data,
                    args.batch_size,
                    args.block_size,
                    device,
                )
                _, val_loss = model(xb_v, yb_v)
            print(
                f"step {step:5d}  train {loss.item():.4f}  val {val_loss.item():.4f}"
            )

    model.eval()
    print("--- after train ---")
    print(sample(model, tok, device, args.sample_len))


if __name__ == "__main__":
    main()
