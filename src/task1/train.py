"""
Train MathBERTa text encoder on ARQMath Task 1 (Answer Retrieval).

Uses GradCache for large effective-batch contrastive learning:
the DataLoader yields full batches (e.g. 256 pairs) but forward passes
run in micro-batches (e.g. 32) that fit in GPU memory.  InfoNCE sees
all 256 pairs → 255 in-batch negatives, giving a much stronger
contrastive signal than naive gradient accumulation (which only sees
micro-batch-sized negatives per forward pass).

Usage:
  python -m src.task1.train --config configs/task1.yaml
  python -m src.task1.train --batch-size 256 --mini-batch-size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.task1.dataset import Task1Dataset
from src.task1.model import DEFAULT_MODEL, TextEncoder


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def info_nce_loss(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE (NT-Xent) with in-batch negatives."""
    sim = (q_emb @ p_emb.T) / temperature
    labels = torch.arange(q_emb.size(0), device=q_emb.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


# ---------------------------------------------------------------------------
# GradCache helpers
# ---------------------------------------------------------------------------

def _collate_strings(features):
    """Pass-through collator: keeps query/positive as string lists."""
    return {
        "query": [f["query"] for f in features],
        "positive": [f["positive"] for f in features],
    }


def _encode_subbatches(model, enc, mini_bs, device, amp_dtype):
    """Forward all micro-batches, returning a list of embedding tensors."""
    use_amp = amp_dtype is not None
    B = enc["input_ids"].size(0)
    embs = []
    for s in range(0, B, mini_bs):
        e = min(s + mini_bs, B)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            emb = model(
                enc["input_ids"][s:e].to(device),
                enc["attention_mask"][s:e].to(device),
            )
        embs.append(emb.float())
    return embs


def _grad_cache_step(
    model, queries, positives, tokenizer,
    max_length, mini_bs, temperature, device, amp_dtype,
):
    """
    One GradCache training step.

    Phase 1: encode all micro-batches without grad, cache float32 embeddings.
    Phase 2: compute InfoNCE on full embedding matrix, get embedding gradients.
    Phase 3: re-forward each micro-batch WITH grad, propagate cached gradients
             to the model via a surrogate dot-product loss.
    """
    B = len(queries)
    q_enc = tokenizer(queries, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    p_enc = tokenizer(positives, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    use_amp = amp_dtype is not None

    # Phase 1 — collect embeddings (no computation graph)
    with torch.no_grad():
        q_embs = _encode_subbatches(model, q_enc, mini_bs, device, amp_dtype)
        p_embs = _encode_subbatches(model, p_enc, mini_bs, device, amp_dtype)

    # Phase 2 — full-batch InfoNCE on cached embeddings
    for emb in q_embs + p_embs:
        emb.requires_grad_(True)
    loss = info_nce_loss(torch.cat(q_embs), torch.cat(p_embs), temperature)
    loss.backward()

    # Phase 3 — propagate embedding gradients to model parameters
    for i, s in enumerate(range(0, B, mini_bs)):
        e = min(s + mini_bs, B)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            q_rep = model(q_enc["input_ids"][s:e].to(device), q_enc["attention_mask"][s:e].to(device))
            p_rep = model(p_enc["input_ids"][s:e].to(device), p_enc["attention_mask"][s:e].to(device))
        surr = (
            torch.dot(q_rep.flatten().float(), q_embs[i].grad.flatten())
            + torch.dot(p_rep.flatten().float(), p_embs[i].grad.flatten())
        )
        surr.backward()

    return loss.item()


def _validate(model, loader, tokenizer, max_length, mini_bs, temperature, device, amp_dtype):
    """Compute full-batch InfoNCE on the validation set."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            q_enc = tokenizer(batch["query"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            p_enc = tokenizer(batch["positive"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            q_embs = _encode_subbatches(model, q_enc, mini_bs, device, amp_dtype)
            p_embs = _encode_subbatches(model, p_enc, mini_bs, device, amp_dtype)
            total += info_nce_loss(torch.cat(q_embs), torch.cat(p_embs), temperature).item()
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg.get("training", {})
    mcfg = cfg.get("model", {})

    model_name = mcfg.get("model_name", DEFAULT_MODEL)
    max_length = mcfg.get("max_length", 256)
    batch_size = tcfg.get("batch_size", 256)
    mini_batch_size = tcfg.get("mini_batch_size", 32)
    epochs = tcfg.get("epochs", 20)
    lr = tcfg.get("lr", 5e-5)
    warmup_steps = tcfg.get("warmup_steps", 100)
    temperature = tcfg.get("temperature", 0.07)
    patience = tcfg.get("early_stopping_patience", 5)
    ckpt_dir = Path(tcfg.get("checkpoint_dir", "checkpoints/task1"))
    include_task2 = tcfg.get("include_task2", False)

    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(
        f"Device: {device}  |  batch={batch_size}  mini_batch={mini_batch_size}  "
        f"negatives={batch_size - 1}  AMP={amp_dtype}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TextEncoder(
        model_name=model_name,
        max_length=max_length,
        normalize=True,
        gradient_checkpointing=True,
    ).to(device)

    full_ds = Task1Dataset(split="train", include_task2=include_task2)
    n_val = max(1, len(full_ds) // 10)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [len(full_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=_collate_strings,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=_collate_strings,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            step_loss = _grad_cache_step(
                model,
                batch["query"],
                batch["positive"],
                tokenizer,
                max_length,
                mini_batch_size,
                temperature,
                device,
                amp_dtype,
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += step_loss
            pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train = epoch_loss / len(train_loader)
        avg_val = _validate(
            model, val_loader, tokenizer, max_length,
            mini_batch_size, temperature, device, amp_dtype,
        )
        print(
            f"Epoch {epoch:3d}  |  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}",
            flush=True,
        )

        if avg_val < best_val:
            best_val = avg_val
            stale = 0
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  → saved best checkpoint (val_loss={avg_val:.4f})", flush=True)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping ({patience} epochs without improvement)", flush=True)
                break

    print(f"Training complete.  Best val_loss: {best_val:.4f}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=_PROJECT_ROOT / "configs/task1.yaml")
    parser.add_argument("--batch-size", type=int, default=None, help="Effective batch (# negatives)")
    parser.add_argument("--mini-batch-size", type=int, default=None, help="Micro-batch (GPU memory)")
    args = parser.parse_args()

    cfg: dict = {}
    if args.config.exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.mini_batch_size is not None:
        cfg.setdefault("training", {})["mini_batch_size"] = args.mini_batch_size
    train(cfg)


if __name__ == "__main__":
    main()
