"""
Stage 1 training: GNN formula encoder on ARQMath Task 2 (Years 1 & 2).

Loss: InfoNCE (NT-Xent) over in-batch negatives + optional hard negatives.
  For a batch of B (query, positive) pairs:
    - The B positive embeddings are the only "matches" on the diagonal.
    - All other B-1 in-batch positives are treated as negatives.
    - If hard negatives are provided, they are appended to the negative pool.
  Temperature τ is a learnable scalar (log-parameterised for stability).

Usage (auto-detects GPU count, no torchrun required):
  python -m src.stage1.train --config configs/gnn_stage1.yaml

  On 2× A100 this spawns two DDP workers internally via mp.spawn.
  On a single GPU or CPU it runs in a single process with no overhead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.stage1.dataset import FormulaRetrievalDataset, collate_fn
from src.stage1.model.formula_encoder import FormulaEncoder


# ---------------------------------------------------------------------------
# InfoNCE loss with optional hard negatives
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (NT-Xent) loss.

    Given:
        q_emb   : [B, D]  query embeddings (L2-normalised)
        p_emb   : [B, D]  positive embeddings (L2-normalised)
        n_emb   : [N, D]  optional hard-negative embeddings (L2-normalised)
        n_counts: [B]     how many hard negatives belong to each query

    The similarity matrix is q @ p.T scaled by 1/τ.  Labels are arange(B)
    (diagonal).  Hard negatives for query i are appended as extra negative
    columns so they contribute to the denominator of query i's softmax.
    """

    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(init_temperature).log())

    @property
    def temperature(self) -> float:
        return self.log_tau.exp().item()

    def forward(
        self,
        q_emb: torch.Tensor,
        p_emb: torch.Tensor,
        n_emb: Optional[torch.Tensor] = None,
        n_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = q_emb.size(0)
        tau = self.log_tau.exp().clamp(min=1e-4, max=1.0)

        sim = (q_emb @ p_emb.T) / tau                      # [B, B]
        labels = torch.arange(B, device=q_emb.device)

        if n_emb is not None and n_emb.size(0) > 0:
            n_sim = (q_emb @ n_emb.T) / tau                # [B, N_total]
            max_n = int(n_counts.max().item()) if n_counts is not None else n_emb.size(0)
            neg_cols = torch.full((B, max_n), float("-inf"), device=q_emb.device)
            offset = 0
            for i, cnt in enumerate(n_counts):
                cnt = int(cnt.item())
                if cnt > 0:
                    neg_cols[i, :cnt] = n_sim[i, offset: offset + cnt]
                offset += cnt
            sim = torch.cat([sim, neg_cols], dim=1)         # [B, B + max_n]

        sim_T = sim[:B, :B].T
        loss_q = F.cross_entropy(sim, labels)
        loss_p = F.cross_entropy(sim_T, labels)
        return (loss_q + loss_p) / 2.0


# ---------------------------------------------------------------------------
# Per-process training worker
# ---------------------------------------------------------------------------

def _worker(rank: int, world_size: int, cfg: dict):
    """
    Runs the full training loop on a single GPU (rank).
    Called directly for single-GPU; spawned via mp.spawn for multi-GPU.
    """
    use_ddp = world_size > 1

    if use_ddp:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = rank == 0

    if is_main:
        print(f"Device: {device}  |  World size: {world_size}", flush=True)

    # --- datasets & loaders --------------------------------------------------
    train_cfg = cfg["training"]
    neg_path = cfg["data"].get("negatives_path")

    train_ds = FormulaRetrievalDataset(
        split="train",
        negatives_path=neg_path,
        num_hard_negatives=train_cfg.get("num_hard_negatives", 4),
        seed=train_cfg.get("seed", 42),
    )
    eval_ds = FormulaRetrievalDataset(
        split="eval",
        negatives_path=None,
        num_hard_negatives=0,
        seed=0,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # --- model ---------------------------------------------------------------
    encoder = FormulaEncoder(cfg["model"]).to(device)
    loss_fn = InfoNCELoss(
        init_temperature=train_cfg.get("temperature_init", 0.07)
    ).to(device)

    if use_ddp:
        encoder = DDP(encoder, device_ids=[rank])

    # --- optimiser & scheduler -----------------------------------------------
    params = list(encoder.parameters()) + list(loss_fn.parameters())
    optimiser = torch.optim.AdamW(
        params,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = int(train_cfg.get("warmup_ratio", 0.05) * total_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=float(train_cfg["lr"]),
        total_steps=total_steps,
        pct_start=warmup_steps / max(total_steps, 1),
        anneal_strategy="cos",
    )

    # Mixed precision (bf16 on A100)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    # --- checkpoint dir ------------------------------------------------------
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints/gnn_stage1"))
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- early stopping state ------------------------------------------------
    patience = train_cfg.get("early_stopping_patience", 5)
    best_val_loss = float("inf")
    patience_counter = 0
    stop_flag = torch.tensor(0, dtype=torch.int32)  # broadcast to workers

    # --- training loop -------------------------------------------------------
    for epoch in range(1, train_cfg["epochs"] + 1):
        encoder.train()
        loss_fn.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main)
        for batch in loop:
            q_batch = batch["query_batch"].to(device)
            p_batch = batch["pos_batch"].to(device)
            n_batch = batch["neg_batch"]
            n_counts = batch["neg_counts"].to(device)

            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                model = encoder.module if use_ddp else encoder
                q_emb = model(q_batch, normalize=True)
                p_emb = model(p_batch, normalize=True)
                n_emb = model(n_batch.to(device), normalize=True) if n_batch is not None else None
                loss = loss_fn(q_emb, p_emb, n_emb, n_counts)

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            loop.set_postfix(loss=f"{loss.item():.4f}", tau=f"{loss_fn.temperature:.4f}")

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # --- validation ------------------------------------------------------
        encoder.eval()
        loss_fn.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in eval_loader:
                q_batch = batch["query_batch"].to(device)
                p_batch = batch["pos_batch"].to(device)

                with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                    model = encoder.module if use_ddp else encoder
                    q_emb = model(q_batch, normalize=True)
                    p_emb = model(p_batch, normalize=True)
                    loss = loss_fn(q_emb, p_emb)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        # Reduce val loss across ranks so all ranks agree on early stopping
        if use_ddp:
            val_tensor = torch.tensor(avg_val_loss, device=device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            avg_val_loss = val_tensor.item()

        if is_main:
            print(
                f"Epoch {epoch:3d} | train loss {avg_train_loss:.4f} "
                f"| val loss {avg_val_loss:.4f} | τ={loss_fn.temperature:.4f}",
                flush=True,
            )

        # --- checkpointing & early stopping (main rank only) -----------------
        if is_main:
            raw_model = encoder.module if use_ddp else encoder
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                stop_flag.fill_(0)
                raw_model.save(
                    ckpt_dir / "best.pt",
                    extra={"epoch": epoch, "val_loss": avg_val_loss},
                )
                print(f"  → New best (val loss {best_val_loss:.4f})", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered (no improvement for {patience} epochs).",
                        flush=True,
                    )
                    stop_flag.fill_(1)

            raw_model.save(
                ckpt_dir / "latest.pt",
                extra={"epoch": epoch, "val_loss": avg_val_loss},
            )

        # Broadcast stop decision from rank 0 so all workers exit together
        if use_ddp:
            stop_flag = stop_flag.to(device)
            dist.broadcast(stop_flag, src=0)
            stop_flag = stop_flag.cpu()

        if stop_flag.item() == 1:
            break

    if use_ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point — auto-spawns workers, no torchrun needed
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GNN formula encoder (Stage 1)")
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs/gnn_stage1.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_gpus = torch.cuda.device_count()

    if n_gpus > 1:
        print(f"Spawning {n_gpus} DDP workers …", flush=True)
        mp.spawn(
            _worker,
            args=(n_gpus, cfg),
            nprocs=n_gpus,
            join=True,
        )
    else:
        # Single GPU or CPU — run directly in this process (easier to debug)
        _worker(rank=0, world_size=1, cfg=cfg)


if __name__ == "__main__":
    main()
