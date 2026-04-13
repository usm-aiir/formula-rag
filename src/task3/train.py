"""
Train GNN formula encoder on ARQMath Task 2 (Formula Retrieval).
Phase 2: Self-Adversarial Hard Negative Fine-Tuning.

Loss: InfoNCE (NT-Xent) with in-batch negatives AND explicit hard negatives.
Temperature is a learnable scalar (log-parameterised for stability).

Usage:
  python -m src.task3.train --config configs/task3.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

# NEW: Import the Phase 2 Dataset
from src.task3.dataset import FormulaRetrievalDataset, Phase2Dataset, collate_fn
from src.task3.model.formula_encoder import FormulaEncoder


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) with in-batch negatives + explicit hard negatives.
    """

    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(init_temperature).log())

    @property
    def temperature(self) -> float:
        return self.log_tau.exp().item()

    def forward(self, q_emb: torch.Tensor, p_emb: torch.Tensor, hn_emb: torch.Tensor = None) -> torch.Tensor:
        tau = self.log_tau.exp().clamp(min=1e-4, max=1.0)
        labels = torch.arange(q_emb.size(0), device=q_emb.device)

        if hn_emb is None:
            # Standard In-Batch Negatives (Used during Eval)
            sim = (q_emb @ p_emb.T) / tau
            return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        else:
            # Explicit Hard Negatives
            # Concatenate Positives and Hard Negatives into a single pool (Size: 2N x D)
            keys = torch.cat([p_emb, hn_emb], dim=0) 
            
            # Query similarity against ALL keys
            sim = (q_emb @ keys.T) / tau 
            
            # The true label is still the index of the Positive (0 to N-1)
            # This pushes the Query away from in-batch negatives AND the Hard Negative (index i + N)
            loss_q = F.cross_entropy(sim, labels)

            # Keep the symmetric positive-to-query constraint for stability
            sim_p = (p_emb @ q_emb.T) / tau
            loss_p = F.cross_entropy(sim_p, labels)

            return (loss_q + loss_p) / 2


# ---------------------------------------------------------------------------
# Per-process training worker
# ---------------------------------------------------------------------------

def _worker(rank: int, world_size: int, cfg: dict):
    use_ddp = world_size > 1

    if use_ddp:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = rank == 0
    if is_main:
        print(f"Device: {device}  |  World size: {world_size}", flush=True)

    # --- datasets & loaders --------------------------------------------------
    train_cfg = cfg["training"]

    train_ds = Phase2Dataset(_PROJECT_ROOT / "data/processed/self_mined_hard_negatives.jsonl")
    
    eval_ds = FormulaRetrievalDataset(split="eval", seed=0)

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if use_ddp else None
    )
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
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/task3"))
    phase1_ckpt = ckpt_dir / "best.pt"
    
    # Load Phase 1 weights so we don't start from scratch
    if phase1_ckpt.exists():
        if is_main:
            print(f"Loading Phase 1 weights from {phase1_ckpt}", flush=True)
        encoder = FormulaEncoder.load(str(phase1_ckpt), map_location=device).to(device)
    else:
        if is_main:
            print("WARNING: Phase 1 weights not found, starting from scratch!", flush=True)
        encoder = FormulaEncoder(cfg["model"]).to(device)
        
    loss_fn = InfoNCELoss(init_temperature=train_cfg.get("temperature_init", 0.07)).to(device)

    if use_ddp:
        encoder = DDP(encoder, device_ids=[rank])

    # --- optimiser & scheduler -----------------------------------------------
    params = list(encoder.parameters()) + list(loss_fn.parameters())
    
    # For fine-tuning, might want to use a slightly lower LR, but OneCycleLR handles this well
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

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- early stopping state ------------------------------------------------
    patience = train_cfg.get("early_stopping_patience", 5)
    best_val_loss = float("inf")
    patience_counter = 0
    stop_flag = torch.tensor(0, dtype=torch.int32)

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
            
            # NEW: Safely extract hard negative batch if it exists
            hn_batch = batch.get("hard_neg_batch")

            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                model = encoder.module if use_ddp else encoder
                q_emb = model(q_batch, normalize=True)
                p_emb = model(p_batch, normalize=True)
                hn_emb = model(hn_batch.to(device), normalize=True) if hn_batch is not None else None
                
                loss = loss_fn(q_emb, p_emb, hn_emb)

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
                # Eval sets don't have hard negatives
                hn_batch = batch.get("hard_neg_batch")

                with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                    model = encoder.module if use_ddp else encoder
                    q_emb = model(q_batch, normalize=True)
                    p_emb = model(p_batch, normalize=True)
                    hn_emb = model(hn_batch.to(device), normalize=True) if hn_batch is not None else None
                    
                    loss = loss_fn(q_emb, p_emb, hn_emb)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        if use_ddp:
            val_tensor = torch.tensor(avg_val_loss, device=device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            avg_val_loss = val_tensor.item()

        if is_main:
            print(
                f"Epoch {epoch:3d}  |  train_loss={avg_train_loss:.4f}  "
                f"val_loss={avg_val_loss:.4f}  tau={loss_fn.temperature:.4f}",
                flush=True,
            )

        # --- checkpointing & early stopping ----------------------------------
        if is_main:
            raw_model = encoder.module if use_ddp else encoder
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                stop_flag.fill_(0)
                
                # Save as phase2 so we don't overwrite the baseline
                raw_model.save(
                    ckpt_dir / "phase2_best.pt",
                    extra={"epoch": epoch, "val_loss": avg_val_loss},
                )
                print(f"  → saved best checkpoint (val_loss={best_val_loss:.4f})", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping ({patience} epochs without improvement)", flush=True)
                    stop_flag.fill_(1)

            raw_model.save(
                ckpt_dir / "phase2_latest.pt",
                extra={"epoch": epoch, "val_loss": avg_val_loss},
            )

        if use_ddp:
            stop_flag = stop_flag.to(device)
            dist.broadcast(stop_flag, src=0)
            stop_flag = stop_flag.cpu()

        if stop_flag.item() == 1:
            break

    if use_ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GNN formula encoder (Task 2)")
    parser.add_argument("--config", type=Path, default=_PROJECT_ROOT / "configs/task3.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Spawning {n_gpus} DDP workers …", flush=True)
        mp.spawn(_worker, args=(n_gpus, cfg), nprocs=n_gpus, join=True)
    else:
        _worker(rank=0, world_size=1, cfg=cfg)


if __name__ == "__main__":
    main()