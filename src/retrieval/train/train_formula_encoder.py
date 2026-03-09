"""
Stage 1: Pre-train the formula encoder on ARQMath Task 2 qrels.

Task 2: given a formula query, retrieve similar formulas. The qrels tell us
which formulas are considered equivalent (same visual group / visual_id).

We treat this as contrastive learning:
  - Positive pair:  (query formula, relevant formula from qrels, relevance > 0)
  - Negatives:      all other formulas in the same batch (in-batch negatives)
  - Loss:           InfoNCE (NT-Xent cross-entropy over cosine similarities)

After Stage 1, the GNN backbone has learned structural formula similarity.
The projection head will be re-aligned to CLIP space in Stage 2.

Usage:
    python -m src.retrieval.train.train_formula_encoder \\
        --epochs 10 --batch-size 64 --lr 1e-4

Outputs:
    checkpoints/formula_encoder_stage1.pt
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.encoders import FormulaEncoder, FormulaGNN

# ── Paths ─────────────────────────────────────────────────────────────────────
OPT_DIR     = REPO_ROOT / "data/raw/arqmath/formulas/opt_representation_v3"
QRELS_DIR   = REPO_ROOT / "data/raw/arqmath/qrels/task2"
CKPT_DIR    = REPO_ROOT / "checkpoints"

# Use ARQMath-1 + ARQMath-2 qrels for training
TRAIN_QRELS = [
    QRELS_DIR / "arqmath1/qrel_task2_2020_all.tsv",
    QRELS_DIR / "arqmath2/qrel_task2_2021_all.tsv",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_opt_index(opt_dir: Path) -> dict[str, str]:
    """
    Stream OPT TSV files → {formula_id_str: mathml_string}.

    Also returns {visual_id: [formula_id, ...]} for qrel matching.
    Returns (formula_id→mathml, visual_id→[formula_ids]).
    """
    id_to_mathml: dict[str, str] = {}
    visual_to_ids: dict[str, list[str]] = defaultdict(list)

    tsv_files = sorted(opt_dir.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No OPT TSV files found in {opt_dir}")

    for tsv in tqdm(tsv_files, desc="Loading OPT index"):
        with tsv.open(encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 9:
                    continue
                formula_id = row[0].strip()
                visual_id  = row[6].strip()   # new_visual_id column
                issue      = row[7].strip()
                mathml     = row[8].strip().strip('"')
                if issue in ("d", "dv") or not mathml:
                    continue
                id_to_mathml[formula_id] = mathml
                if visual_id:
                    visual_to_ids[visual_id].append(formula_id)

    return id_to_mathml, dict(visual_to_ids)


def load_task2_qrels(qrel_paths: list[Path]) -> list[tuple[str, str]]:
    """
    Load Task 2 qrels → list of (query_visual_id, relevant_visual_id) pairs
    where relevance > 0.

    Task 2 qrel format: topic_id  0  visual_id  relevance
    (The topic_id maps to a formula's visual_id via the topic XML, but for
    training we treat each topic as a visual group and build positive pairs
    between all visual groups with relevance > 0.)
    """
    pairs: list[tuple[str, str]] = []
    for path in qrel_paths:
        if not path.exists():
            print(f"Warning: qrel file not found: {path}")
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                topic_id   = parts[0]
                visual_id  = parts[2]
                relevance  = int(parts[3])
                if relevance > 0:
                    pairs.append((topic_id, visual_id))
    return pairs


# ── Dataset ───────────────────────────────────────────────────────────────────

class FormulaPairDataset(Dataset):
    """
    Each item is (query_mathml, positive_mathml).
    Negatives are handled in-batch by the InfoNCE loss.
    """

    def __init__(
        self,
        id_to_mathml: dict[str, str],
        visual_to_ids: dict[str, list[str]],
        qrel_pairs: list[tuple[str, str]],
    ) -> None:
        self.id_to_mathml   = id_to_mathml
        self.visual_to_ids  = visual_to_ids

        # Build (query_formula_id, positive_formula_id) pairs
        self.pairs: list[tuple[str, str]] = []
        for q_visual, d_visual in qrel_pairs:
            q_ids = visual_to_ids.get(q_visual, [])
            d_ids = visual_to_ids.get(d_visual, [])
            if q_ids and d_ids:
                # Sample one formula from each group
                self.pairs.append((random.choice(q_ids), random.choice(d_ids)))

        print(f"Dataset: {len(self.pairs):,} positive pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        q_id, d_id = self.pairs[idx]
        return self.id_to_mathml[q_id], self.id_to_mathml[d_id]


# ── Loss ──────────────────────────────────────────────────────────────────────

def infonce_loss(q_embs: torch.Tensor, d_embs: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE loss over a batch of (query, document) pairs.
    Assumes embeddings are already L2-normalised.

    Diagonal entries are the positive pairs; all off-diagonal are negatives.
    """
    logits = (q_embs @ d_embs.T) / temperature   # [B, B]
    labels = torch.arange(len(q_embs), device=q_embs.device)
    loss_q = F.cross_entropy(logits, labels)
    loss_d = F.cross_entropy(logits.T, labels)
    return (loss_q + loss_d) / 2


# ── Training loop ─────────────────────────────────────────────────────────────

def collate_fn(batch: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    queries, docs = zip(*batch)
    return list(queries), list(docs)


def train(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    temperature: float = 0.07,
    node_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 4,
    clip_dim: int = 512,
    save_every: int = 1,
) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("Loading OPT formula index (this may take several minutes)...")
    id_to_mathml, visual_to_ids = load_opt_index(OPT_DIR)
    print(f"  {len(id_to_mathml):,} formulas loaded")

    qrel_pairs = load_task2_qrels(TRAIN_QRELS)
    print(f"  {len(qrel_pairs):,} positive qrel pairs loaded")

    dataset = FormulaPairDataset(id_to_mathml, visual_to_ids, qrel_pairs)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Model
    model = FormulaGNN(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=clip_dim,
    ).to(device)

    encoder = FormulaEncoder(model=model, device=device, clip_dim=clip_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for q_strs, d_strs in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            encoder.to_train_mode()

            # Encode query and positive document formulas
            q_embs, q_valid = encoder.encode(q_strs, normalize=True)
            d_embs, d_valid = encoder.encode(d_strs, normalize=True)

            # Keep only pairs where both formulas parsed successfully
            valid_mask = set(q_valid) & set(d_valid)
            if len(valid_mask) < 2:
                continue

            q_idx = [q_valid.index(i) for i in sorted(valid_mask)]
            d_idx = [d_valid.index(i) for i in sorted(valid_mask)]

            q_embs = q_embs[q_idx].to(device)
            d_embs = d_embs[d_idx].to(device)

            loss = infonce_loss(q_embs, d_embs, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}: avg loss = {avg_loss:.4f}")

        if epoch % save_every == 0:
            ckpt = CKPT_DIR / f"formula_encoder_stage1_epoch{epoch}.pt"
            encoder.save(ckpt)
            print(f"  Saved: {ckpt}")

    # Save final checkpoint
    final_ckpt = CKPT_DIR / "formula_encoder_stage1.pt"
    encoder.save(final_ckpt)
    print(f"\nStage 1 complete. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: pre-train formula encoder")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-layers",  type=int,   default=4)
    parser.add_argument("--hidden-dim",  type=int,   default=256)
    parser.add_argument("--clip-dim",    type=int,   default=512)
    parser.add_argument("--save-every",  type=int,   default=1)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        clip_dim=args.clip_dim,
        save_every=args.save_every,
    )
