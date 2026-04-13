"""
Phase 3: Late Fusion Dual Encoder Training (5-Fold Cross Validation)

Requires the phase 2 checkpoint for hybrid initialization. Each fold trains a DualFormulaEncoder with the InfoNCE loss, 
using the provided hard negative samples. 
The best model for each fold is saved separately. Early stopping is implemented based on validation loss to prevent overfitting.

Once all folds are completed, use them to create a model soup by running src/task3/utils/brew_model_soup.py with the saved checkpoints.
"""

import sys
import yaml
import json
import gc
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.task3.model.formula_encoder import DualFormulaEncoder
from src.task3.dataset import Phase3Dataset, dual_collate_fn

_CONFIG_PATH = _PROJECT_ROOT / "configs/task3.yaml"
_DATA_PATH = _PROJECT_ROOT / "data/processed/phase3_hard_negatives.jsonl"
_PHASE2_CKPT = _PROJECT_ROOT / "checkpoints/task3/phase2_best.pt"
_OUT_DIR = _PROJECT_ROOT / "checkpoints/task3/phase3_fusion"

class InfoNCELoss(nn.Module):
    def __init__(self, init_temp=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))
        
    def forward(self, q_embs, p_embs, hn_embs):
        q_embs = F.normalize(q_embs, p=2, dim=-1)
        p_embs = F.normalize(p_embs, p=2, dim=-1)
        hn_embs = F.normalize(hn_embs, p=2, dim=-1)
        
        pos_sim = (q_embs * p_embs).sum(dim=-1, keepdim=True)
        hn_sim = torch.bmm(q_embs.unsqueeze(1), hn_embs.unsqueeze(2)).squeeze(2)
        
        in_batch_sim = torch.mm(q_embs, p_embs.t())
        mask = torch.eye(q_embs.size(0), device=q_embs.device).bool()
        in_batch_sim.masked_fill_(mask, -float('inf'))
        
        logits = torch.cat([pos_sim, hn_sim, in_batch_sim], dim=-1)
        logits = logits * self.logit_scale.exp()
        
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

def get_k_folds(jsonl_path: Path, k: int = 5) -> list:
    """Reads the JSONL and splits the unique topic_ids into K folds."""
    unique_topics = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            unique_topics.add(json.loads(line)["topic_id"])
            
    topics_list = list(unique_topics)
    random.seed(42) # Deterministic splits
    random.shuffle(topics_list)
    
    folds = [topics_list[i::k] for i in range(k)]
    return folds

def run_fold(fold_idx: int, val_topics: list, train_topics: list, cfg: dict, device: torch.device):
    print(f"\n{'='*60}")
    print(f"STARTING FOLD {fold_idx + 1}/5")
    print(f"Train Topics: {len(train_topics)} | Val Topics: {len(val_topics)}")
    print(f"{'='*60}")
    
    # Dataloaders
    train_dataset = Phase3Dataset(_DATA_PATH, allowed_topics=set(train_topics))
    val_dataset = Phase3Dataset(_DATA_PATH, allowed_topics=set(val_topics))
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], 
        shuffle=True, collate_fn=dual_collate_fn, 
        num_workers=cfg["training"]["num_workers"], drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["training"]["batch_size"], 
        shuffle=False, collate_fn=dual_collate_fn, 
        num_workers=cfg["training"]["num_workers"], drop_last=False
    )
    
    # Model Init (Hybrid Initialization for each fold)
    model = DualFormulaEncoder(cfg["model"]).to(device)
    phase2_ckpt = torch.load(_PHASE2_CKPT, map_location=device, weights_only=True)
    
    opt_state_dict = {}
    for k, v in phase2_ckpt["state_dict"].items():
        if k.startswith("gnn."):
            opt_state_dict[k.replace("gnn.", "")] = v
    model.opt_gnn.load_state_dict(opt_state_dict, strict=True)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["training"]["lr"]), 
        weight_decay=float(cfg["training"]["weight_decay"])
    )
    criterion = InfoNCELoss(init_temp=cfg["training"]["temperature_init"]).to(device)
    
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"].get("early_stopping_patience", 5)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx + 1} - Epoch {epoch}/{epochs} [TRAIN]")
        for batch in pbar:
            if not batch: continue
            
            q_opt, q_slt = batch["q_opt_batch"].to(device), batch["q_slt_batch"].to(device)
            p_opt, p_slt = batch["p_opt_batch"].to(device), batch["p_slt_batch"].to(device)
            hn_opt, hn_slt = batch["hn_opt_batch"].to(device), batch["hn_slt_batch"].to(device)
            
            optimizer.zero_grad()
            q_embs = model(q_opt, q_slt)
            p_embs = model(p_opt, p_slt)
            hn_embs = model(hn_opt, hn_slt)
            
            loss = criterion(q_embs, p_embs, hn_embs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if not batch: continue
                q_opt, q_slt = batch["q_opt_batch"].to(device), batch["q_slt_batch"].to(device)
                p_opt, p_slt = batch["p_opt_batch"].to(device), batch["p_slt_batch"].to(device)
                hn_opt, hn_slt = batch["hn_opt_batch"].to(device), batch["hn_slt_batch"].to(device)
                
                q_embs = model(q_opt, q_slt)
                p_embs = model(p_opt, p_slt)
                hn_embs = model(hn_opt, hn_slt)
                
                loss = criterion(q_embs, p_embs, hn_embs)
                val_loss += loss.item()
                val_batches += 1
                
        avg_train = train_loss / max(1, train_batches)
        avg_val = val_loss / max(1, val_batches)
        print(f"Fold {fold_idx + 1} | Epoch {epoch} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        # Early Stopping Logic
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            save_path = _OUT_DIR / f"phase3_atten_fusion_fold{fold_idx + 1}_best.pt"
            model.save(save_path)
            print(f"  --> Saved new best checkpoint: {save_path.name}")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered for Fold {fold_idx + 1} at Epoch {epoch}!")
                break
            
    # Explicit Memory Cleanup to prevent OOM across folds
    del model
    del optimizer
    del criterion
    del train_loader
    del val_loader
    torch.cuda.empty_cache()
    gc.collect()

def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate the 5 distinct folds
    folds = get_k_folds(_DATA_PATH, k=5)
    
    # Iterate through folds 0, 1, 2, 3, 4
    for fold_idx in range(5):
        val_topics = folds[fold_idx]
        train_topics = []
        for i in range(5):
            if i != fold_idx:
                train_topics.extend(folds[i])
                
        # Launch the fold with the dynamic index
        run_fold(fold_idx, val_topics, train_topics, cfg, device)

if __name__ == "__main__":
    main()