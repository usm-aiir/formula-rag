"""
Formula Encoder schemas for Single and Dual-Encoder architectures.

FormulaEncoder: thin wrapper around GATFormulaEncoder that handles:
  - Loading / saving checkpoints with config embedded
  - Single-formula inference from an OPT string (used during index building)
  - Batch inference from a list of OPT strings

Keeping this layer separate from the GNN itself makes it easy to swap the
backbone (e.g. to a TreeLSTM or Graph Transformer) without touching the
training or eval scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch  # NEW IMPORT


from src.data.formula_graph import opt_to_pyg, slt_to_pyg
from src.task3.model.gnn import GATFormulaEncoder


# ===========================================================================
# SINGLE BRANCH ENCODER (Used to load Phase 1 & 2 Checkpoints)
# ===========================================================================
class FormulaEncoder(nn.Module):
    """Wraps the single-branch GAT for backward compatibility with older checkpoints."""
    
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        
        # If loading an old checkpoint, default to 81 and 256
        v_size = cfg.get("opt_vocab_size", 81)
        out_dim = cfg.get("output_dim", cfg.get("branch_output_dim", 256))
        
        self.gnn = GATFormulaEncoder(
            vocab_size=v_size,
            node_emb_dim=cfg["node_emb_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            output_dim=out_dim,
            dropout=cfg.get("dropout", 0.1),
        )

    def forward(self, batch: Batch, normalize: bool = True) -> torch.Tensor:
        return self.gnn(batch, normalize=normalize)

    @torch.no_grad()
    def encode_batch_opt(
        self,
        opt_list: List[Optional[str]],
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, List[bool]]:
        """Restored for the evaluation and mining scripts."""
        graphs, valid_mask = [], []
        for opt in opt_list:
            g = opt_to_pyg(opt) if opt else None
            graphs.append(g)
            valid_mask.append(g is not None)

        valid_graphs = [g for g in graphs if g is not None]
        if not valid_graphs:
            # Safely grab the actual output dim of the initialized GNN
            return torch.empty(0, self.gnn.proj[-1].out_features), valid_mask

        self.eval()
        batch = Batch.from_data_list(valid_graphs).to(device)
        return self.gnn(batch, normalize=True), valid_mask

    def save(self, path: Union[str, Path], extra: Optional[dict] = None):
        payload = {"cfg": self.cfg, "state_dict": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: Union[str, Path], map_location: Union[str, torch.device] = "cpu") -> "FormulaEncoder":
        payload = torch.load(str(path), map_location=map_location)
        encoder = cls(payload["cfg"])
        encoder.load_state_dict(payload["state_dict"])
        return encoder

    @property
    def output_dim(self) -> int:
        # Fallback logic to support both old and new configs
        return self.cfg.get("output_dim", self.cfg.get("branch_output_dim", 256))

# ===========================================================================
# CROSS-ATTENTION DUAL ENCODER (Phase 3)
# ===========================================================================
class DualFormulaEncoder(nn.Module):
    """
    Cross-Attention Dual-Encoder for Math Formulas.
    Branch 1: Semantic Logic (OPT)
    Branch 2: Visual Layout (SLT)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        
        self.opt_gnn = GATFormulaEncoder(
            vocab_size=cfg["opt_vocab_size"], 
            node_emb_dim=cfg["node_emb_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            output_dim=cfg["branch_output_dim"], 
            dropout=cfg.get("dropout", 0.1),
        )

        self.slt_gnn = GATFormulaEncoder(
            vocab_size=cfg["slt_vocab_size"], 
            node_emb_dim=cfg["node_emb_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            output_dim=cfg["branch_output_dim"], 
            dropout=cfg.get("dropout", 0.1),
        )

        # Bi-directional Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg["branch_output_dim"],
            num_heads=cfg.get("num_fusion_heads", 4),
            dropout=cfg.get("dropout", 0.1),
            batch_first=True
        )
        
        # Layer Norms to stabilize the residual connections
        self.norm_opt = nn.LayerNorm(cfg["branch_output_dim"])
        self.norm_slt = nn.LayerNorm(cfg["branch_output_dim"])

        # The MLP processes the attention-enriched vectors
        fused_dim = cfg["branch_output_dim"] * 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, cfg["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(cfg.get("dropout", 0.1)),
            nn.Linear(cfg["hidden_dim"], cfg["final_output_dim"]) 
        )

    def forward(
        self, 
        opt_batch: Batch, 
        slt_batch: Batch, 
        normalize: bool = True
    ) -> torch.Tensor:
        
        # Get independent embeddings [Batch, 256]
        opt_emb = self.opt_gnn(opt_batch, normalize=False)
        slt_emb = self.slt_gnn(slt_batch, normalize=False)
        
        # Reshape to Sequence Length of 1 for Attention -> [Batch, 1, 256]
        opt_seq = opt_emb.unsqueeze(1)
        slt_seq = slt_emb.unsqueeze(1)
        
        # Cross-Attention (Q, K, V)
        # OPT asks SLT for context
        opt_attended, _ = self.cross_attn(query=opt_seq, key=slt_seq, value=slt_seq)
        
        # SLT asks OPT for context
        slt_attended, _ = self.cross_attn(query=slt_seq, key=opt_seq, value=opt_seq)
        
        # Squeeze back to [Batch, 256]
        opt_attended = opt_attended.squeeze(1)
        slt_attended = slt_attended.squeeze(1)
        
        # Residual Connection + LayerNorm (Crucial for deep stability)
        opt_fused = self.norm_opt(opt_emb + opt_attended)
        slt_fused = self.norm_slt(slt_emb + slt_attended)
        
        # Final MLP Fusion
        fused = torch.cat([opt_fused, slt_fused], dim=-1)
        out = self.fusion_mlp(fused)
        
        if normalize:
            out = F.normalize(out, p=2, dim=-1)
            
        return out

    def save(self, path: Union[str, Path], extra: Optional[dict] = None):
        payload = {"cfg": self.cfg, "state_dict": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: Union[str, Path], map_location: Union[str, torch.device] = "cpu") -> "DualFormulaEncoder":
        payload = torch.load(str(path), map_location=map_location, weights_only=True)
        encoder = cls(payload["cfg"])
        encoder.load_state_dict(payload["state_dict"])
        return encoder

    @property
    def output_dim(self) -> int:
        return self.cfg.get("final_output_dim", 256)