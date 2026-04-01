"""
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
from torch_geometric.data import Batch

from src.data.formula_graph import opt_to_pyg
from src.task3.model.gnn import GATFormulaEncoder


class FormulaEncoder(nn.Module):
    """
    Wraps GATFormulaEncoder and adds checkpoint / OPT-string convenience methods.
    """

    def __init__(self, cfg: dict):
        """
        Parameters
        ----------
        cfg : dict
            Corresponds to the ``model`` section of task3.yaml.
        """
        super().__init__()
        self.cfg = cfg
        self.gnn = GATFormulaEncoder(
            node_emb_dim=cfg["node_emb_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            output_dim=cfg["output_dim"],
            dropout=cfg.get("dropout", 0.1),
        )

    def forward(self, batch: Batch, normalize: bool = True) -> torch.Tensor:
        """Encode a PyG Batch → [G, output_dim] embeddings."""
        return self.gnn(batch, normalize=normalize)

    @torch.no_grad()
    def encode_opt(
        self,
        opt_xml: str,
        device: Union[str, torch.device] = "cpu",
    ) -> Optional[torch.Tensor]:
        """Encode a single OPT XML string → 1-D embedding, or None if unparseable."""
        graph = opt_to_pyg(opt_xml)
        if graph is None:
            return None
        self.eval()
        batch = Batch.from_data_list([graph]).to(device)
        return self.gnn(batch, normalize=True)[0]

    @torch.no_grad()
    def encode_batch_opt(
        self,
        opt_list: List[Optional[str]],
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, List[bool]]:
        """
        Encode a list of OPT strings.

        Returns
        -------
        embeddings : Tensor[num_valid, output_dim]
        valid_mask : list[bool] of length len(opt_list)
        """
        graphs, valid_mask = [], []
        for opt in opt_list:
            g = opt_to_pyg(opt) if opt else None
            graphs.append(g)
            valid_mask.append(g is not None)

        valid_graphs = [g for g in graphs if g is not None]
        if not valid_graphs:
            return torch.empty(0, self.cfg["output_dim"]), valid_mask

        self.eval()
        batch = Batch.from_data_list(valid_graphs).to(device)
        return self.gnn(batch, normalize=True), valid_mask

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path], extra: Optional[dict] = None):
        """Save model weights + config + optional extra metadata."""
        payload = {"cfg": self.cfg, "state_dict": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, str(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
    ) -> "FormulaEncoder":
        """Load a checkpoint saved with .save()."""
        payload = torch.load(str(path), map_location=map_location)
        encoder = cls(payload["cfg"])
        encoder.load_state_dict(payload["state_dict"])
        return encoder

    @property
    def output_dim(self) -> int:
        return self.cfg["output_dim"]
