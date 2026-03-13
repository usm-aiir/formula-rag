"""
Graph Attention Network (GAT) encoder for Content MathML (OPT) trees.

Architecture
------------
1. Node embedding lookup: integer type-ID → dense vector (trainable).
2. N × GATConv layers with residual connections and layer norm.
3. Global mean + max pooling (concatenated) → formula representation.

The pooling strategy is intentionally conservative: mean-pool captures the
"average" operator distribution while max-pool captures the most salient
operator in each dimension.  Both together outperform either alone on
tree-structured retrieval tasks without adding parameters.

Graph sizes are bounded at MAX_NODES=256 (enforced in src/data/formula_graph.py),
so memory consumption per batch is predictable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, LayerNorm, global_max_pool, global_mean_pool

from src.data.formula_graph import VOCAB_SIZE


class GATFormulaEncoder(nn.Module):
    """
    GAT-based encoder that maps a PyG Batch of OPT graphs to a matrix of
    L2-normalised embeddings, one row per graph.

    Parameters
    ----------
    node_emb_dim : int
        Dimension of the learnable node-type embedding table.
    hidden_dim : int
        Hidden dimension per GATConv head output (before concatenation).
    num_heads : int
        Number of attention heads per GATConv layer.
    num_layers : int
        Number of GATConv layers.
    output_dim : int
        Final embedding dimension (after the projection MLP).
    dropout : float
        Dropout rate applied between layers and on attention weights.
    """

    def __init__(
        self,
        node_emb_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dropout = dropout
        in_dim = node_emb_dim

        self.node_emb = nn.Embedding(VOCAB_SIZE, node_emb_dim, padding_idx=0)

        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        for _ in range(num_layers):
            gat = GATConv(
                in_channels=in_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True,
                add_self_loops=True,
            )
            gat_out_dim = num_heads * hidden_dim
            self.gat_layers.append(gat)
            self.layer_norms.append(LayerNorm(gat_out_dim))
            self.residual_projs.append(
                nn.Linear(in_dim, gat_out_dim, bias=False)
                if in_dim != gat_out_dim
                else nn.Identity()
            )
            in_dim = gat_out_dim

        pooled_dim = 2 * in_dim
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_emb.weight)
        for module in self.proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: Batch, normalize: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : PyG Batch
            Batched graph with x (node type IDs), edge_index, and batch vector.
        normalize : bool
            If True, L2-normalise the output embeddings.

        Returns
        -------
        Tensor of shape [num_graphs, output_dim]
        """
        x = self.node_emb(batch.x)
        edge_index = batch.edge_index
        batch_vec = batch.batch

        for gat, ln, res_proj in zip(self.gat_layers, self.layer_norms, self.residual_projs):
            residual = res_proj(x)
            x = gat(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = ln(x + residual, batch_vec)

        x_pool = torch.cat([global_mean_pool(x, batch_vec),
                             global_max_pool(x, batch_vec)], dim=-1)
        out = self.proj(x_pool)

        if normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out
