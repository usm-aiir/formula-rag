"""
Structure-aware formula encoder.

Input:  A mathematical formula as a Content MathML string (OPT representation
        from the ARQMath formula index).
Output: An embedding vector in CLIP space (via a learned projection head).

Pipeline:
  Content MathML string
    → parse XML → tree of MathML nodes
    → build PyG graph (nodes = elements, edges = parent→child)
    → GIN (Graph Isomorphism Network) layers
    → global mean pooling
    → MLP projection head → R^clip_dim

The GIN is more expressive than GCN for graph-level tasks (Xu et al., 2019).
The projection head aligns formula embeddings with CLIP space during Stage 2
training (Task 1 qrels). Stage 1 training (Task 2 qrels) trains the GIN in
a formula-similarity space before projection.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_mean_pool


# ── Node vocabulary ───────────────────────────────────────────────────────────
# Content MathML element tag names. Covers the vast majority of MSE formulas.
# Unknown tags fall back to the UNK token.

_MATHML_TAGS = [
    # Structure
    "math", "apply", "lambda", "piecewise", "piece", "otherwise",
    # Arithmetic
    "plus", "minus", "times", "divide", "power", "root", "abs",
    "factorial", "floor", "ceiling", "exp", "ln", "log",
    # Calculus
    "diff", "partialdiff", "int", "sum", "product", "limit",
    "tendsto", "divergence", "grad", "curl", "laplacian",
    # Algebra
    "eq", "neq", "lt", "gt", "leq", "geq",
    "and", "or", "not", "implies", "forall", "exists",
    # Sets
    "set", "list", "union", "intersect", "in", "notin",
    "subset", "prsubset", "emptyset", "cartesianproduct",
    # Linear algebra
    "matrix", "matrixrow", "vector", "transpose", "determinant",
    "selector",
    # Leaves
    "cn", "ci", "csymbol", "true", "false", "infinity", "pi",
    "eulergamma", "exponentiale",
    # MathML wrappers
    "bvar", "degree", "lowlimit", "uplimit", "condition",
    "interval", "domainofapplication", "annotation",
]

# Build tag → index mapping; 0 reserved for UNK
TAG2IDX: dict[str, int] = {tag: i + 1 for i, tag in enumerate(_MATHML_TAGS)}
VOCAB_SIZE: int = len(TAG2IDX) + 1  # +1 for UNK


# ── MathML → PyG graph ────────────────────────────────────────────────────────

def _strip_namespace(tag: str) -> str:
    """Remove XML namespace prefix: '{http://...}plus' → 'plus'."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def mathml_to_graph(mathml_str: str) -> Optional[Data]:
    """
    Parse a Content MathML string and return a PyG Data object.

    Node features: one-hot index into TAG2IDX (shape: [n_nodes, 1] as int)
    Edge index:    directed parent → child edges

    Returns None if parsing fails or the tree is empty.
    """
    try:
        root = ET.fromstring(mathml_str)
    except ET.ParseError:
        return None

    node_tags: list[int] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []

    # BFS to assign node indices and collect edges
    queue: list[tuple[ET.Element, int]] = [(root, 0)]
    node_tags.append(TAG2IDX.get(_strip_namespace(root.tag), 0))
    node_counter = 1

    while queue:
        elem, parent_idx = queue.pop(0)
        for child in elem:
            child_idx = node_counter
            node_counter += 1
            tag = _strip_namespace(child.tag)
            node_tags.append(TAG2IDX.get(tag, 0))
            # Parent → child edge
            edge_src.append(parent_idx)
            edge_dst.append(child_idx)
            # Child → parent edge (undirected GNN)
            edge_src.append(child_idx)
            edge_dst.append(parent_idx)
            queue.append((child, child_idx))

    if node_counter == 0:
        return None

    x = torch.tensor(node_tags, dtype=torch.long)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, num_nodes=node_counter)


# ── GIN model ─────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FormulaGNN(nn.Module):
    """
    Graph Isomorphism Network for formula encoding.

    Architecture:
      Embedding(VOCAB_SIZE, node_dim)
        → num_layers × GINConv(MLP)
        → global mean pooling
        → MLP projection head → out_dim

    The out_dim should match the CLIP embedding dimension (default 512).
    """

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        out_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, node_dim, padding_idx=0)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dims = [node_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            mlp = _MLP(dims[i], dims[i + 1], dims[i + 1])
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        self.dropout = nn.Dropout(dropout)

        # Projection head: GNN output → CLIP space
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data: Data | Batch) -> Tensor:
        x = self.embedding(data.x)          # [n_nodes, node_dim]
        edge_index = data.edge_index

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Graph-level pooling
        x = global_mean_pool(x, data.batch)  # [n_graphs, hidden_dim]
        x = self.proj(x)                     # [n_graphs, out_dim]
        return x


# ── High-level encoder ────────────────────────────────────────────────────────

class FormulaEncoder:
    """
    Encodes Content MathML formula strings into embedding vectors.

    Wraps FormulaGNN with MathML parsing, batching, and optional L2 norm.
    Call .to_train_mode() before Stage 1/2 training; .to_eval_mode() for inference.
    """

    def __init__(
        self,
        model: FormulaGNN | None = None,
        device: str | None = None,
        clip_dim: int = 512,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or FormulaGNN(out_dim=clip_dim)
        self.model = self.model.to(self.device)
        self.model.eval()

    def to_train_mode(self) -> None:
        self.model.train()

    def to_eval_mode(self) -> None:
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        mathml_strings: list[str],
        batch_size: int = 128,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Encode a list of Content MathML strings.

        Formulas that fail to parse are skipped. Returns embeddings alongside
        a list of the original indices that succeeded, so callers can track
        which formulas were encoded.

        Args:
            mathml_strings — list of Content MathML strings
            batch_size     — number of graphs per forward pass
            normalize      — if True, L2-normalize embeddings

        Returns:
            (embeddings tensor of shape [n_valid, dim], list of valid indices)
        """
        graphs: list[Data] = []
        valid_indices: list[int] = []

        for i, s in enumerate(mathml_strings):
            g = mathml_to_graph(s)
            if g is not None:
                graphs.append(g)
                valid_indices.append(i)

        if not graphs:
            return torch.empty(0, self.model.proj[-1].out_features), []

        all_embeddings: list[torch.Tensor] = []

        for i in range(0, len(graphs), batch_size):
            batch_graphs = graphs[i : i + batch_size]
            batch = Batch.from_data_list(batch_graphs).to(self.device)
            emb = self.model(batch)
            if normalize:
                emb = F.normalize(emb, dim=-1)
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0), valid_indices

    def encode_single(
        self, mathml_str: str, normalize: bool = True
    ) -> torch.Tensor | None:
        """
        Encode a single formula. Returns None if parsing fails.
        Returns shape (dim,).
        """
        embs, valid = self.encode([mathml_str], normalize=normalize)
        if not valid:
            return None
        return embs.squeeze(0)

    def save(self, path: str | Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
