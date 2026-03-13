"""
Convert Content MathML (OPT) strings into PyTorch Geometric Data objects.

Each XML element becomes a node; parent→child relationships become directed
edges (we add reverse edges to make the graph undirected for message passing).
Node features are integer type IDs looked up from a fixed vocabulary built from
the 96 unique tags observed across the full ARQMath formula index.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Node-type vocabulary
# All 96 tags seen across the corpus.
# ID 0  → unknown / unseen tag
# IDs 1… → actual MathML tag names
# ---------------------------------------------------------------------------

_TAGS: List[str] = [
    # high-frequency structural / operator tags
    "ci", "apply", "csymbol", "cn", "math", "times", "eq", "minus",
    "divide", "plus", "interval", "and", "cerror", "in", "list",
    "mtext", "infinity", "leq", "abs", "root", "sum", "matrixrow",
    "int", "lt", "gt", "geq", "set", "sin", "cos", "vector", "log",
    "factorial", "partialdiff", "subset", "neq", "limit", "merror",
    "matrix", "intersect", "equivalent",
    # medium-frequency
    "union", "ln", "compose", "ns2", "approx", "html", "setdiff", "exp",
    "tan", "emptyset", "not", "floor", "degree", "implies", "or",
    "exists", "notin", "determinant", "max", "gcd", "min", "mpadded",
    "arctan", "sec", "cot", "sinh", "cosh", "ceiling", "real", "mrow",
    "arcsin", "csc", "prsubset", "arccos", "tanh", "arg", "imaginary",
    # presentation MathML leak-through (rare but present)
    "mo", "mi", "mn",
]

TAG2ID = {tag: idx + 1 for idx, tag in enumerate(_TAGS)}
VOCAB_SIZE = len(_TAGS) + 1  # 97 total IDs (0 = unknown)

# Maximum tree nodes per formula; trees larger than this are truncated (BFS order)
MAX_NODES = 256


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix: {http://...}foo → foo."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def opt_to_pyg(opt_xml: str) -> Optional[Data]:
    """
    Parse a Content MathML string and return a PyG Data object, or None if
    the XML is malformed or yields fewer than 2 nodes.

    Returns
    -------
    Data with attributes:
        x          : LongTensor[num_nodes]       node type IDs
        edge_index : LongTensor[2, num_edges]    COO undirected edges
        num_nodes  : int
    """
    if not opt_xml:
        return None

    try:
        root = ET.fromstring(opt_xml)
    except ET.ParseError:
        return None

    node_ids: List[int] = []
    edges_src: List[int] = []
    edges_dst: List[int] = []
    pending: List[Tuple] = [(root, -1)]  # (element, parent_idx)

    while pending and len(node_ids) < MAX_NODES:
        elem, parent_idx = pending.pop(0)
        idx = len(node_ids)
        tag = _strip_ns(elem.tag)
        node_ids.append(TAG2ID.get(tag, 0))

        if parent_idx >= 0:
            edges_src.extend([parent_idx, idx])
            edges_dst.extend([idx, parent_idx])

        for child in elem:
            pending.append((child, idx))

    if len(node_ids) < 2:
        return None

    x = torch.tensor(node_ids, dtype=torch.long)
    edge_index = (
        torch.tensor([edges_src, edges_dst], dtype=torch.long)
        if edges_src
        else torch.zeros((2, 1), dtype=torch.long)  # self-loop fallback
    )
    return Data(x=x, edge_index=edge_index, num_nodes=len(node_ids))


def batch_opt_to_pyg(opt_list: List[Optional[str]]) -> Tuple[List[Data], List[bool]]:
    """Batch-convert OPT strings; returns (graphs, valid_mask)."""
    graphs: List[Data] = []
    mask: List[bool] = []
    for opt in opt_list:
        g = opt_to_pyg(opt)
        if g is not None:
            graphs.append(g)
            mask.append(True)
        else:
            mask.append(False)
    return graphs, mask
