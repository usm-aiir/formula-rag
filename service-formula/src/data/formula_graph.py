"""
Convert Content MathML (OPT) and Presentation MathML (SLT) strings 
into PyTorch Geometric Data objects.

Each XML element becomes a node; parent-->child relationships become directed
edges (we add reverse edges to make the graph undirected for message passing).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Data

# ===========================================================================
# OPT (Semantic) Vocabulary
# ===========================================================================

_OPT_TAGS: List[str] = [
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

OPT_TAG2ID = {tag: idx + 1 for idx, tag in enumerate(_OPT_TAGS)}
OPT_VOCAB_SIZE = len(_OPT_TAGS) + 1  # (0 = unknown)

# Maintain backward compatibility for Phase 1 code
VOCAB_SIZE = OPT_VOCAB_SIZE 

# ===========================================================================
# 2. SLT (Visual Layout) Vocabulary
# ===========================================================================

_SLT_TAGS: List[str] = [
    # Core Presentation MathML Tags
    "math", "mrow", "mi", "mo", "mn", "mfrac", "msup", "msub", 
    "msubsup", "msqrt", "mroot", "mfenced", "mtable", "mtr", 
    "mtd", "munderover", "mover", "munder", "mspace", "mphantom", 
    "mtext", "mstyle", "maligngroup", "malignmark", "menclose", "maction"
]

SLT_TAG2ID = {tag: idx + 1 for idx, tag in enumerate(_SLT_TAGS)}
SLT_VOCAB_SIZE = len(_SLT_TAGS) + 1  # (0 = unknown)


# Maximum tree nodes per formula
MAX_NODES = 256


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix: {http://...}foo → foo."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


# ===========================================================================
# Parsers
# ===========================================================================

def _xml_to_pyg(xml_str: str, tag2id: dict) -> Optional[Data]:
    """Internal base parser for both modalities."""
    if not xml_str:
        return None

    try:
        root = ET.fromstring(xml_str)
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
        
        # Use the specific vocabulary (OPT or SLT)
        node_ids.append(tag2id.get(tag, 0))

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
        else torch.zeros((2, 1), dtype=torch.long)
    )
    return Data(x=x, edge_index=edge_index, num_nodes=len(node_ids))


def opt_to_pyg(opt_xml: str) -> Optional[Data]:
    """Parse Content MathML (OPT) → PyG Data"""
    return _xml_to_pyg(opt_xml, OPT_TAG2ID)

def slt_to_pyg(slt_xml: str) -> Optional[Data]:
    """Parse Presentation MathML (SLT) → PyG Data"""
    return _xml_to_pyg(slt_xml, SLT_TAG2ID)


def batch_opt_to_pyg(opt_list: List[Optional[str]]) -> Tuple[List[Data], List[bool]]:
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

def batch_slt_to_pyg(slt_list: List[Optional[str]]) -> Tuple[List[Data], List[bool]]:
    graphs: List[Data] = []
    mask: List[bool] = []
    for slt in slt_list:
        g = slt_to_pyg(slt)
        if g is not None:
            graphs.append(g)
            mask.append(True)
        else:
            mask.append(False)
    return graphs, mask