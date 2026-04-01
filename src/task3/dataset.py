"""
Task 2 dataset: formula-to-formula retrieval.

Loads ARQMath Task 2 qrels (Years 1 & 2 for training, Year 3 for eval) and
builds (query_opt, positive_opt) pairs for contrastive training with in-batch
negatives.

Formula lookup pipeline:
  - Candidate formulas: keyed by old_visual_id in the formula index.
  - Query formulas: extracted from topic XML by LaTeX match in the index.

The index is loaded lazily and cached for the lifetime of the process.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.data.formula_graph import opt_to_pyg

# ---------------------------------------------------------------------------
# Paths (resolved relative to project root)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FORMULA_INDEX_DIR = _PROJECT_ROOT / "data/processed/formula_index"

_QREL_PATHS: Dict[str, List[Path]] = {
    "train": [
        _PROJECT_ROOT / "data/raw/arqmath/qrels/task2/arqmath1/qrel_task2_2020_official.tsv",
        _PROJECT_ROOT / "data/raw/arqmath/qrels/task2/arqmath2/qrel_task2_2021_official.tsv",
    ],
    "eval": [
        _PROJECT_ROOT / "data/raw/arqmath/qrels/task2/arqmath3/qrel_task2_2022_official.tsv",
    ],
}

_TOPIC_PATHS: Dict[str, List[Path]] = {
    "train": [
        _PROJECT_ROOT / "data/raw/arqmath/topics/task2/arqmath1/Topics_V1.1.xml",
        _PROJECT_ROOT / "data/raw/arqmath/topics/task2/arqmath2/Topics_Task2_2021_V1.1.xml",
    ],
    "eval": [
        _PROJECT_ROOT / "data/raw/arqmath/topics/task2/arqmath3/Topics_Task2_2022_V1.2.xml",
    ],
}

POSITIVE_GRADE = 2.0

# ---------------------------------------------------------------------------
# Formula index — lazy global cache (old_visual_id → first non-null OPT string)
# ---------------------------------------------------------------------------

_OPT_INDEX: Optional[Dict[str, str]] = None


def load_opt_index() -> Dict[str, str]:
    """
    Load old_visual_id → OPT mapping from the sharded formula index.
    Only the first non-null OPT per old_visual_id is kept (formulas with the
    same visual_id appear in multiple posts but share the same OPT tree).
    Cached globally so it is only built once per process.
    """
    global _OPT_INDEX
    if _OPT_INDEX is not None:
        return _OPT_INDEX

    index: Dict[str, str] = {}
    shards = sorted(_FORMULA_INDEX_DIR.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No formula index shards found in {_FORMULA_INDEX_DIR}")

    print(f"Building OPT index from {len(shards)} shards …", flush=True)
    for shard in shards:
        table = pq.read_table(shard, columns=["old_visual_id", "opt"])
        for ovid, opt in zip(table["old_visual_id"].to_pylist(),
                              table["opt"].to_pylist()):
            if ovid and opt and ovid not in index:
                index[ovid] = opt

    _OPT_INDEX = index
    print(f"OPT index: {len(index):,} unique visual IDs with non-null OPT", flush=True)
    return index


def _latex_to_opt(latex: str) -> Optional[str]:
    """
    Look up OPT for a query formula by its LaTeX string (exact match).
    Only called for the ~200 topic query formulas, so full shard scan is fine.
    """
    shards = sorted(_FORMULA_INDEX_DIR.glob("*.parquet"))
    latex_strip = latex.strip()
    for shard in shards:
        table = pq.read_table(shard, columns=["latex", "opt"])
        for lat, opt in zip(table["latex"].to_pylist(), table["opt"].to_pylist()):
            if lat and lat.strip() == latex_strip and opt:
                return opt
    return None


# ---------------------------------------------------------------------------
# Topic / qrel loaders
# ---------------------------------------------------------------------------

def load_topics(split: str) -> Dict[str, str]:
    """
    Parse topic XML files for the given split.
    Returns {topic_number: latex_string}.
    """
    topics: Dict[str, str] = {}
    for path in _TOPIC_PATHS[split]:
        if not path.exists():
            raise FileNotFoundError(f"Topic file not found: {path}")
        tree = ET.parse(str(path))
        for topic in tree.getroot():
            num = topic.get("number", "")
            latex_elem = topic.find("Latex")
            if num and latex_elem is not None and latex_elem.text:
                topics[num] = latex_elem.text.strip()
    return topics


def load_qrels(split: str) -> Dict[str, Dict[str, float]]:
    """
    Load qrel files for the given split.
    Returns {topic_number: {old_visual_id: relevance_grade}}.
    """
    qrels: Dict[str, Dict[str, float]] = {}
    for path in _QREL_PATHS[split]:
        if not path.exists():
            raise FileNotFoundError(f"Qrel file not found: {path}")
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                topic, _, cand_id, grade = parts[0], parts[1], parts[2], parts[3]
                qrels.setdefault(topic, {})[cand_id] = float(grade)
    return qrels


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class FormulaRetrievalDataset(Dataset):
    """
    Each item is a dict with:
        query_graph : PyG Data   (query formula)
        pos_graph   : PyG Data   (a positive candidate, grade >= POSITIVE_GRADE)
        topic       : str        (for tracking / debugging)

    Pairs where the query or positive cannot be converted to a graph (null OPT,
    malformed XML) are silently skipped during __init__.
    """

    def __init__(self, split: str = "train", seed: int = 42):
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")

        rng = random.Random(seed)
        opt_index = load_opt_index()
        topics = load_topics(split)
        qrels = load_qrels(split)

        self.items: List[dict] = []

        print(f"Building {split} dataset …", flush=True)
        for topic, latex in topics.items():
            if topic not in qrels:
                continue

            query_opt = _latex_to_opt(latex)
            query_graph = opt_to_pyg(query_opt) if query_opt else None
            if query_graph is None:
                continue

            pos_ids = [cid for cid, g in qrels[topic].items() if g >= POSITIVE_GRADE]
            if not pos_ids:
                continue

            rng.shuffle(pos_ids)
            for pos_id in pos_ids:
                pos_opt = opt_index.get(pos_id)
                pos_graph = opt_to_pyg(pos_opt) if pos_opt else None
                if pos_graph is None:
                    continue

                self.items.append({
                    "query_graph": query_graph,
                    "pos_graph": pos_graph,
                    "topic": topic,
                })

        print(f"  {split}: {len(self.items):,} (query, positive) pairs", flush=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> dict:
    """
    Collates a list of items into batched PyG graphs.

    Returns a dict with:
        query_batch : PyG Batch
        pos_batch   : PyG Batch
    """
    query_graphs = [item["query_graph"] for item in batch]
    pos_graphs = [item["pos_graph"] for item in batch]

    return {
        "query_batch": Batch.from_data_list(query_graphs),
        "pos_batch": Batch.from_data_list(pos_graphs),
    }
