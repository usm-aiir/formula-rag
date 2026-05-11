"""
PyTorch Dataset for Task 1 contrastive training.

Each item is (query_text, positive_text) where positive is an answer post with
relevance >= POSITIVE_GRADE. Uses in-batch negatives only (standard ARQMath practice).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset

from src.task1.data import POSITIVE_GRADE, load_post_texts, load_qrels, load_topics

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_task2_pairs() -> List[tuple]:
    """Load (query_latex, positive_latex) from Task 2 for joint training."""
    import xml.etree.ElementTree as ET
    import pyarrow.parquet as pq

    t2_qrels_dir = _PROJECT_ROOT / "data/raw/arqmath/qrels/task2"
    judged_ovids = set()
    qrels_by_topic: Dict[str, Dict[str, float]] = {}
    for year in ["arqmath1", "arqmath2"]:
        for p in (t2_qrels_dir / year).glob("qrel*.tsv"):
            with open(p) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        tid, _, ovid, grade = parts[0], parts[1], parts[2], float(parts[3])
                        judged_ovids.add(ovid)
                        qrels_by_topic.setdefault(tid, {})[ovid] = grade

    idx_dir = _PROJECT_ROOT / "data/processed/formula_index"
    ovid_to_latex: Dict[str, str] = {}
    for shard in sorted(idx_dir.glob("*.parquet")):
        t = pq.read_table(shard, columns=["old_visual_id", "latex"])
        for ovid, lat in zip(t["old_visual_id"].to_pylist(), t["latex"].to_pylist()):
            if ovid in judged_ovids and ovid not in ovid_to_latex and lat:
                ovid_to_latex[ovid] = lat

    pairs = []
    for year in ["arqmath1", "arqmath2"]:
        topic_files = list((_PROJECT_ROOT / "data/raw/arqmath/topics/task2" / year).glob("*.xml"))
        if not topic_files:
            continue
        tree = ET.parse(topic_files[0])
        for topic in tree.getroot():
            tid = topic.get("number")
            latexe = topic.find("Latex")
            if not tid or latexe is None or latexe.text is None:
                continue
            qlatex = latexe.text.strip()
            if tid not in qrels_by_topic:
                continue
            for ovid, g in qrels_by_topic[tid].items():
                if g >= 2.0 and ovid in ovid_to_latex:
                    platex = ovid_to_latex[ovid]
                    if qlatex and platex:
                        pairs.append((f"[MATH]{qlatex}[/MATH]", f"[MATH]{platex}[/MATH]"))
    return pairs


class Task1Dataset(Dataset):
    """
    Yields (query, positive) for contrastive training with in-batch negatives.
    """

    def __init__(
        self,
        split: str = "train",
        include_task2: bool = True,
        seed: int = 42,
    ):
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")

        self.split = split
        rng = random.Random(seed)

        topics = load_topics(split)
        qrels = load_qrels(split)

        pairs: List[tuple] = []
        for topic_id, query_text in topics.items():
            if topic_id not in qrels:
                continue
            pos_ids = [pid for pid, g in qrels[topic_id].items() if g >= POSITIVE_GRADE]
            if not pos_ids:
                continue
            for pid in pos_ids:
                pairs.append((topic_id, query_text, pid))

        post_ids = {pid for _, _, pid in pairs}
        self.post_texts = load_post_texts(post_ids)

        self.items: List[tuple] = []
        for topic_id, query_text, pid in pairs:
            pos_text = self.post_texts.get(pid)
            if not pos_text:
                continue
            self.items.append((query_text, pos_text))

        if include_task2 and split == "train":
            t2_pairs = _load_task2_pairs()
            for q, p in t2_pairs:
                self.items.append((q, p))
            print(f"  + {len(t2_pairs):,} Task 2 formula pairs", flush=True)

        rng.shuffle(self.items)
        print(f"Task1 {split}: {len(self.items):,} (query, positive) pairs", flush=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        q, p = self.items[idx]
        return {"query": q, "positive": p}
