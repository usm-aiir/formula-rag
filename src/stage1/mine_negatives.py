"""
Offline BM25 hard-negative mining for Task 2 training.

For each query topic in Years 1 & 2, this script:
  1. Builds a BM25 index over all *judged* candidate formulas (LaTeX strings).
  2. For each query (by LaTeX), retrieves the top-K BM25 candidates.
  3. Filters out known positives (grade ≥ POSITIVE_GRADE).
  4. Writes the result as JSON: {topic_id: [old_visual_id, ...]}

Mining once offline keeps the training loop clean — no BM25 overhead at
batch time, and the same negatives are reproducible across runs.

Usage
-----
    python -m src.stage1.mine_negatives \
        --out data/processed/task2_hard_negatives.json \
        --top-k 50 \
        --negatives 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.stage1.dataset import POSITIVE_GRADE, load_qrels, load_topics


def _tokenize(latex: str) -> List[str]:
    """Coarse LaTeX tokeniser for BM25 (token bags, not syntax trees)."""
    tokens = re.findall(r"\\[a-zA-Z]+|[0-9]+|[a-zA-Z]+|[^\s\w]", latex)
    return [t.lower() for t in tokens if t.strip()]


def mine(top_k: int = 50, n_negatives: int = 20, out_path: Path = None):
    qrels_train = load_qrels("train")
    topics_train = load_topics("train")

    # Collect all judged candidate old_visual_ids
    cand_id_set: set = set()
    for cands in qrels_train.values():
        cand_id_set.update(cands.keys())

    print(f"Total judged candidate formulas: {len(cand_id_set):,}", flush=True)

    # Load LaTeX for every judged candidate
    idx_dir = _PROJECT_ROOT / "data/processed/formula_index"
    shards = sorted(idx_dir.glob("*.parquet"))

    id_to_latex: Dict[str, str] = {}
    print("Loading LaTeX from formula index …", flush=True)
    for shard in tqdm(shards, desc="shards"):
        table = pq.read_table(shard, columns=["old_visual_id", "latex"])
        for ovid, latex in zip(table["old_visual_id"].to_pylist(),
                                table["latex"].to_pylist()):
            if ovid in cand_id_set and ovid not in id_to_latex and latex:
                id_to_latex[ovid] = latex

    valid_cand_ids = [cid for cid in cand_id_set if cid in id_to_latex]
    print(f"Candidates with LaTeX: {len(valid_cand_ids):,}", flush=True)

    # Build BM25 index
    print("Building BM25 index …", flush=True)
    corpus_tokens = [_tokenize(id_to_latex[cid]) for cid in valid_cand_ids]
    bm25 = BM25Okapi(corpus_tokens)

    # Mine negatives per topic
    results: Dict[str, List[str]] = {}
    print(f"Mining top-{top_k} BM25 negatives for {len(topics_train)} topics …", flush=True)

    for topic, latex in tqdm(topics_train.items(), desc="topics"):
        pos_ids = {
            cid for cid, g in qrels_train.get(topic, {}).items()
            if g >= POSITIVE_GRADE
        }
        scores = bm25.get_scores(_tokenize(latex))
        ranked = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)

        negatives: List[str] = []
        for idx, _score in ranked:
            cid = valid_cand_ids[idx]
            if cid not in pos_ids:
                negatives.append(cid)
            if len(negatives) >= n_negatives:
                break

        results[topic] = negatives

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"Saved hard negatives → {out_path}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Mine BM25 hard negatives for Task 2")
    parser.add_argument(
        "--out",
        type=Path,
        default=_PROJECT_ROOT / "data/processed/task2_hard_negatives.json",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=20)
    args = parser.parse_args()
    mine(top_k=args.top_k, n_negatives=args.negatives, out_path=args.out)


if __name__ == "__main__":
    main()
