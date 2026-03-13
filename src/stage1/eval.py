"""
Task 2 evaluation: formula-to-formula retrieval on ARQMath-3 (Year 3).

Pipeline
--------
1. Encode all *judged* corpus formulas with the GNN → FAISS index.
2. For each topic query (Year 3), encode the query formula.
3. Retrieve top-1000 by cosine similarity (inner product on L2-normalised vecs).
4. Score with pytrec_eval using the Year-3 official qrels.
5. Report nDCG′@10, nDCG′@1000, MAP′ (ARQMath-style prime metrics).

Usage
-----
    python -m src.stage1.eval \
        --checkpoint checkpoints/gnn_stage1/best.pt \
        [--top-k 1000] \
        [--batch-size 512] \
        [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pyarrow.parquet as pq
import pytrec_eval
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.stage1.dataset import load_qrels, load_topics
from src.data.formula_graph import opt_to_pyg
from src.stage1.model.formula_encoder import FormulaEncoder

_FORMULA_INDEX_DIR = _PROJECT_ROOT / "data/processed/formula_index"
_EVAL_SPLIT = "eval"


# ---------------------------------------------------------------------------
# Corpus encoding
# ---------------------------------------------------------------------------

def _encode_corpus(
    encoder: FormulaEncoder,
    device: torch.device,
    batch_size: int = 512,
    judged_ids: Optional[set] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode all judged corpus formulas (old_visual_id → first non-null OPT).

    Returns
    -------
    embeddings : np.ndarray[N, D]   float32, L2-normalised
    id_list    : List[str]          old_visual_id for each row
    """
    shards = sorted(_FORMULA_INDEX_DIR.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No formula index shards in {_FORMULA_INDEX_DIR}")

    seen: set = set()
    corpus: List[Tuple[str, str]] = []

    for shard in tqdm(shards, desc="Loading corpus OPTs"):
        table = pq.read_table(shard, columns=["old_visual_id", "opt"])
        for ovid, opt in zip(table["old_visual_id"].to_pylist(),
                              table["opt"].to_pylist()):
            if ovid and opt and ovid not in seen:
                if judged_ids is None or ovid in judged_ids:
                    corpus.append((ovid, opt))
                    seen.add(ovid)

    print(f"Corpus size: {len(corpus):,} unique formulas", flush=True)

    id_list: List[str] = []
    all_embs: List[np.ndarray] = []
    batch_ids: List[str] = []
    batch_opts: List[str] = []

    encoder.eval()

    def _flush():
        if not batch_opts:
            return
        graphs = [opt_to_pyg(o) for o in batch_opts]
        valid_graphs = [g for g in graphs if g is not None]
        valid_ids = [bid for g, bid in zip(graphs, batch_ids) if g is not None]
        if not valid_graphs:
            return
        pyg_batch = Batch.from_data_list(valid_graphs).to(device)
        with torch.no_grad():
            embs = encoder(pyg_batch, normalize=True)
        all_embs.append(embs.cpu().float().numpy())
        id_list.extend(valid_ids)

    for ovid, opt in tqdm(corpus, desc="Encoding corpus"):
        batch_ids.append(ovid)
        batch_opts.append(opt)
        if len(batch_opts) >= batch_size:
            _flush()
            batch_ids, batch_opts = [], []
    _flush()

    return np.vstack(all_embs).astype(np.float32), id_list


# ---------------------------------------------------------------------------
# Query encoding
# ---------------------------------------------------------------------------

def _encode_queries(
    encoder: FormulaEncoder,
    device: torch.device,
) -> Dict[str, Optional[np.ndarray]]:
    """Encode each eval topic by its LaTeX query formula."""
    topics = load_topics(_EVAL_SPLIT)
    shards = sorted(_FORMULA_INDEX_DIR.glob("*.parquet"))

    def _latex_to_opt(latex: str) -> Optional[str]:
        latex_strip = latex.strip()
        for shard in shards:
            table = pq.read_table(shard, columns=["latex", "opt"])
            for lat, opt in zip(table["latex"].to_pylist(), table["opt"].to_pylist()):
                if lat and lat.strip() == latex_strip and opt:
                    return opt
        return None

    query_embs: Dict[str, Optional[np.ndarray]] = {}
    encoder.eval()

    for topic, latex in tqdm(topics.items(), desc="Encoding queries"):
        opt = _latex_to_opt(latex)
        if opt is None:
            query_embs[topic] = None
            continue
        graph = opt_to_pyg(opt)
        if graph is None:
            query_embs[topic] = None
            continue
        batch = Batch.from_data_list([graph]).to(device)
        with torch.no_grad():
            emb = encoder(batch, normalize=True)[0].cpu().float().numpy()
        query_embs[topic] = emb

    resolved = sum(1 for v in query_embs.values() if v is not None)
    print(f"Resolved {resolved}/{len(topics)} query topics", flush=True)
    return query_embs


# ---------------------------------------------------------------------------
# FAISS retrieval
# ---------------------------------------------------------------------------

def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Inner-product index over L2-normalised vectors = cosine similarity."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def _retrieve(
    index: faiss.IndexFlatIP,
    id_list: List[str],
    query_emb: np.ndarray,
    top_k: int = 1000,
) -> List[Tuple[str, float]]:
    scores, indices = index.search(query_emb.reshape(1, -1), top_k)
    return [
        (id_list[idx], float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx >= 0
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str,
    top_k: int = 1000,
    batch_size: int = 512,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    encoder = FormulaEncoder.load(checkpoint_path, map_location=device).to(device)
    encoder.eval()

    qrels_eval = load_qrels(_EVAL_SPLIT)
    judged_ids = {cid for cands in qrels_eval.values() for cid in cands}

    corpus_embs, corpus_ids = _encode_corpus(
        encoder, device, batch_size=batch_size, judged_ids=judged_ids
    )

    print("Building FAISS index …", flush=True)
    faiss_index = _build_faiss_index(corpus_embs)

    query_embs = _encode_queries(encoder, device)

    run: Dict[str, Dict[str, float]] = {
        topic: {doc_id: score for doc_id, score in _retrieve(faiss_index, corpus_ids, emb, top_k)}
        for topic, emb in query_embs.items()
        if emb is not None
    }

    qrels_int: Dict[str, Dict[str, int]] = {
        topic: {cid: int(g) for cid, g in cands.items()}
        for topic, cands in qrels_eval.items()
        if topic in run
    }

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_int,
        {"ndcg_cut", "map_cut"},
        relevance_level=2,
    )
    results = evaluator.evaluate(run)

    metrics_agg: Dict[str, float] = defaultdict(float)
    n = len(results)
    for topic_results in results.values():
        for metric, value in topic_results.items():
            metrics_agg[metric] += value
    metrics_agg = {k: v / n for k, v in metrics_agg.items()}

    print(f"\n{'='*50}")
    print(f"Task 2 Evaluation — {n} topics")
    print(f"{'='*50}")
    for k in [10, 100, 1000]:
        ndcg_key = f"ndcg_cut_{k}"
        map_key = f"map_cut_{k}"
        if ndcg_key in metrics_agg:
            print(f"  nDCG@{k:<5} {metrics_agg[ndcg_key]:.4f}")
        if map_key in metrics_agg:
            print(f"  MAP@{k:<6} {metrics_agg[map_key]:.4f}")
    print(f"{'='*50}\n")

    return metrics_agg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate GNN formula encoder on Task 2")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_PROJECT_ROOT / "checkpoints/gnn_stage1/best.pt"),
    )
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
