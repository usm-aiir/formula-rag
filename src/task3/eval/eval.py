"""
ARQMath Task 2 evaluation: formula-to-formula retrieval on ARQMath-3 (Year 3).
NOTE: This eval script was designed for the Phase 2 OPT-based encoder. For the dual-encoder, use src/task3/eval_dual.py instead.

Pipeline
--------
1. Encode the full formula corpus (all unique old_visual_ids in the formula index).
2. For each topic query (Year 3), encode the query formula.
3. Retrieve top-1000 by cosine similarity (inner product on L2-normalised vecs).
4. Score with pytrec_eval using the Year-3 official qrels.
5. Report nDCG′@10, nDCG′@1000, MAP′ (ARQMath-style prime metrics).

Usage
-----
    python -m src.task3.eval --checkpoint checkpoints/task3/best.pt
    python -m src.task3.eval --checkpoint checkpoints/task3/best.pt --quick-run
"""

from __future__ import annotations

import argparse
import random
import re
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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.task3.dataset import load_qrels, load_topics
from src.data.formula_graph import opt_to_pyg
from src.task3.model.formula_encoder import FormulaEncoder

_FORMULA_INDEX_DIR = _PROJECT_ROOT / "data/processed/formula_index"
_EVAL_SPLIT = "eval"
_QUICK_RUN_CORPUS_SIZE = 100_000
_QUICK_RUN_SEED = 42


# ---------------------------------------------------------------------------
# Corpus encoding
# ---------------------------------------------------------------------------

def _encode_corpus(
    encoder: FormulaEncoder,
    device: torch.device,
    batch_size: int = 512,
    quick_run: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode the full formula corpus (one entry per unique old_visual_id).
    When quick_run=True, reservoir-sample _QUICK_RUN_CORPUS_SIZE formulas.

    Returns
    -------
    embeddings : np.ndarray[N, D]   float32, L2-normalised
    id_list    : List[str]          old_visual_id for each row
    """
    shards = sorted(_FORMULA_INDEX_DIR.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No formula index shards in {_FORMULA_INDEX_DIR}")

    seen: set = set()
    all_entries: List[Tuple[str, str]] = []

    for shard in tqdm(shards, desc="Loading corpus OPTs"):
        table = pq.read_table(shard, columns=["old_visual_id", "opt"])
        for ovid, opt in zip(table["old_visual_id"].to_pylist(), table["opt"].to_pylist()):
            if ovid and opt and ovid not in seen:
                all_entries.append((ovid, opt))
                seen.add(ovid)

    if quick_run:
        rng = random.Random(_QUICK_RUN_SEED)
        if len(all_entries) > _QUICK_RUN_CORPUS_SIZE:
            all_entries = rng.sample(all_entries, _QUICK_RUN_CORPUS_SIZE)

    print(f"Corpus size: {len(all_entries):,} unique formulas", flush=True)

    id_list: List[str] = []
    all_embs: List[np.ndarray] = []
    batch_ids: List[str] = []
    batch_opts: List[str] = []

    encoder.eval()

    def _flush():
        if not batch_opts:
            return
        graphs = [opt_to_pyg(o) for o in batch_opts]
        valid = [(g, bid) for g, bid in zip(graphs, batch_ids) if g is not None]
        if not valid:
            return
        valid_graphs, valid_ids = zip(*valid)
        pyg_batch = Batch.from_data_list(list(valid_graphs)).to(device)
        with torch.no_grad():
            embs = encoder(pyg_batch, normalize=True)
        all_embs.append(embs.cpu().float().numpy())
        id_list.extend(valid_ids)

    for ovid, opt in tqdm(all_entries, desc="Encoding corpus"):
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

def normalize_latex(s: str) -> str:
    """Safe normalization: whitespace, sizing, and basic synonyms only."""
    if not s:
        return ""
    s = re.sub(r'\s+', '', s)
    s = s.replace(r'\left', '').replace(r'\right', '')
    synonyms = {
        r'\leq': r'\le', 
        r'\geq': r'\ge', 
        r'\rightarrow': r'\to', 
        r'\gets': r'\leftarrow', 
        r'\ne': r'\neq'
    }
    for old, new in synonyms.items():
        s = s.replace(old, new)
    return s

def _encode_queries(
    encoder: FormulaEncoder,
    device: torch.device,
) -> Dict[str, Optional[np.ndarray]]:
    """Encode each eval topic by its LaTeX query formula."""
    topics = load_topics(_EVAL_SPLIT)
    shards = sorted(_FORMULA_INDEX_DIR.glob("*.parquet"))

    unique_queries = {latex.strip() for latex in topics.values()}
    norm_to_orig = {normalize_latex(q): q for q in unique_queries}
    normalized_query_set = set(norm_to_orig.keys())
    
    query_opt_map = {}
    print(f"Scanning shards for {len(unique_queries)} normalized queries...", flush=True)
    
    for shard in shards:
        # Early exit if we found them all
        if len(query_opt_map) == len(unique_queries):
            break 
            
        table = pq.read_table(shard, columns=["latex", "opt"])
        for lat, opt in zip(table["latex"].to_pylist(), table["opt"].to_pylist()):
            if lat:
                norm_lat = normalize_latex(lat)
                if norm_lat in normalized_query_set and opt:
                    orig_query = norm_to_orig[norm_lat]
                    if orig_query not in query_opt_map:
                        query_opt_map[orig_query] = opt

    query_embs: Dict[str, Optional[np.ndarray]] = {}
    encoder.eval()

    for topic, latex in tqdm(topics.items(), desc="Encoding queries"):
        opt = query_opt_map.get(latex.strip())
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
    quick_run: bool = False,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    if quick_run:
        print(
            f"[quick-run] Sampling {_QUICK_RUN_CORPUS_SIZE:,} formulas "
            f"(seed={_QUICK_RUN_SEED}). Metrics are approximate.",
            flush=True,
        )

    encoder = FormulaEncoder.load(checkpoint_path, map_location=device).to(device)
    encoder.eval()

    qrels_eval = load_qrels(_EVAL_SPLIT)

    corpus_embs, corpus_ids = _encode_corpus(encoder, device, batch_size=batch_size, quick_run=quick_run)

    print("Building FAISS index …", flush=True)
    faiss_index = _build_faiss_index(corpus_embs)

    query_embs = _encode_queries(encoder, device)

    run: Dict[str, Dict[str, float]] = {
        topic: dict(_retrieve(faiss_index, corpus_ids, emb, top_k))
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
            {"ndcg_cut", "map_cut", "P", "bpref"}, # Added Precision and bpref
            relevance_level=2,
        )
    results = evaluator.evaluate(run)

    metrics_agg: Dict[str, float] = defaultdict(float)
    n = len(results)
    for topic_results in results.values():
        for metric, value in topic_results.items():
            metrics_agg[metric] += value
    metrics_agg = {k: v / n for k, v in metrics_agg.items()}

    label = "Task 3 Evaluation (quick-run — approximate)" if quick_run else "Task 3 Evaluation"
    print(f"\n{'='*50}")
    print(f"{label} — {n} topics")
    print(f"{'='*50}")
    
    # bpref is calculated over the entire retrieved list, so it has no @k cutoff
    if "bpref" in metrics_agg:
        print(f"  bpref      {metrics_agg['bpref']:.4f}")
        print(f"{'-'*30}")

    for k in [5, 10, 100, 1000]:
        ndcg_key = f"ndcg_cut_{k}"
        map_key = f"map_cut_{k}"
        p_key = f"P_{k}"
        
        if ndcg_key in metrics_agg:
            print(f"  nDCG@{k:<5} {metrics_agg[ndcg_key]:.4f}")
        if map_key in metrics_agg:
            print(f"  MAP@{k:<6} {metrics_agg[map_key]:.4f}")
        if p_key in metrics_agg:
            print(f"  P@{k:<8} {metrics_agg[p_key]:.4f}")
        
        if k != 1000:
            print(f"{'-'*30}")
            
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
        default=str(_PROJECT_ROOT / "checkpoints/task3/best.pt"),
    )
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help=(
            f"Randomly sample {_QUICK_RUN_CORPUS_SIZE:,} corpus formulas for a fast "
            "approximate run. Do not use for reporting results."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        batch_size=args.batch_size,
        quick_run=args.quick_run,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()