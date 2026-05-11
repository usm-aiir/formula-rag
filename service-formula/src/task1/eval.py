"""
Evaluate MathBERTa text encoder on ARQMath Task 1 (Answer Retrieval), Year 3.

Pipeline:
  1. Encode full answer-post corpus
  2. Encode queries (Year 3 topics)
  3. FAISS retrieval, top-1000
  4. pytrec_eval for nDCG', MAP'

Usage:
  python -m src.task1.eval --checkpoint checkpoints/task1
  python -m src.task1.eval --checkpoint checkpoints/task1 --quick-run
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pytrec_eval
import torch
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.task1.data import iter_posts, load_qrels, load_topics
from src.task1.model import TextEncoder

_QUICK_RUN_CORPUS_SIZE = 50_000
_QUICK_RUN_SEED = 42


# ---------------------------------------------------------------------------
# Result caching
# ---------------------------------------------------------------------------

def _checkpoint_mtime(path: Path) -> float:
    """Mtime for cache invalidation — checks safetensors, pytorch_model.bin, or .pt."""
    if path.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin"):
            f = path / name
            if f.exists():
                return f.stat().st_mtime
        return 0.0
    return path.stat().st_mtime


def _cache_path(checkpoint_path: str, quick_run: bool) -> Path:
    p = Path(checkpoint_path)
    suffix = "_quick" if quick_run else ""
    if p.is_dir():
        return p / f"eval{suffix}.json"
    return p.parent / f"{p.stem}_eval{suffix}.json"


def _load_cache(checkpoint_path: str, quick_run: bool) -> Optional[dict]:
    cp = Path(checkpoint_path)
    if not cp.exists():
        return None
    cache = _cache_path(checkpoint_path, quick_run)
    if not cache.exists():
        return None
    try:
        with open(cache) as f:
            cached = json.load(f)
        if cached.get("checkpoint_mtime") != _checkpoint_mtime(cp):
            return None
        return cached
    except Exception:
        return None


def _save_cache(
    checkpoint_path: str,
    quick_run: bool,
    metrics: Dict[str, float],
    n_topics: int,
    corpus_size: int,
):
    cp = Path(checkpoint_path)
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(cp),
        "checkpoint_mtime": _checkpoint_mtime(cp),
        "quick_run": quick_run,
        "n_topics": n_topics,
        "corpus_size": corpus_size,
        "metrics": metrics,
    }
    with open(_cache_path(checkpoint_path, quick_run), "w") as f:
        json.dump(payload, f, indent=2)


def _print_metrics(metrics: Dict[str, float], n_topics: int, label: str):
    print(f"\n{'='*50}")
    print(f"{label} — {n_topics} topics")
    print(f"{'='*50}")
    for k in [10, 100, 1000]:
        ndcg = f"ndcg_cut_{k}"
        mapk = f"map_cut_{k}"
        if ndcg in metrics:
            print(f"  nDCG@{k:<5} {metrics[ndcg]:.4f}")
        if mapk in metrics:
            print(f"  MAP@{k:<6} {metrics[mapk]:.4f}")
    print(f"{'='*50}\n")


def _load_corpus(quick_run: bool = False) -> Tuple[List[str], List[str]]:
    """
    Stream all answer posts from posts.jsonl into (ids, texts).
    When quick_run=True, reservoir-sample _QUICK_RUN_CORPUS_SIZE posts so the
    run finishes quickly while still drawing from the real distribution.
    """
    if quick_run:
        rng = random.Random(_QUICK_RUN_SEED)
        reservoir_ids: List[str] = []
        reservoir_texts: List[str] = []
        for i, (pid, text) in enumerate(iter_posts(post_type="answer")):
            if i < _QUICK_RUN_CORPUS_SIZE:
                reservoir_ids.append(pid)
                reservoir_texts.append(text)
            else:
                j = rng.randint(0, i)
                if j < _QUICK_RUN_CORPUS_SIZE:
                    reservoir_ids[j] = pid
                    reservoir_texts[j] = text
        return reservoir_ids, reservoir_texts
    else:
        ids, texts = [], []
        for pid, text in iter_posts(post_type="answer"):
            ids.append(pid)
            texts.append(text)
        return ids, texts


def _encode_corpus(
    encoder: TextEncoder,
    corpus_ids: List[str],
    corpus_texts: List[str],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode corpus texts, return [N, D] float32 array."""
    encoder.eval()
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Encoding corpus"):
            batch = corpus_texts[i : i + batch_size]
            embs = encoder.encode(batch, device=device, batch_size=batch_size)
            all_embs.append(embs.numpy())
    return np.vstack(all_embs).astype(np.float32)


def _encode_queries(
    encoder: TextEncoder,
    topics: Dict[str, str],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Encode topic queries. Returns {topic_id: embedding}."""
    encoder.eval()
    topic_ids = list(topics.keys())
    texts = [topics[tid] for tid in topic_ids]
    embs = encoder.encode(texts, device=device, batch_size=32)
    return dict(zip(topic_ids, embs.numpy()))


def evaluate(
    checkpoint_path: str,
    top_k: int = 1000,
    batch_size: int = 64,
    quick_run: bool = False,
    force: bool = False,
    device_str: str = "cuda",
):
    label = "Task 1 Evaluation (quick-run — approximate)" if quick_run else "Task 1 Evaluation"

    if not force:
        cached = _load_cache(checkpoint_path, quick_run)
        if cached is not None:
            print(
                f"[cached] Loaded results from {_cache_path(checkpoint_path, quick_run)}\n"
                f"         Run at {cached['timestamp']} | corpus: {cached['corpus_size']:,} posts",
                flush=True,
            )
            _print_metrics(cached["metrics"], cached["n_topics"], label + " (cached)")
            return cached["metrics"]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    if quick_run:
        print(
            f"[quick-run] Reservoir-sampling {_QUICK_RUN_CORPUS_SIZE:,} posts "
            f"(seed={_QUICK_RUN_SEED}). Metrics are approximate.",
            flush=True,
        )

    encoder = TextEncoder.load(checkpoint_path, map_location=device).to(device)
    encoder.eval()

    qrels_eval = load_qrels("eval")
    topics_eval = load_topics("eval")

    corpus_ids, corpus_texts = _load_corpus(quick_run=quick_run)
    print(f"Corpus: {len(corpus_ids):,} posts", flush=True)

    corpus_embs = _encode_corpus(encoder, corpus_ids, corpus_texts, device, batch_size=batch_size)

    print("Building FAISS index …", flush=True)
    index = faiss.IndexFlatIP(corpus_embs.shape[1])
    index.add(corpus_embs)

    query_embs = _encode_queries(encoder, topics_eval, device)

    run: Dict[str, Dict[str, float]] = {}
    for topic_id, emb in query_embs.items():
        scores, indices = index.search(emb.reshape(1, -1).astype(np.float32), top_k)
        run[topic_id] = {
            corpus_ids[idx]: float(scores[0][j])
            for j, idx in enumerate(indices[0])
            if idx >= 0
        }

    qrels_int = {
        topic: {pid: int(g) for pid, g in cands.items()}
        for topic, cands in qrels_eval.items()
        if topic in run
    }

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_int,
        {"ndcg_cut", "map_cut"},
        relevance_level=1,
    )
    results = evaluator.evaluate(run)

    metrics: Dict[str, float] = defaultdict(float)
    n = len(results)
    for tr in results.values():
        for k, v in tr.items():
            metrics[k] += v
    metrics = {k: v / n for k, v in metrics.items()}

    _save_cache(checkpoint_path, quick_run, dict(metrics), n, len(corpus_ids))
    _print_metrics(metrics, n, label)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_PROJECT_ROOT / "checkpoints/task1"),
        help="Path to checkpoint (.pt file or HF-format directory)",
    )
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help=(
            f"Reservoir-sample {_QUICK_RUN_CORPUS_SIZE:,} corpus posts for a fast "
            "approximate run. Do not use for reporting results."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached results and re-run evaluation.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        batch_size=args.batch_size,
        quick_run=args.quick_run,
        force=args.force,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
