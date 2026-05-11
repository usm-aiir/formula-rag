#!/usr/bin/env python3
"""
eval_matvb.py — Evaluate LongCLIP image retrieval on the MATVB benchmark.

Encodes all images in the MATVB image collection with
longclip_handler.encode_image(), encodes each query with
longclip_handler.encode_text(), computes cosine similarity (embeddings are
already L2-normalised), and reports ranx metrics against the MATVB qrel.

Both the natural-language query split and the caption query split are
evaluated by default.

Usage:
    python eval_matvb.py                           # all defaults
    python eval_matvb.py --split nl                # only natural-language
    python eval_matvb.py \\
        --image-dir /path/to/MATVB/images \\
        --query-nl  /path/to/MATVB_query_nl.json \\
        --query-cap /path/to/MATVB_query_caption.json \\
        --qrel      /path/to/MATVB_qrel.qrel \\
        --top-k 100

Image IDs are derived from filenames by stripping the extension, which
matches the identifiers used in MATVB_qrel.qrel (e.g. "Unit_root_1").

MATVB image collection must be downloaded separately — see MATVB/data/README.md.

Requires: ranx, numpy, Pillow, torch  (pip install ranx)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ranx import Qrels, Run, evaluate

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import longclip_handler as lch

DEFAULT_IMAGE_DIR = _HERE.parent.parent / "MATVB" / "images"
DEFAULT_QUERY_NL  = _HERE.parent.parent / "MATVB" / "data" / "MATVB_query_nl.json"
DEFAULT_QUERY_CAP = _HERE.parent.parent / "MATVB" / "data" / "MATVB_query_caption.json"
DEFAULT_QREL      = _HERE.parent.parent / "MATVB" / "data" / "MATVB_qrel.qrel"
DEFAULT_METRICS   = ["mrr", "recall@5", "recall@10", "ndcg@100"]

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg"}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """Load a TREC-format qrel file into {qid: {doc_id: relevance}}.

    Entries with relevance grade 0 are excluded.
    """
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, doc_id, rel = parts[0], parts[2], int(parts[3])
            if rel > 0:
                qrels[qid][doc_id] = rel
    return dict(qrels)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_image_collection(
    image_dir: Path,
) -> Tuple[np.ndarray, List[str]]:
    """Encode every image in image_dir; return (embeddings [N, D], image_ids).

    image_id = filename stem (no extension), matching MATVB qrel doc IDs.
    Files that cannot be encoded are skipped with a warning.
    """
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(
            f"No image files found in {image_dir}.\n"
            "Download the MATVB image collection — see MATVB/data/README.md."
        )

    embeddings: List[np.ndarray] = []
    image_ids:  List[str]        = []
    total = len(paths)

    for i, path in enumerate(paths, 1):
        print(f"\r  Encoding image [{i:>5}/{total}]  {path.name:<40}", end="", flush=True)
        try:
            vec = lch.encode_image(path)   # shape (1, D), already L2-normalised
            embeddings.append(vec[0])
            image_ids.append(path.stem)
        except Exception as exc:
            print(f"\n  WARNING: skipping {path.name} — {exc}")

    print()
    if not embeddings:
        raise RuntimeError("All images failed to encode.")

    return np.stack(embeddings, axis=0), image_ids


def encode_text_queries(
    query_dict: Dict[str, str],
) -> Tuple[np.ndarray, List[str]]:
    """Encode all text queries; return (embeddings [Q, D], ordered qids)."""
    qids:  List[str]        = list(query_dict.keys())
    texts: List[str]        = [query_dict[q] for q in qids]
    embeddings: List[np.ndarray] = []
    total = len(texts)

    for i, text in enumerate(texts, 1):
        print(f"\r  Encoding query [{i:>4}/{total}]", end="", flush=True)
        vec = lch.encode_text(text)   # shape (1, D), already L2-normalised
        embeddings.append(vec[0])

    print()
    return np.stack(embeddings, axis=0), qids


# ---------------------------------------------------------------------------
# Run construction
# ---------------------------------------------------------------------------

def build_run(
    text_embeds: np.ndarray,
    qids: List[str],
    img_embeds: np.ndarray,
    image_ids: List[str],
    top_k: int,
) -> Dict[str, Dict[str, float]]:
    """Compute cosine similarity scores and return a ranx-compatible run dict.

    Because both text_embeds and img_embeds are L2-normalised, the dot
    product equals cosine similarity.
    """
    # scores[q, d] = cosine(query_q, image_d)
    scores = text_embeds @ img_embeds.T          # [Q, N]
    run: Dict[str, Dict[str, float]] = {}

    for qi, qid in enumerate(qids):
        sims       = scores[qi]
        ranked_idx = np.argsort(sims)[::-1][:top_k]
        run[qid]   = {image_ids[int(idx)]: float(sims[idx]) for idx in ranked_idx}

    return run


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------

def evaluate_split(
    split_name: str,
    query_dict: Dict[str, str],
    qrels_dict: Dict[str, Dict[str, int]],
    img_embeds: np.ndarray,
    image_ids: List[str],
    top_k: int,
    metrics: List[str],
) -> None:
    print(f"\n--- {split_name} ({len(query_dict)} queries) ---")
    print("Encoding queries ...")
    text_embeds, qids = encode_text_queries(query_dict)

    overlap = [q for q in qids if q in qrels_dict]
    if not overlap:
        print(
            "  WARNING: no query IDs overlap with the qrel.\n"
            f"  Query IDs  (first 5): {qids[:5]}\n"
            f"  Qrel IDs   (first 5): {list(qrels_dict.keys())[:5]}"
        )
        return

    print(f"  {len(overlap)} of {len(qids)} queries have qrel entries")

    run_dict = build_run(text_embeds, qids, img_embeds, image_ids, top_k)
    qrels    = Qrels(qrels_dict)
    run      = Run(run_dict, name=split_name)
    results  = evaluate(qrels, run, metrics)

    for metric, score in results.items():
        print(f"  {metric:<22s}  {score:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image-dir", default=str(DEFAULT_IMAGE_DIR),
        help=f"Directory containing MATVB images (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--query-nl", default=str(DEFAULT_QUERY_NL),
        help=f"MATVB_query_nl.json (default: {DEFAULT_QUERY_NL})",
    )
    parser.add_argument(
        "--query-cap", default=str(DEFAULT_QUERY_CAP),
        help=f"MATVB_query_caption.json (default: {DEFAULT_QUERY_CAP})",
    )
    parser.add_argument(
        "--qrel", default=str(DEFAULT_QREL),
        help=f"MATVB_qrel.qrel in TREC format (default: {DEFAULT_QREL})",
    )
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="Ranking depth for retrieval (default: 100)",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        help="ranx metric names to compute (default: ndcg@10 ndcg@100 map precision@10)",
    )
    parser.add_argument(
        "--split", choices=["nl", "caption", "both"], default="both",
        help="Which query split to evaluate (default: both)",
    )
    parser.add_argument(
        "--base", action="store_true",
        help="Force use of the base (non-fine-tuned) checkpoint even if a fine-tuned one exists",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    qrel_path = Path(args.qrel)
    query_nl  = Path(args.query_nl)
    query_cap = Path(args.query_cap)

    if not image_dir.is_dir():
        parser.error(
            f"--image-dir not found: {image_dir}\n"
            "Download the MATVB image collection — see MATVB/data/README.md."
        )
    if not qrel_path.is_file():
        parser.error(f"--qrel file not found: {qrel_path}")

    # --- Load qrels ---
    print(f"Loading qrels from:   {qrel_path}")
    qrels_dict = load_qrels(qrel_path)
    print(f"  {len(qrels_dict)} queries have relevance judgements")

    # --- Load model (optionally force base checkpoint) ---
    if args.base:
        print("\nForcing base (non-fine-tuned) checkpoint ...")
        lch.load_base_model()

    # --- Encode image collection once (shared across both splits) ---
    print(f"\nEncoding image collection from {image_dir} ...")
    img_embeds, image_ids = encode_image_collection(image_dir)
    print(f"  {len(image_ids)} images encoded  |  embedding dim: {img_embeds.shape[1]}")

    # --- Natural-language queries ---
    if args.split in ("nl", "both"):
        if not query_nl.is_file():
            print(f"WARNING: --query-nl file not found: {query_nl}  (skipping)")
        else:
            with open(query_nl, encoding="utf-8") as fh:
                query_nl_dict: Dict[str, str] = json.load(fh)
            evaluate_split(
                "Natural Language Queries",
                query_nl_dict, qrels_dict,
                img_embeds, image_ids,
                args.top_k, args.metrics,
            )

    # --- Caption queries ---
    if args.split in ("caption", "both"):
        if not query_cap.is_file():
            print(f"WARNING: --query-cap file not found: {query_cap}  (skipping)")
        else:
            with open(query_cap, encoding="utf-8") as fh:
                query_cap_dict: Dict[str, str] = json.load(fh)
            evaluate_split(
                "Caption Queries",
                query_cap_dict, qrels_dict,
                img_embeds, image_ids,
                args.top_k, args.metrics,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
