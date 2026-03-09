"""
Build the FAISS retrieval index over all answer posts.

For each answer post in the corpus we compute a fused embedding by averaging
the CLIP embeddings that are available (text, images, formulas). All three
branches output L2-normalised vectors in CLIP space, so averaging and
re-normalising is the simplest valid fusion.

Inputs:
  data/processed/posts.jsonl          — parsed posts (from build_corpus.py)
  data/raw/arqmath/images/            — downloaded post images
  data/raw/arqmath/formulas/opt_*/    — OPT (Content MathML) formula index TSV

Outputs:
  data/processed/index/index.faiss    — FAISS inner-product index
  data/processed/index/metadata.jsonl — one record per indexed post

Usage:
  python -m src.retrieval.index.build_index
  python -m src.retrieval.index.build_index --formula-ckpt path/to/stage2.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.encoders import CLIPEncoder, FormulaEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
POSTS_JSONL  = REPO_ROOT / "data/processed/posts.jsonl"
IMAGES_DIR   = REPO_ROOT / "data/raw/arqmath/images"
OPT_DIR      = REPO_ROOT / "data/raw/arqmath/formulas/opt_representation_v3"
INDEX_DIR    = REPO_ROOT / "data/processed/index"

# Modality weights for fusion (must sum to 1 or be re-normalised after mean)
TEXT_WEIGHT    = 1.0
IMAGE_WEIGHT   = 1.0
FORMULA_WEIGHT = 1.0


# ── Formula cache ─────────────────────────────────────────────────────────────

def build_formula_cache(opt_dir: Path) -> dict[int, list[str]]:
    """
    Stream all OPT TSV files and return {post_id: [mathml_str, ...]}.

    The OPT index is ~9 GB unzipped; we stream it rather than load it whole.
    Only retains rows without issues (issue column is empty or missing).
    """
    cache: dict[int, list[str]] = {}
    tsv_files = sorted(opt_dir.glob("*.tsv"))

    if not tsv_files:
        print(f"Warning: no OPT TSV files found in {opt_dir}. Formulas will be skipped.")
        return cache

    for tsv in tqdm(tsv_files, desc="Loading OPT formula index"):
        with tsv.open(encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 9:
                    continue
                # Fields: id, post_id, thread_id, type, comment_id,
                #         old_visual_id, visual_id, issue, formula
                issue = row[7].strip()
                if issue in ("d", "dv"):   # formula missing from XML
                    continue
                try:
                    post_id = int(row[1])
                except ValueError:
                    continue
                mathml = row[8].strip().strip('"')
                if mathml:
                    cache.setdefault(post_id, []).append(mathml)

    return cache


# ── Fused embedding ────────────────────────────────────────────────────────────

def fuse(embeddings: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """
    Weighted mean of embeddings, then L2-normalise.
    All inputs must have the same dimension.
    """
    weighted = sum(w * e for w, e in zip(weights, embeddings))
    norm = np.linalg.norm(weighted)
    if norm < 1e-8:
        return weighted
    return weighted / norm


# ── Main ──────────────────────────────────────────────────────────────────────

def build_index(
    posts_jsonl: Path = POSTS_JSONL,
    images_dir: Path = IMAGES_DIR,
    opt_dir: Path = OPT_DIR,
    index_dir: Path = INDEX_DIR,
    formula_ckpt: Path | None = None,
    batch_size: int = 256,
    answers_only: bool = True,
) -> None:
    """
    Build and save the FAISS index.

    Args:
        posts_jsonl   — parsed posts JSONL
        images_dir    — root directory of downloaded images
        opt_dir       — directory of unzipped OPT TSV files
        index_dir     — where to write index.faiss + metadata.jsonl
        formula_ckpt  — path to trained formula encoder weights (Stage 2)
        batch_size    — posts per encoding batch
        answers_only  — if True, only index answer posts (PostType=2)
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load encoders
    clip = CLIPEncoder(device=device)
    dim = clip.dim
    formula_enc = FormulaEncoder(device=device, clip_dim=dim)
    if formula_ckpt and formula_ckpt.exists():
        print(f"Loading formula encoder from {formula_ckpt}")
        formula_enc.load(formula_ckpt)
    else:
        print("Warning: no formula encoder checkpoint — formula branch uses random weights.")

    # Load formula cache
    formula_cache = build_formula_cache(opt_dir)
    print(f"Formula cache: {len(formula_cache):,} posts with formulas")

    # FAISS index (inner product on L2-normalised vectors = cosine similarity)
    index = faiss.IndexFlatIP(dim)
    metadata: list[dict] = []

    # Process posts in batches
    pending_texts:    list[str]         = []
    pending_post_ids: list[int]         = []
    pending_posts:    list[dict]        = []

    def flush(posts: list[dict]) -> None:
        """Encode a batch of posts and add to FAISS."""
        if not posts:
            return

        texts    = [p["text"] or "" for p in posts]
        text_embs = clip.encode_text(texts, batch_size=len(texts), normalize=True).numpy()

        fused_embs = []
        for i, post in enumerate(posts):
            parts: list[np.ndarray] = []
            weights: list[float]    = []

            # Text
            parts.append(text_embs[i])
            weights.append(TEXT_WEIGHT)

            # Images
            image_dir = images_dir / str(post["post_id"])
            image_files = sorted(image_dir.glob("*")) if image_dir.exists() else []
            if image_files:
                img_embs = clip.encode_images(
                    image_files, batch_size=len(image_files), normalize=True
                ).numpy()
                parts.append(img_embs.mean(axis=0))
                weights.append(IMAGE_WEIGHT)

            # Formulas
            formulas = formula_cache.get(post["post_id"], [])
            if formulas:
                form_embs, _ = formula_enc.encode(formulas, normalize=True)
                if form_embs.shape[0] > 0:
                    parts.append(form_embs.mean(dim=0).numpy())
                    weights.append(FORMULA_WEIGHT)

            fused_embs.append(fuse(parts, weights))

        vectors = np.stack(fused_embs).astype(np.float32)
        index.add(vectors)

        for post in posts:
            metadata.append({
                "post_id":   post["post_id"],
                "post_type": post["post_type"],
                "score":     post["score"],
                "tags":      post["tags"],
                "title":     post.get("title"),
            })

    batch: list[dict] = []
    with posts_jsonl.open(encoding="utf-8") as f:
        for line in tqdm(f, desc="Indexing posts"):
            post = json.loads(line)
            if answers_only and post["post_type"] != "answer":
                continue
            batch.append(post)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
    flush(batch)

    # Save
    faiss.write_index(index, str(index_dir / "index.faiss"))
    with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for record in metadata:
            f.write(json.dumps(record) + "\n")

    print(f"\nIndex built: {index.ntotal:,} vectors ({dim}d)")
    print(f"Saved to: {index_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index")
    parser.add_argument("--formula-ckpt", type=Path, default=None)
    parser.add_argument("--batch-size",   type=int,  default=256)
    parser.add_argument("--all-posts",    action="store_true",
                        help="Index questions too, not just answers")
    args = parser.parse_args()

    build_index(
        formula_ckpt=args.formula_ckpt,
        batch_size=args.batch_size,
        answers_only=not args.all_posts,
    )
