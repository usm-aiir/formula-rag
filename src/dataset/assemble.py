"""
Assemble the trimodal ARQMath dataset from its components.

Joins posts (body text, metadata) with:
  - Formula representations (formula_id, LaTeX) from the formula index
  - Rendered formula image paths (typeset + graph) from rendered_formulas/
  - Downloaded post image paths from data/raw/arqmath/images/

The formula index is loaded into memory using Polars (only the lightweight
formula_id + latex columns — OPT/SLT remain in the formula index shards as
a separate joinable artifact).  Memory usage is approximately 1-2 GB.

Output
------
    data/processed/dataset/shard_NNNN.parquet   (100 K posts per shard)
    data/processed/dataset/dataset_info.json

Schema per row (one row = one post)
------------------------------------
post_id             int64
post_type           string    "question" | "answer"
parent_id           int64     null for questions
accepted_answer_id  int64     null for answers / unanswered questions
title               string    null for answers
tags                list<str> null for answers
score               int64
creation_date       string    ISO-8601
body_text           string    HTML-stripped; formula positions as [FORMULA_i]
formulas            list<struct{
                        formula_id      string
                        latex           string
                        rendered_typeset string | null  (relative path)
                        rendered_graph   string | null  (relative path)
                    }>
post_images         list<str>  relative paths under data/raw/arqmath/images/
has_formulas        bool
has_post_images     bool
has_rendered_formulas bool

Usage
-----
    python scripts/dataset/03_assemble_dataset.py [--force]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT         = Path(__file__).resolve().parents[2]
FORMULA_INDEX_DIR = REPO_ROOT / "data/processed/formula_index"
RENDERED_DIR      = REPO_ROOT / "data/processed/rendered_formulas"
IMAGES_DIR        = REPO_ROOT / "data/raw/arqmath/images"
POSTS_JSONL       = REPO_ROOT / "data/processed/posts.jsonl"
DATASET_DIR       = REPO_ROOT / "data/processed/dataset"

POSTS_PER_SHARD = 100_000

# ── Parquet schema ────────────────────────────────────────────────────────────

FORMULA_STRUCT = pa.struct([
    pa.field("formula_id",       pa.string()),
    pa.field("latex",            pa.string()),
    pa.field("rendered_typeset", pa.string()),
    pa.field("rendered_graph",   pa.string()),
])

SCHEMA = pa.schema([
    pa.field("post_id",               pa.int64()),
    pa.field("post_type",             pa.string()),
    pa.field("parent_id",             pa.int64()),
    pa.field("accepted_answer_id",    pa.int64()),
    pa.field("title",                 pa.string()),
    pa.field("tags",                  pa.list_(pa.string())),
    pa.field("score",                 pa.int64()),
    pa.field("creation_date",         pa.string()),
    pa.field("body_text",             pa.string()),
    pa.field("formulas",              pa.list_(FORMULA_STRUCT)),
    pa.field("post_images",           pa.list_(pa.string())),
    pa.field("has_formulas",          pa.bool_()),
    pa.field("has_post_images",       pa.bool_()),
    pa.field("has_rendered_formulas", pa.bool_()),
])


# ── Formula index loading ─────────────────────────────────────────────────────

def load_formula_map() -> dict[int, list[tuple[str, str]]]:
    """Load formula_index shards → {post_id: [(old_visual_id, latex), ...]} ordered by formula id.

    Only loads formula_id (old_visual_id) and latex — OPT/SLT remain in
    the formula_index Parquet shards as a separate joinable artifact.
    Memory: ~1-2 GB for the full 28M-formula corpus.
    """
    if not FORMULA_INDEX_DIR.exists():
        print(f"ERROR: Formula index not found at {FORMULA_INDEX_DIR}")
        print("Run: python scripts/dataset/01_build_formula_index.py")
        sys.exit(1)

    print("Loading formula index (this may take a few minutes)...")
    df = (
        pl.scan_parquet(str(FORMULA_INDEX_DIR / "*.parquet"))
        .filter(pl.col("type").is_in(["question", "answer"]))
        .select(["id", "post_id", "old_visual_id", "latex"])
        .group_by("post_id")
        .agg([
            pl.col("old_visual_id").sort_by("id"),
            pl.col("latex").sort_by("id"),
        ])
        .collect()
    )
    print(f"  {df.height:,} posts have formula records")

    formula_map: dict[int, list[tuple[str, str]]] = {}
    for row in df.iter_rows(named=True):
        formula_map[row["post_id"]] = list(zip(row["old_visual_id"], row["latex"]))

    return formula_map


# ── Rendered image path helpers ───────────────────────────────────────────────

def _typeset_path(formula_id: str) -> str | None:
    p = RENDERED_DIR / f"{formula_id}_typeset.png"
    return str(p.relative_to(REPO_ROOT)) if p.exists() else None


def _graph_path(formula_id: str) -> str | None:
    p = RENDERED_DIR / f"{formula_id}_graph.png"
    return str(p.relative_to(REPO_ROOT)) if p.exists() else None


def _post_image_paths(post_id: int) -> list[str]:
    post_dir = IMAGES_DIR / str(post_id)
    if not post_dir.is_dir():
        return []
    return sorted(
        str(f.relative_to(REPO_ROOT))
        for f in sorted(post_dir.iterdir())
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    )


# ── Shard writing ─────────────────────────────────────────────────────────────

def _write_shard(records: list[dict], shard_idx: int) -> None:
    out_path = DATASET_DIR / f"shard_{shard_idx:04d}.parquet"

    columns: dict[str, list] = {field.name: [] for field in SCHEMA}

    for r in records:
        columns["post_id"].append(r["post_id"])
        columns["post_type"].append(r["post_type"])
        columns["parent_id"].append(r.get("parent_id"))
        columns["accepted_answer_id"].append(r.get("accepted_answer_id"))
        columns["title"].append(r.get("title"))
        columns["tags"].append(r.get("tags") or [])
        columns["score"].append(r.get("score") or 0)
        columns["creation_date"].append(r.get("creation_date"))
        columns["body_text"].append(r.get("body_text", ""))
        columns["formulas"].append(r["formulas"])
        columns["post_images"].append(r["post_images"])
        columns["has_formulas"].append(bool(r["formulas"]))
        columns["has_post_images"].append(bool(r["post_images"]))
        columns["has_rendered_formulas"].append(
            any(f["rendered_typeset"] is not None for f in r["formulas"])
        )

    table = pa.table(columns, schema=SCHEMA)
    pq.write_table(table, out_path, compression="snappy")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if not POSTS_JSONL.exists():
        print(f"ERROR: posts.jsonl not found at {POSTS_JSONL}")
        sys.exit(1)

    # Check for existing shards
    existing_shards = sorted(DATASET_DIR.glob("shard_*.parquet"))
    if existing_shards and not force:
        total_existing = sum(pq.read_metadata(s).num_rows for s in existing_shards)
        print(
            f"Dataset already assembled: {len(existing_shards)} shards, "
            f"{total_existing:,} posts."
        )
        print("Use --force to rebuild.")
        return

    formula_map = load_formula_map()

    print(f"\nAssembling dataset from {POSTS_JSONL.name}...")
    total_posts = 0
    shard_idx   = 0
    shard_buf: list[dict] = []

    with POSTS_JSONL.open(encoding="utf-8") as f:
        lines = f
        for line in tqdm(lines, desc="Processing posts", unit="post"):
            post = json.loads(line)
            pid  = int(post["post_id"])

            formula_pairs = formula_map.get(pid, [])
            formulas = [
                {
                    "formula_id":       vid,
                    "latex":            latex,
                    "rendered_typeset": _typeset_path(vid),
                    "rendered_graph":   _graph_path(vid),
                }
                for vid, latex in formula_pairs
            ]

            shard_buf.append({
                "post_id":            pid,
                "post_type":          post.get("post_type", ""),
                "parent_id":          post.get("parent_id"),
                "accepted_answer_id": post.get("accepted_answer_id"),
                "title":              post.get("title"),
                "tags":               post.get("tags") or [],
                "score":              post.get("score") or 0,
                "creation_date":      post.get("creation_date"),
                "body_text":          post.get("text", ""),
                "formulas":           formulas,
                "post_images":        _post_image_paths(pid),
            })
            total_posts += 1

            if len(shard_buf) >= POSTS_PER_SHARD:
                _write_shard(shard_buf, shard_idx)
                shard_idx += 1
                shard_buf = []

    if shard_buf:
        _write_shard(shard_buf, shard_idx)
        shard_idx += 1

    # Write dataset info
    shards = sorted(DATASET_DIR.glob("shard_*.parquet"))
    posts_with_formulas  = sum(
        pq.read_table(s, columns=["has_formulas"]).column("has_formulas").to_pylist().count(True)
        for s in shards
    )
    posts_with_images = sum(
        pq.read_table(s, columns=["has_post_images"]).column("has_post_images").to_pylist().count(True)
        for s in shards
    )
    posts_with_rendered = sum(
        pq.read_table(s, columns=["has_rendered_formulas"]).column("has_rendered_formulas").to_pylist().count(True)
        for s in shards
    )

    info = {
        "total_posts":              total_posts,
        "posts_with_formulas":      posts_with_formulas,
        "posts_with_post_images":   posts_with_images,
        "posts_with_rendered_formulas": posts_with_rendered,
        "n_shards":                 shard_idx,
        "posts_per_shard":          POSTS_PER_SHARD,
    }
    with (DATASET_DIR / "dataset_info.json").open("w") as f:
        json.dump(info, f, indent=2)

    print(
        f"\nDataset assembled."
        f"\n  Total posts              : {total_posts:,}"
        f"\n  Posts with formulas      : {posts_with_formulas:,}"
        f"\n  Posts with images        : {posts_with_images:,}"
        f"\n  Posts with rendered fmls : {posts_with_rendered:,}"
        f"\n  Shards written           : {shard_idx}"
        f"\n  Output                   : {DATASET_DIR}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble the trimodal ARQMath dataset")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset shards",
    )
    args = parser.parse_args()
    main(force=args.force)
