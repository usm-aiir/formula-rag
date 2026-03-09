"""
Build the extended ARQMath corpus.

Runs the full Phase 1 pipeline:
  1. Parse Posts.V1.3.xml → data/processed/posts.jsonl
  2. Download images      → data/raw/arqmath/images/

Usage (from repo root):
    python scripts/build_corpus.py

    # Parse only (skip image download)
    python scripts/build_corpus.py --no-images

    # Parse only posts that have images
    python scripts/build_corpus.py --images-only

    # Test run: process first 1000 posts, download max 100 images
    python scripts/build_corpus.py --limit 1000 --image-limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from repo root without installing the package
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.parsing.parse_posts import parse_posts
from src.data_pipeline.collection.download_images import download_images

# ── Paths ──────────────────────────────────────────────────────────────────────
POSTS_XML = REPO_ROOT / "data/raw/arqmath/collection/Posts.V1.3.xml"
POSTS_JSONL = REPO_ROOT / "data/processed/posts.jsonl"
IMAGES_DIR = REPO_ROOT / "data/raw/arqmath/images"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build extended ARQMath corpus")
    parser.add_argument(
        "--no-images", action="store_true",
        help="Skip image download step",
    )
    parser.add_argument(
        "--images-only", action="store_true",
        help="Only include posts that contain at least one image in posts.jsonl",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after parsing this many posts (for testing)",
    )
    parser.add_argument(
        "--image-limit", type=int, default=None,
        help="Stop after attempting images from this many posts (for testing)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=16,
        help="Max simultaneous image downloads (default: 16)",
    )
    parser.add_argument(
        "--posts-xml", type=Path, default=POSTS_XML,
        help=f"Path to Posts.V1.3.xml (default: {POSTS_XML})",
    )
    args = parser.parse_args()

    posts_xml: Path = args.posts_xml

    # ── Step 1: Parse posts ────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Parsing posts XML")
    print("=" * 60)

    if not posts_xml.exists():
        print(f"\nError: Posts XML not found at {posts_xml}")
        print("Run: bash scripts/download_arqmath.sh")
        print("Then: cd data/raw/arqmath/collection && unzip Posts.V1.3.zip")
        sys.exit(1)

    counts = parse_posts(
        posts_xml,
        POSTS_JSONL,
        images_only=args.images_only,
    )

    print(f"\nParsed posts written to: {POSTS_JSONL}")
    for k, v in counts.items():
        print(f"  {k}: {v:,}")

    # ── Step 2: Download images ────────────────────────────────────────────────
    if args.no_images:
        print("\nSkipping image download (--no-images).")
        return

    print("\n" + "=" * 60)
    print("Step 2: Downloading images")
    print("=" * 60)

    asyncio.run(
        download_images(
            POSTS_JSONL,
            IMAGES_DIR,
            concurrency=args.concurrency,
            limit=args.image_limit,
        )
    )

    print("\nCorpus build complete.")
    print(f"  posts  : {POSTS_JSONL}")
    print(f"  images : {IMAGES_DIR}")


if __name__ == "__main__":
    main()
