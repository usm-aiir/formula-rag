"""
Download images referenced in parsed ARQMath posts.

Reads the JSONL produced by parse_posts.py and downloads all image URLs
to a local directory, organized by post_id.

Output layout:
    images_dir/
        {post_id}/
            0.png
            1.jpg
            ...
        manifest.jsonl   ← one record per image:
                            {post_id, index, url, local_path, status}

Only downloads images from allowed hosts (i.stack.imgur.com by default).
Skips images that have already been downloaded (idempotent).
Uses async HTTP with a configurable concurrency limit.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import httpx
from tqdm.asyncio import tqdm as async_tqdm

# MSE images are hosted here; reject anything else to avoid surprise fetches
ALLOWED_HOSTS = {"i.stack.imgur.com", "i.sstatic.net"}

# Timeout per image request
REQUEST_TIMEOUT = 20.0

# Max simultaneous downloads
DEFAULT_CONCURRENCY = 16


def _allowed(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in ALLOWED_HOSTS


def _extension(url: str, content_type: str | None) -> str:
    """Determine file extension from URL or Content-Type header."""
    path_ext = Path(urlparse(url).path).suffix.lower()
    if path_ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        return path_ext
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext
    return ".img"


async def _download_one(
    client: httpx.AsyncClient,
    post_id: int,
    index: int,
    url: str,
    images_dir: Path,
    semaphore: asyncio.Semaphore,
) -> dict:
    record = {"post_id": post_id, "index": index, "url": url, "local_path": None, "status": None}

    if not _allowed(url):
        record["status"] = "skipped_host"
        return record

    post_dir = images_dir / str(post_id)

    # Check if already downloaded
    existing = list(post_dir.glob(f"{index}.*"))
    if existing:
        record["local_path"] = str(existing[0])
        record["status"] = "exists"
        return record

    async with semaphore:
        try:
            response = await client.get(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
            response.raise_for_status()

            ext = _extension(url, response.headers.get("content-type"))
            post_dir.mkdir(parents=True, exist_ok=True)
            local_path = post_dir / f"{index}{ext}"
            local_path.write_bytes(response.content)

            record["local_path"] = str(local_path)
            record["status"] = "ok"

        except httpx.HTTPStatusError as e:
            record["status"] = f"http_{e.response.status_code}"
        except Exception as e:
            record["status"] = f"error_{type(e).__name__}"

    return record


async def download_images(
    posts_jsonl: Path,
    images_dir: Path,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
) -> Path:
    """
    Download all images from a parsed posts JSONL file.

    Args:
        posts_jsonl — path to JSONL produced by parse_posts.py
        images_dir  — root directory for downloaded images
        concurrency — max simultaneous HTTP requests
        limit       — if set, stop after this many posts (useful for testing)

    Returns:
        path to manifest.jsonl
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = images_dir / "manifest.jsonl"

    # Build the full task list from the JSONL
    tasks: list[tuple[int, int, str]] = []  # (post_id, index, url)
    with posts_jsonl.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            post = json.loads(line)
            for idx, url in enumerate(post.get("image_urls", [])):
                tasks.append((post["post_id"], idx, url))

    if not tasks:
        print("No image URLs found in posts file.")
        return manifest_path

    print(f"Found {len(tasks):,} images across posts. Starting downloads...")

    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async with httpx.AsyncClient(http2=True) as client:
        coros = [
            _download_one(client, post_id, idx, url, images_dir, semaphore)
            for post_id, idx, url in tasks
        ]
        for coro in async_tqdm.as_completed(coros, total=len(coros), desc="Downloading images"):
            result = await coro
            results.append(result)

    with manifest_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ok = sum(1 for r in results if r["status"] == "ok")
    exists = sum(1 for r in results if r["status"] == "exists")
    skipped = sum(1 for r in results if r["status"] and r["status"].startswith("skipped"))
    errors = len(results) - ok - exists - skipped

    print(f"\nDownload complete.")
    print(f"  downloaded : {ok:,}")
    print(f"  already existed: {exists:,}")
    print(f"  skipped (host) : {skipped:,}")
    print(f"  errors         : {errors:,}")
    print(f"  manifest       : {manifest_path}")

    return manifest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download images from parsed ARQMath posts")
    parser.add_argument("posts_jsonl", type=Path, help="Path to parsed posts .jsonl")
    parser.add_argument("images_dir", type=Path, help="Directory to save images into")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max simultaneous downloads (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N posts (for testing)",
    )
    args = parser.parse_args()

    asyncio.run(
        download_images(
            args.posts_jsonl,
            args.images_dir,
            concurrency=args.concurrency,
            limit=args.limit,
        )
    )
