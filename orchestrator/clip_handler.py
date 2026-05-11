"""
clip_handler.py — Image search using LongCLIP.
Calls service-clip/longclip_handler.search() directly.
"""

import logging
import sys
from pathlib import Path

from utils.scrape_url import scrape_post_url

logger = logging.getLogger(__name__)

_SERVICE_CLIP = Path(__file__).resolve().parents[1] / "service-clip"
if str(_SERVICE_CLIP) not in sys.path:
    sys.path.insert(0, str(_SERVICE_CLIP))

import longclip_handler as _longclip


def fetch_clip_results(query: str, k: int = 5) -> list:
    """
    Search images using LongCLIP and return results enriched with scraped post text.

    Args:
        query: A natural language question or math topic.
        k:     Number of image results to return.

    Returns:
        A list of dicts with keys: rank, score, image_id, source, title, url,
        file_path, scraped_text.
    """
    try:
        hits = _longclip.search(query, k=k)
    except Exception as exc:
        logger.error("CLIP search failed: %s", exc)
        return []

    seen_urls = set()
    results = []
    for hit in hits:
        url = hit.get("url", "")
        scraped_text = ""
        if url and url not in seen_urls:
            seen_urls.add(url)
            scraped_text = scrape_post_url(url)
        results.append({
            "rank":         hit.get("rank"),
            "score":        hit.get("score"),
            "image_id":     hit.get("image_id"),
            "source":       hit.get("source"),
            "title":        hit.get("title"),
            "url":          url,
            "file_path":    hit.get("file_path"),
            "scraped_text": scraped_text,
        })

    return results
