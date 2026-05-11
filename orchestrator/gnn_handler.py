"""
gnn_handler.py — Formula search using the GNN dual-encoder.
Calls service-formula/gnn_handler.search() directly.
"""

import importlib.util
import logging
import sys
from pathlib import Path

from formula_utils import extract_formulas
from utils.scrape_url import scrape_post_url

logger = logging.getLogger(__name__)

_SERVICE_FORMULA = Path(__file__).resolve().parents[1] / "service-formula"
if str(_SERVICE_FORMULA) not in sys.path:
    sys.path.insert(0, str(_SERVICE_FORMULA))

# Load service-formula's gnn_handler under a unique module name to avoid
# colliding with this file's own name in sys.modules (circular import).
_spec = importlib.util.spec_from_file_location(
    "_formula_gnn_service",
    str(_SERVICE_FORMULA / "gnn_handler.py"),
)
_formula_gnn = importlib.util.module_from_spec(_spec)
sys.modules["_formula_gnn_service"] = _formula_gnn
_spec.loader.exec_module(_formula_gnn)


def fetch_gnn_results(query: str, k: int = 5) -> list:
    """
    Extract LaTeX formulas from a natural language query, search each one
    via the GNN formula model, and return results enriched with scraped post text.

    Args:
        query: A natural language question that may contain LaTeX formulas.
        k:     Number of formula results per formula found in the query.

    Returns:
        A list of dicts with keys: rank, score, visual_id, latex, post_id, url,
        scraped_text. Returns an empty list if no formulas are found.
    """
    formulas = extract_formulas(query)
    if not formulas:
        logger.info("No formulas found in query, skipping GNN retrieval.")
        return []

    seen_post_ids = set()
    results = []
    overall_rank = 1

    for latex in formulas:
        try:
            hits = _formula_gnn.search(latex, k=k)
        except Exception as exc:
            logger.error("GNN search failed: %s", exc)
            return results

        for hit in hits:
            post_id = hit.get("post_id")
            if not post_id or post_id in seen_post_ids:
                continue
            seen_post_ids.add(post_id)

            url = f"https://math.stackexchange.com/q/{post_id}"
            scraped_text = scrape_post_url(url)

            results.append({
                "rank":         overall_rank,
                "score":        hit.get("score"),
                "visual_id":    hit.get("visual_id"),
                "latex":        hit.get("latex"),
                "post_id":      post_id,
                "url":          url,
                "scraped_text": scraped_text,
            })
            overall_rank += 1

    return results
