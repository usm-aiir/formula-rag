import logging 
import sys
from pathlib import Path
from typing import List, Optional

# personally I dont like this pattern of modifying sys.path. Look into other ways to 
# fix this. Maybe a pyproject.toml file?
_THIS_DIR = Path(__file__).resolve().parent
# /home/first.last/formula-rag/service-formula

_PROJECT_ROOT = _THIS_DIR if (_THIS_DIR / "src").exists() else _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
# adds /home/first.last/formula-rag/service-formula to sys.path 

from src.task3.utils.formula_retriever import FormulaRetriever

logger = logging.getLogger(__name__)

PARQUET_DIR: Path = _PROJECT_ROOT / "data" / "processed" / "formula_index"

_retriever: Optional[FormulaRetriever] = None
_formula_data: Optional[dict] = None


def _get_retriever() -> FormulaRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FormulaRetriever()
    return _retriever


def _get_formula_data() -> dict:
    global _formula_data
    if _formula_data is None:
        # we are dealing with parquet files here, so have to handle them slightly differently. Was reccomended
        # to use pyarrow, but panadas would also work. Feel free to try the change
        import pyarrow.parquet as pq
        _formula_data = {}
        for shard in sorted(PARQUET_DIR.glob("shard_*.parquet")):
            table = pq.read_table(shard, columns=["visual_id", "latex", "post_id"])
            # zip since parquet files are columnar, so we need to iterate row-wise
            for vid, latex, post_id in zip(
                # .to_pylist() converts from pyarrow array to regular python list
                table["visual_id"].to_pylist(),
                table["latex"].to_pylist(),
                table["post_id"].to_pylist(),
            ):
                _formula_data[str(vid)] = {
                    "latex":   str(latex) if latex else "",
                    "post_id": int(post_id) if post_id is not None else None,
                }
        logger.info("[gnn] loaded formula data: %d formulas", len(_formula_data))
    return _formula_data


def search(latex_query: str, k: int = 10) -> List[dict]:
    """
    Retrieve the top-k most similar formulas for a LaTeX query string.

    Returns a list of dicts with keys: rank, visual_id, latex, post_id, score.
    """
    retriever = _get_retriever()
    formula_data = _get_formula_data()

    raw_results = retriever.search(latex_query, final_top_k=k)

    results = []
    for rank, (visual_id, score) in enumerate(raw_results, start=1):
        entry = formula_data.get(visual_id, {})
        results.append({
            "rank":      rank,
            "visual_id": visual_id,
            "latex":     entry.get("latex", ""),
            "post_id":   entry.get("post_id"),
            "score":     float(score),
        })
    return results

if __name__ == "__main__":
    # quick test - when yall put this into production remove the main
    import argparse

    parser = argparse.ArgumentParser(description="GNN formula retrieval handler")
    parser.add_argument("--search", type=str,             help="LaTeX query to search")
    parser.add_argument("--k",      type=int, default=10, help="Number of results")
    args = parser.parse_args()

    if args.search:
        hits = search(args.search, k=args.k)
        if not hits:
            print("No results.")
        else:
            for h in hits:
                post_id_str = str(h['post_id']) if h.get('post_id') else 'N/A'
                print(f"[{h['rank']:>3}] score={h['score']:.4f}  id={h['visual_id']}  post_id={post_id_str}  latex={h['latex'][:60]}")
    else:
        parser.print_help()
