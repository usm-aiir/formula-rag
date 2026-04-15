# GNN formula handler - thin adapter over the Task 3 FormulaRetriever
# python gnn_handler.py --search "\frac{x^2}{2}" --k 10
#
# 1. Download and extract the ARQMath dataset (~2.8 GB compressed)
#       bash scripts/setup.sh
#
# 2. Build Parquet shards from raw ARQMath TSVs (latex / opt / slt columns)
#       python -m src.data.index
#
# 3. Train the Phase 1 single-branch GAT encoder (produces best.pt)
#       python -m src.task3.train --config configs/task3.yaml
#
# 4. Mine hard negatives using the Phase 1 checkpoint
#    (produces data/processed/self_mined_hard_negatives.jsonl)
#       python src/task3/mine_hard_negatives.py
#
# 5. Train the Phase 2 encoder on hard negatives (produces phase2_best.pt)
#       python src/task3/train.py
#
# 6. 5-fold cross-validation training of the Dual Encoder
#    (produces checkpoints/task3/phase3_fusion/phase3_atten_fusion_fold{1-5}_best.pt)
#       python src/task3/train_fusion.py
#
# 7. Average the 5 fold checkpoints into a single model soup
#    (produces checkpoints/task3/phase3_fusion/phase3_atten_fusion_ensemble_soup.pt)
#       python src/task3/utils/brew_model_soup.py
#
# 8. Encode the 8.3M formula corpus and write the FAISS index
#    (produces checkpoints/task3/faiss_index/phase3_dense.faiss + corpus_ids.npy)
#       python src/task3/utils/build_faiss.py
#
# 9. Tokenize MathML and build the BM25 sparse index
#    (produces checkpoints/task3/bm25_index/)
#       python src/task3/utils/build_bm25.py
#
# 10. Migrate Parquet shards to a SQLite cache for fast MathML lookups
#     (produces data/processed/formula_cache.db)
#       python src/task3/utils/build_sqlite_cache.py
#
# After all steps are complete, this handler is ready to use.

import sys
from pathlib import Path
from typing import List, Optional

_TRISEARCH_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TRISEARCH_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.task3.utils.formula_retriever import FormulaRetriever

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
        import pyarrow.parquet as pq
        _formula_data = {}
        for shard in sorted(PARQUET_DIR.glob("shard_*.parquet")):
            table = pq.read_table(shard, columns=["visual_id", "latex", "post_id"])
            for vid, latex, post_id in zip(
                table["visual_id"].to_pylist(),
                table["latex"].to_pylist(),
                table["post_id"].to_pylist(),
            ):
                _formula_data[str(vid)] = {
                    "latex":   str(latex) if latex else "",
                    "post_id": int(post_id) if post_id is not None else None,
                }
        print(f"[gnn] loaded formula data: {len(_formula_data)} formulas")
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
