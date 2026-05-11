#!/usr/bin/env python3
"""
eval_arqmath.py — Evaluate GNN formula handler on ARQMath formula queries.

Reads formula queries from Topics_Formulas_SLT.V0.1.tsv, runs each formula
through gnn_handler.search(), and scores the results with ranx.

Each row in the TSV is treated as an independent query (query ID = the
formula's `id` column, e.g. "q_1").  The GNN handler returns visual_ids
as document identifiers, so the qrel file should use the same visual_ids
as doc IDs.

Usage:
    python eval_arqmath.py --qrel /path/to/arqmath_task2.qrel
    python eval_arqmath.py --qrel /path/to/arqmath_task2.qrel \\
        --formulas ../../ARQMathQueries/Topics_Formulas_SLT.V0.1.tsv \\
        --top-k 100 --type title

QRELs must be in 4-column TREC format:
    formula_id  iter  visual_id  relevance
    e.g.:  q_1  0  some_visual_id  2

LaTeX for each formula is extracted from the MathML alttext attribute in
the TSV.  Rows with no recoverable alttext are skipped.

Requires: ranx  (pip install ranx)
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from ranx import Qrels, Run, evaluate

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import gnn_handler

logger = logging.getLogger(__name__)

DEFAULT_FORMULAS = _HERE.parent.parent / "ARQMathQueries" / "Topics_Formulas_SLT.V0.1.tsv"
DEFAULT_METRICS  = ["ndcg@10", "ndcg@100", "map", "precision@10", "recall@100"]

# Matches the LaTeX string inside alttext="..." in MathML attributes.
# MathML in this file contains no escaped double-quotes inside alttext values.
_ALTTEXT_RE = re.compile(r'alttext="([^"]*)"')


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


def parse_formula_queries(tsv_path: Path, type_filter: str) -> List[Dict[str, str]]:
    """Parse Topics_Formulas_SLT TSV; return list of query dicts.

    Each dict has keys: id, topic_id, thread_id, type, latex.

    Rows with no recoverable alttext (trivial MathML with no LaTeX content)
    are silently skipped.  If type_filter is "title" or "body", only rows
    of that type are kept.
    """
    entries: List[Dict[str, str]] = []
    with open(tsv_path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            fid, topic_id, thread_id, ftype, mathml = (
                row[0].strip(), row[1].strip(),
                row[2].strip(), row[3].strip(), row[4],
            )
            if type_filter != "all" and ftype != type_filter:
                continue
            m = _ALTTEXT_RE.search(mathml)
            if not m:
                continue
            latex = m.group(1).strip()
            if not latex:
                continue
            entries.append({
                "id":        fid,
                "topic_id":  topic_id,
                "thread_id": thread_id,
                "type":      ftype,
                "latex":     latex,
            })
    return entries


# ---------------------------------------------------------------------------
# Run construction
# ---------------------------------------------------------------------------

def build_run(
    formula_queries: List[Dict[str, str]],
    top_k: int,
    qrels_dict: Dict[str, Dict[str, int]],
    include_all: bool,
) -> Dict[str, Dict[str, float]]:
    """Search each formula query and return a ranx-compatible run dict.

    By default only queries that appear in the qrel are executed.  Pass
    include_all=True to run every formula regardless.

    Doc IDs in the run are the visual_ids returned by gnn_handler.search().
    If the same visual_id appears more than once for a query (should not
    happen), only the highest score is kept.
    """
    if not include_all:
        formula_queries = [q for q in formula_queries if q["id"] in qrels_dict]

    run: Dict[str, Dict[str, float]] = {}
    total = len(formula_queries)

    for idx, entry in enumerate(formula_queries, 1):
        qid   = entry["id"]
        latex = entry["latex"]
        print(
            f"\r  [{idx:>5}/{total}] {qid}  ({entry['topic_id']}  {entry['type']})",
            end="", flush=True,
        )
        try:
            hits = gnn_handler.search(latex, k=top_k)
        except Exception as exc:
            logger.warning("Search failed for %s (%s): %s", qid, latex[:40], exc)
            continue

        doc_scores: Dict[str, float] = {}
        for hit in hits:
            vid   = str(hit["visual_id"])
            score = float(hit["score"])
            if vid not in doc_scores or score > doc_scores[vid]:
                doc_scores[vid] = score
        if doc_scores:
            run[qid] = doc_scores

    print()
    return run


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--qrel", required=True,
        help="Path to TREC-format ARQMath formula qrel file (required)",
    )
    parser.add_argument(
        "--formulas", default=str(DEFAULT_FORMULAS),
        help=f"Path to Topics_Formulas_SLT.V0.1.tsv (default: {DEFAULT_FORMULAS})",
    )
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="Number of results to retrieve per query (default: 100)",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        help="ranx metric names to compute (default: ndcg@10 ndcg@100 map precision@10 recall@100)",
    )
    parser.add_argument(
        "--type", dest="type_filter",
        choices=["title", "body", "all"], default="all",
        help="Restrict queries to title-only or body-only formulas (default: all)",
    )
    parser.add_argument(
        "--all-queries", action="store_true",
        help="Run every formula query even if it has no qrel entry",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    qrel_path     = Path(args.qrel)
    formulas_path = Path(args.formulas)

    if not qrel_path.is_file():
        parser.error(f"Qrel file not found: {qrel_path}")
    if not formulas_path.is_file():
        parser.error(f"Formulas file not found: {formulas_path}")

    # --- Load data ---
    print(f"Loading qrels from:   {qrel_path}")
    qrels_dict = load_qrels(qrel_path)
    print(f"  {len(qrels_dict)} queries have relevance judgements")

    print(f"Parsing formulas from: {formulas_path}")
    formula_queries = parse_formula_queries(formulas_path, args.type_filter)
    print(f"  {len(formula_queries)} formula queries parsed  (type filter: {args.type_filter})")

    if not formula_queries:
        print("No formula queries found — check --formulas path and --type filter.")
        sys.exit(1)

    overlap = [q for q in formula_queries if q["id"] in qrels_dict]
    if not overlap and not args.all_queries:
        print(
            "WARNING: no formula IDs match the qrel query IDs.\n"
            "  Formula IDs  (first 5): " + str([q["id"] for q in formula_queries[:5]]) + "\n"
            "  Qrel IDs     (first 5): " + str(list(qrels_dict.keys())[:5]) + "\n"
            "  Use --all-queries to run all formulas regardless."
        )
    else:
        print(f"  {len(overlap)} of {len(formula_queries)} formulas have qrel entries")

    # --- Retrieve ---
    print(f"\nInitialising GNN retriever and running search [top_k={args.top_k}] ...")
    run_dict = build_run(formula_queries, args.top_k, qrels_dict, args.all_queries)

    if not run_dict:
        print("ERROR: no results were returned — cannot evaluate.")
        sys.exit(1)

    # --- Evaluate ---
    qrels   = Qrels(qrels_dict)
    run     = Run(run_dict, name="GNN")
    results = evaluate(qrels, run, args.metrics)

    print("\n=== ARQMath Formula Retrieval — GNN Results ===")
    for metric, score in results.items():
        print(f"  {metric:<22s}  {score:.4f}")


if __name__ == "__main__":
    main()
