#!/usr/bin/env python3
"""
eval_arqmath.py — Evaluate TextHandler retrieval on ARQMath Task 1 queries.

Parses the ARQMath Topics XML, runs each query through TextHandler.search(),
extracts a numeric post-ID from the returned URL, then scores with ranx.

Usage:
    python eval_arqmath.py --qrel arqmath_task1.qrel
    python eval_arqmath.py --qrel arqmath_task1.qrel \\
        --topics ../../ARQMathQueries/Topics_Task1_2022.xml \\
        --top-k 100 --metrics ndcg@10 map precision@10 recall@100

QRELs must be in 4-column TREC format:
    topic_id  iter  doc_id  relevance
    e.g.:  A.301  0  12345678  2

Doc IDs in the run are the numeric post IDs extracted from the OpenSearch
result URLs (the part after /questions/ in a Math Stack Exchange or
MathOverflow URL).  This aligns with standard ARQMath Task 1 qrel format.

Requires: ranx, beautifulsoup4, lxml  (pip install ranx beautifulsoup4 lxml)
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup
from ranx import Qrels, Run, evaluate

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from text_handler import TextHandler

logger = logging.getLogger(__name__)

# Extracts the numeric post/thread ID from SE/MO question URLs.
# Handles patterns like /questions/4235721/ and /q/4235721
_POST_ID_RE = re.compile(r"/(?:questions?|q)/(\d+)", re.IGNORECASE)

DEFAULT_TOPICS  = _HERE.parent.parent / "ARQMathQueries" / "Topics_Task1_2022.xml"
DEFAULT_METRICS = ["ndcg@10", "ndcg@100", "map", "precision@10", "recall@100"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """Load a TREC-format qrel file into {qid: {doc_id: relevance}}.

    Lines with relevance == 0 are excluded because ranx treats grade 0 as
    non-relevant and including them would inflate denominator counts.
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


def parse_topics(xml_path: Path) -> Dict[str, str]:
    """Parse ARQMath Task 1 Topics XML and return {topic_id: query_text}.

    HTML markup and LaTeX dollar-sign delimiters are stripped so the query
    matches the preprocessing done inside TextHandler.search().
    """
    tree = ET.parse(xml_path)
    queries: Dict[str, str] = {}
    for topic in tree.findall(".//Topic"):
        topic_id = (topic.get("number") or "").strip()
        if not topic_id:
            continue
        title_raw = topic.findtext("Title", "")
        body_raw  = topic.findtext("Question", "")
        title = BeautifulSoup(title_raw, "lxml").get_text(" ", strip=True)
        body  = BeautifulSoup(body_raw,  "lxml").get_text(" ", strip=True)
        queries[topic_id] = f"{title} {body}".strip()
    return queries


def url_to_doc_id(url: str) -> str:
    """Extract numeric post ID from a Math Stack Exchange / MathOverflow URL.

    Returns the raw URL unchanged if no numeric ID can be parsed, so the
    run still contains an entry (it just won't match any qrel).
    """
    m = _POST_ID_RE.search(url)
    return m.group(1) if m else url


# ---------------------------------------------------------------------------
# Run construction
# ---------------------------------------------------------------------------

def build_run(
    topics: Dict[str, str],
    handler: TextHandler,
    top_k: int,
) -> Dict[str, Dict[str, float]]:
    """Query TextHandler for every topic and assemble a ranx-compatible run dict.

    If the same post ID appears more than once in the results (e.g. from
    different indices), only the highest score is kept.
    """
    run: Dict[str, Dict[str, float]] = {}
    total = len(topics)
    for idx, (topic_id, query_text) in enumerate(topics.items(), 1):
        print(f"\r  [{idx:>4}/{total}] {topic_id}", end="", flush=True)
        hits = handler.search(query_text, top_k=top_k)
        doc_scores: Dict[str, float] = {}
        for hit in hits:
            doc_id = url_to_doc_id(hit.doc_id)
            if doc_id not in doc_scores or hit.score > doc_scores[doc_id]:
                doc_scores[doc_id] = hit.score
        if doc_scores:
            run[topic_id] = doc_scores
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
        help="Path to TREC-format ARQMath Task 1 qrel file (required)",
    )
    parser.add_argument(
        "--topics", default=str(DEFAULT_TOPICS),
        help=f"Path to Topics_Task1_2022.xml (default: {DEFAULT_TOPICS})",
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
        "--verbose", action="store_true",
        help="Enable DEBUG logging from TextHandler and OpenSearch client",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    qrel_path   = Path(args.qrel)
    topics_path = Path(args.topics)

    if not qrel_path.is_file():
        parser.error(f"Qrel file not found: {qrel_path}")
    if not topics_path.is_file():
        parser.error(f"Topics file not found: {topics_path}")

    # --- Load data ---
    print(f"Loading topics from:  {topics_path}")
    topics = parse_topics(topics_path)
    print(f"  {len(topics)} topics loaded")

    print(f"Loading qrels from:   {qrel_path}")
    qrels_dict = load_qrels(qrel_path)
    print(f"  {len(qrels_dict)} queries have relevance judgements")

    eval_topics: Dict[str, str] = {
        tid: q for tid, q in topics.items() if tid in qrels_dict
    }
    if not eval_topics:
        print(
            "WARNING: No overlap between topic IDs and qrel query IDs.\n"
            "  Topic IDs  (first 5): " + str(list(topics.keys())[:5]) + "\n"
            "  Qrel IDs   (first 5): " + str(list(qrels_dict.keys())[:5]) + "\n"
            "Proceeding with all topics; metrics will likely be 0."
        )
        eval_topics = topics
    else:
        print(f"  {len(eval_topics)} of {len(topics)} topics have qrel entries")

    # --- Retrieve ---
    print(f"\nRunning TextHandler.search() [top_k={args.top_k}] ...")
    handler  = TextHandler()
    run_dict = build_run(eval_topics, handler, args.top_k)

    if not run_dict:
        print("ERROR: no results were returned — cannot evaluate.")
        sys.exit(1)

    # --- Evaluate ---
    qrels   = Qrels(qrels_dict)
    run     = Run(run_dict, name="TextHandler")
    results = evaluate(qrels, run, args.metrics)

    print("\n=== ARQMath Task 1 — TextHandler Results ===")
    for metric, score in results.items():
        print(f"  {metric:<22s}  {score:.4f}")


if __name__ == "__main__":
    main()
