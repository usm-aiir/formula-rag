"""
Structural Path Re-ranker (Tangent-style)

Replaces text-based BM25 with a custom IDF-Weighted Subtree Coverage algorithm.
This strictly measures if the query's mathematical structure exists inside 
the candidate document, without punishing the document for being long 
(solving the Sub-Expression Matching problem).

This module implements a mathematical re-ranking pipeline based on IDF-weighted
Subtree Coverage. It utilizes path-based structural extraction from MathML (OPT/SLT)
to perform fine-grained re-ranking of candidates retrieved.

Key Logic:
    - Max Path Depth = 4 (Optimized for MathML structure).
    - Multi-level path extraction (Exact, Alpha, Universal).
    - Deterministic tree canonicalization for commutative operators.
    - Asymmetrical Reciprocal Rank Fusion (RRF) with structural prioritization.
    - Double-weighting of Exact (E::) paths via list duplication.
    - Unfiltered (Standard) vs Filtered (Prime) Evaluation metrics.
"""

import gc
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pyarrow.parquet as pq
import pytrec_eval
from tqdm import tqdm

# Internal project imports
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.task3.dataset import load_qrels, load_topics

# --- Paths ---
_PHASE3_RUN_PATH = _PROJECT_ROOT / "data/processed/phase3_hybrid_run.json"
_OUT_RUN_PATH = _PROJECT_ROOT / "data/processed/phase4_structural_run_pb.json"
_PARQUET_DIR = _PROJECT_ROOT / "data/processed/formula_index"
_OUT_METRICS_PATH = _PROJECT_ROOT / "data/processed/phase4_structural_metrics_pb.json"

# --- Constants ---
EVAL_SPLIT = "eval"
MAX_PATH_DEPTH = 4  
K_RRF_DENSE = 60          
K_RRF_STRUCT = 15  # Sharp structural override spike

IGNORE_TAGS: Set[str] = {
    'mrow', 'mstyle', 'mpadded', 'mphantom', 'maligngroup', 'malignmark',
    'semantics', 'annotation', 'annotation-xml', 'id', 'mspace', 'menclose', 'maction'
}


def extract_structural_paths(xml_str: str, max_depth: int = MAX_PATH_DEPTH) -> List[str]:
    """
    Parses a MathML string and extracts structural n-gram paths with PB weighting.

    Args:
        xml_str: Raw MathML string (OPT or SLT).
        max_depth: Maximum length of the structural n-gram paths.

    Returns:
        A list of string-encoded paths including duplicate E:: entries for weighting.
    """
    if not xml_str or not isinstance(xml_str, str):
        return []
        
    xml_str = re.sub(r'\sxmlns="[^"]+"', '', xml_str, count=1)
    try:
        root = ET.fromstring(xml_str)
    except Exception:
        return []
        
    def get_shape_string(node: ET.Element, mask_vars: bool = True) -> str:
        """Generates a recursive string representation of a node's shape for sorting."""
        tag = node.tag.split('}')[-1]
        if tag in IGNORE_TAGS:
            return "".join(get_shape_string(child, mask_vars) for child in node)
            
        if mask_vars and tag in {'mi', 'ci', 'identifier'}:
            res = f"<{tag}>V</{tag}>"
        else:
            text = (node.text or "").strip()
            res = f"<{tag}>{text}</{tag}>"
            
        for child in node:
            res += get_shape_string(child, mask_vars)
        return res

    def canonicalize_tree(node: ET.Element):
        """Recursively sorts children of commutative operators to ensure determinism."""
        for child in list(node):
            canonicalize_tree(child)
            
        tag = node.tag.split('}')[-1]
        if tag == 'apply' and len(node) > 1:
            op_tag = node[0].tag.split('}')[-1]
            if op_tag in {'plus', 'times', 'eq', 'and', 'or', 'union', 'intersect', 'equivalent'}:
                op_node = node[0]
                args = list(node)[1:]
                # Primary sort: Masked Shape. Secondary sort: Exact Text (Alphabetical)
                args.sort(key=lambda n: (get_shape_string(n, True), get_shape_string(n, False)))
                node[:] = [op_node] + args
        elif tag in {'set', 'list'}:
            args = list(node)
            args.sort(key=lambda n: (get_shape_string(n, True), get_shape_string(n, False)))
            node[:] = args

    canonicalize_tree(root)
        
    paths = []
    var_map = {}
    var_counter = 1
    
    def dfs(node: ET.Element, path_exact: List[str], path_alpha: List[str], 
            path_univ: List[str], edge_pos: str = "R"):
        """Depth-first traversal to build multi-level path n-grams."""
        nonlocal var_counter
        tag = node.tag.split('}')[-1]
        
        if tag in IGNORE_TAGS:
            for idx, child in enumerate(node):
                dfs(child, path_exact, path_alpha, path_univ, edge_pos)
            return

        text = (node.text or "").strip()
        base_repr = f"{edge_pos}_{tag}"
        
        repr_exact, repr_alpha, repr_univ = base_repr, base_repr, base_repr
        
        if text and len(text) < 15:
            clean_text = re.sub(r'\W+', '', text)
            if clean_text:
                if tag in ['mi', 'ci', 'identifier']:
                    repr_exact = f"{base_repr}_{clean_text}"
                    if clean_text not in var_map:
                        var_map[clean_text] = f"V{var_counter}"
                        var_counter += 1
                    repr_alpha = f"{base_repr}_{var_map[clean_text]}"
                    repr_univ = f"{base_repr}_V"
                else:
                    repr_exact = f"{base_repr}_{clean_text}"
                    repr_alpha = f"{base_repr}_{clean_text}"
                    repr_univ = f"{base_repr}_{clean_text}"
                
        pe, pa, pu = path_exact + [repr_exact], path_alpha + [repr_alpha], path_univ + [repr_univ]
        
        for i in range(1, min(len(pe), max_depth) + 1):
            # PB Logic: List-append double weighting for exact matches
            paths.append("E::" + "::".join(pe[-i:]))
            paths.append("E::" + "::".join(pe[-i:])) 
            paths.append("A::" + "::".join(pa[-i:]))
            paths.append("U::" + "::".join(pu[-i:]))
            
        for idx, child in enumerate(node):
            dfs(child, pe, pa, pu, str(idx))
            
    dfs(root, [], [], [], "R")
    return paths


def get_depth_weight(path_token: str) -> float:
    """
    Rewards deep alignments more strictly based on path length.

    Args:
        path_token: String representation of the path n-gram.

    Returns:
        Weight multiplier (Depth^1.5).
    """
    depth = float(len(path_token.split("::")) - 1)
    return depth ** 1.5


def main():
    """Main execution block for re-ranking and evaluation."""
    print("\n" + "="*50)
    print("PHASE 4: PB STRUCTURAL RE-RANKER")
    print("="*50)

    if not _PHASE3_RUN_PATH.exists():
        print(f"Error: Phase 3 run file not found: {_PHASE3_RUN_PATH.name}")
        sys.exit(1)
        
    with open(_PHASE3_RUN_PATH, "r") as f:
        phase3_run = json.load(f)

    topics = load_topics(EVAL_SPLIT)
    qrels = load_qrels(EVAL_SPLIT)
    
    # Map topics to proxy visual IDs for query representation
    proxy_vid_to_topic = {}
    for topic_id in topics.keys():
        if topic_id in qrels:
            positives = [str(vid) for vid, grade in qrels[topic_id].items() if grade >= 2.0]
            if positives:
                proxy_vid_to_topic[positives[0]] = topic_id

    print("\nExtracting required XMLs from Parquet shards...")
    target_vids = set(proxy_vid_to_topic.keys())
    for candidates in phase3_run.values():
        target_vids.update(candidates.keys())

    xml_cache = {}
    shard_files = sorted(list(_PARQUET_DIR.glob("shard_*.parquet")))
    for shard in tqdm(shard_files, desc="Scanning Shards"):
        df = pq.read_table(shard, columns=["visual_id", "opt", "slt"]).to_pandas()
        df = df.dropna(subset=["opt", "slt"])
        mask = df["visual_id"].astype(str).isin(target_vids)
        target_df = df[mask]
        
        for _, row in target_df.iterrows():
            xml_cache[str(row["visual_id"])] = {"opt": row["opt"], "slt": row["slt"]}
            
        del df
        gc.collect()
        if len(xml_cache) == len(target_vids):
            break

    print("\nExecuting IDF-Weighted Subtree Coverage Scoring...")
    structural_run = {}
    
    for topic_id, candidates in tqdm(phase3_run.items(), desc="Processing Topics"):
        proxy_vid = next((vid for vid, tid in proxy_vid_to_topic.items() if tid == topic_id), None)
        if not proxy_vid or proxy_vid not in xml_cache:
            structural_run[topic_id] = candidates
            continue
            
        q_xml = xml_cache[proxy_vid]
        q_tokens = extract_structural_paths(q_xml["opt"]) + extract_structural_paths(q_xml["slt"])
        if not q_tokens:
            structural_run[topic_id] = candidates
            continue

        valid_doc_vids, doc_tokens_list = [], []
        for vid in candidates.keys():
            if vid in xml_cache:
                d_tokens = extract_structural_paths(xml_cache[vid]["opt"]) + \
                           extract_structural_paths(xml_cache[vid]["slt"])
                if d_tokens:
                    doc_tokens_list.append(d_tokens)
                    valid_doc_vids.append(vid)
                    
        if not valid_doc_vids:
            structural_run[topic_id] = candidates
            continue
        
        # Calculate Local IDF
        n_docs = len(valid_doc_vids)
        df_counts = defaultdict(int)
        for tokens in doc_tokens_list:
            for t in set(tokens):
                df_counts[t] += 1
        idf = {t: math.log(1.0 + (n_docs - f + 0.5) / (f + 0.5)) for t, f in df_counts.items()}
        
        # IDF-Weighted Scoring
        q_counts = Counter(q_tokens)
        q_total_tokens = sum(q_counts.values())
        max_q_score = sum(count * idf.get(t, 1.0) * get_depth_weight(t) 
                          for t, count in q_counts.items()) or 1.0
        
        structural_scores = {}
        for vid, d_tokens in zip(valid_doc_vids, doc_tokens_list):
            d_counts = Counter(d_tokens)
            d_total_tokens = sum(d_counts.values())
            
            overlap_score = sum(min(count, d_counts.get(t, 0)) * idf.get(t, 1.0) * get_depth_weight(t)
                                for t, count in q_counts.items())
                
            # Soft Length Penalty (PB 0.25 exponent)
            length_ratio = q_total_tokens / max(1.0, d_total_tokens)
            length_penalty = min(1.0, length_ratio) ** 0.25
            
            structural_scores[vid] = (overlap_score / max_q_score) * length_penalty
        
        # Asymmetrical Reciprocal Rank Fusion
        struct_ranked = sorted(structural_scores.keys(), key=lambda v: structural_scores[v], reverse=True)
        dense_ranked = list(candidates.keys()) 
        
        fused_scores = defaultdict(float)
        for rank, vid in enumerate(dense_ranked):
            fused_scores[vid] += 1.0 / (K_RRF_DENSE + rank + 1)
        for rank, vid in enumerate(struct_ranked):
            fused_scores[vid] += 3.0 / (K_RRF_STRUCT + rank + 1)  
            
        structural_run[topic_id] = dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))

    with open(_OUT_RUN_PATH, "w") as f:
        json.dump(structural_run, f)

    print("\n" + "="*60)
    print("TASK 2 EVALUATION (Standard vs. Prime Metrics)")
    print("="*60)
    
    # Format qrels for pytrec_eval
    qrels_dict = {str(tid): {str(vid): int(score) for vid, score in items.items()} 
                  for tid, items in qrels.items()}
                  
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {"ndcg_cut", "map_cut", "P", "bpref"}, relevance_level=2)
    
    # ---------------------------------------------------------
    # 1. STANDARD EVALUATION (Unfiltered)
    # ---------------------------------------------------------
    metrics_std = evaluator.evaluate(structural_run)
    
    # ---------------------------------------------------------
    # 2. PRIME EVALUATION (Filtered to Judged Documents Only)
    # ---------------------------------------------------------
    filtered_run = {}
    for tid, hits in structural_run.items():
        filtered_run[tid] = {}
        if tid in qrels_dict:
            for vid, score in hits.items():
                # The Prime filter: Only keep the document if it exists in the qrels
                if vid in qrels_dict[tid]:  
                    filtered_run[tid][vid] = score
                    
    metrics_prime = evaluator.evaluate(filtered_run)

    # ---------------------------------------------------------
    # 3. AGGREGATION & DISPLAY
    # ---------------------------------------------------------
    def aggregate_metrics(metrics_dict):
        agg = defaultdict(float)
        n = len(metrics_dict)
        if n == 0: return agg
        for topic_results in metrics_dict.values():
            for metric, value in topic_results.items():
                agg[metric] += value
        return {k: v / n for k, v in agg.items()}

    agg_std = aggregate_metrics(metrics_std)
    agg_prime = aggregate_metrics(metrics_prime)

    print(f"{'Metric':<15} | {'Standard (Hard Mode)':<22} | {'Prime (Filtered)':<20}")
    print("-" * 60)
    
    if "bpref" in agg_std:
        print(f"{'bpref':<15} | {agg_std['bpref']:<22.4f} | {agg_prime.get('bpref', 0):<20.4f}")
        print("-" * 60)

    for k in [5, 10, 100, 1000]:
        for metric_base in ["ndcg_cut", "map_cut", "P"]:
            metric_name = f"{metric_base}_{k}"
            if metric_name in agg_std:
                label = metric_base.replace("_cut", "")
                if label == "ndcg": label = "nDCG"
                elif label == "map": label = "MAP"
                
                label_fmt = f"{label}@{k}"
                std_val = agg_std[metric_name]
                prime_val = agg_prime.get(metric_name, 0.0)
                
                print(f"{label_fmt:<15} | {std_val:<22.4f} | {prime_val:<20.4f}")
        
        if k != 1000:
            print("-" * 60)
    print("=" * 60)

    # --- SAVE METRICS ---
    print(f"\nSaving detailed evaluation metrics to {_OUT_METRICS_PATH.name}...")
    with open(_OUT_METRICS_PATH, "w") as f:
        json.dump({
            "aggregate_standard": agg_std,
            "aggregate_prime": agg_prime,
            "per_topic_standard": metrics_std,
            "per_topic_prime": metrics_prime
        }, f, indent=4)
    print("Success! Metrics saved.")


if __name__ == "__main__":
    main()