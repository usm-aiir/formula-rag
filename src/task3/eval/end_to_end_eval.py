"""
End-to-End Mathematical Retrieval Pipeline (Phase 3 + Phase 4)

This module unifies the first-stage Hybrid Retriever and the second-stage 
Structural Re-ranker into a single execution flow
"""

import gc
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pyarrow.parquet as pq
import pytrec_eval
import torch
import faiss
import bm25s
from tqdm import tqdm
from torch_geometric.data import Batch

# Internal project imports
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.task3.model.formula_encoder import DualFormulaEncoder
from src.task3.dataset import load_qrels, load_topics
from src.data.formula_graph import opt_to_pyg, slt_to_pyg

# --- Paths ---
_CHECKPOINT_PATH = _PROJECT_ROOT / "checkpoints/task3/phase3_fusion/phase3_atten_fusion_ensemble_soup.pt"
_FAISS_INDEX_PATH = _PROJECT_ROOT / "checkpoints/task3/faiss_index/phase3_dense.faiss"
_FAISS_IDS_PATH = _PROJECT_ROOT / "checkpoints/task3/faiss_index/phase3_corpus_ids.npy"
_BM25_DIR = _PROJECT_ROOT / "checkpoints/task3/bm25_index"
_PARQUET_DIR = _PROJECT_ROOT / "data/processed/formula_index"

# Final Output Paths
_FINAL_RUN_PATH = _PROJECT_ROOT / "data/processed/end_to_end_run.json"
_FINAL_METRICS_PATH = _PROJECT_ROOT / "data/processed/end_to_end_metrics.json"

# --- Phase 3 Parameters ---
ALPHA = 0.80  
TOP_K = 1000
EVAL_SPLIT = "eval"

# --- Phase 4 Parameters (PB Configuration) ---
MAX_PATH_DEPTH = 4  
K_RRF_DENSE = 60          
K_RRF_STRUCT = 15  
IGNORE_TAGS: Set[str] = {
    'mrow', 'mstyle', 'mpadded', 'mphantom', 'maligngroup', 'malignmark',
    'semantics', 'annotation', 'annotation-xml', 'id', 'mspace', 'menclose', 'maction'
}

# ==========================================
# PHASE 3 HELPER FUNCTIONS
# ==========================================
def min_max_normalize(scores_dict: dict) -> dict:
    if not scores_dict: return {}
    vals = list(scores_dict.values())
    min_v, max_v = min(vals), max(vals)
    if max_v == min_v:
        return {k: 1.0 for k in scores_dict}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

def extract_math_tokens(xml_str: str) -> list:
    if not isinstance(xml_str, str): return []
    raw_tokens = re.findall(r'>\s*([^<]+?)\s*<', xml_str)
    return [t.strip() for t in raw_tokens if t.strip()]

# ==========================================
# PHASE 4 HELPER FUNCTIONS
# ==========================================
def extract_structural_paths(xml_str: str, max_depth: int = MAX_PATH_DEPTH) -> List[str]:
    if not xml_str or not isinstance(xml_str, str): return []
    xml_str = re.sub(r'\sxmlns="[^"]+"', '', xml_str, count=1)
    try:
        root = ET.fromstring(xml_str)
    except Exception:
        return []
        
    def get_shape_string(node: ET.Element, mask_vars: bool = True) -> str:
        tag = node.tag.split('}')[-1]
        if tag in IGNORE_TAGS:
            return "".join(get_shape_string(child, mask_vars) for child in node)
        if mask_vars and tag in {'mi', 'ci', 'identifier'}:
            res = f"<{tag}>V</{tag}>"
        else:
            text = (node.text or "").strip()
            res = f"<{tag}>{text}</{tag}>"
        for child in node: res += get_shape_string(child, mask_vars)
        return res

    def canonicalize_tree(node: ET.Element):
        for child in list(node): canonicalize_tree(child)
        tag = node.tag.split('}')[-1]
        if tag == 'apply' and len(node) > 1:
            op_tag = node[0].tag.split('}')[-1]
            if op_tag in {'plus', 'times', 'eq', 'and', 'or', 'union', 'intersect', 'equivalent'}:
                op_node = node[0]
                args = list(node)[1:]
                args.sort(key=lambda n: (get_shape_string(n, True), get_shape_string(n, False)))
                node[:] = [op_node] + args
        elif tag in {'set', 'list'}:
            args = list(node)
            args.sort(key=lambda n: (get_shape_string(n, True), get_shape_string(n, False)))
            node[:] = args

    canonicalize_tree(root)
        
    paths = []
    var_map, var_counter = {}, 1
    
    def dfs(node: ET.Element, path_exact: List[str], path_alpha: List[str], 
            path_univ: List[str], edge_pos: str = "R"):
        nonlocal var_counter
        tag = node.tag.split('}')[-1]
        
        if tag in IGNORE_TAGS:
            for idx, child in enumerate(node): dfs(child, path_exact, path_alpha, path_univ, edge_pos)
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
            paths.append("E::" + "::".join(pe[-i:]))
            paths.append("E::" + "::".join(pe[-i:])) # DOUBLE WEIGHTING
            paths.append("A::" + "::".join(pa[-i:]))
            paths.append("U::" + "::".join(pu[-i:]))
            
        for idx, child in enumerate(node): dfs(child, pe, pa, pu, str(idx))
            
    dfs(root, [], [], [], "R")
    return paths

def get_depth_weight(path_token: str) -> float:
    return float(len(path_token.split("::")) - 1) ** 1.5

# ==========================================
# END-TO-END EXECUTION
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print("INITIALIZING END-TO-END FORMULA RETRIEVAL PIPELINE")
    print("="*60)

    # ---------------------------------------------------------
    # PART 1: LOAD MODELS & DATA
    # ---------------------------------------------------------
    print(f"Loading Dual Encoder Soup from {_CHECKPOINT_PATH.name}...")
    encoder = DualFormulaEncoder.load(_CHECKPOINT_PATH, map_location=device).to(device)
    encoder.eval()

    print("Loading FAISS Index...")
    faiss_index = faiss.read_index(str(_FAISS_INDEX_PATH))
    faiss_ids = np.load(str(_FAISS_IDS_PATH))

    print("Loading BM25 Sparse Index...")
    bm25_retriever = bm25s.BM25.load(str(_BM25_DIR), load_corpus=True)

    print(f"Loading '{EVAL_SPLIT}' queries...")
    topics = load_topics(EVAL_SPLIT)
    qrels = load_qrels(EVAL_SPLIT)
    
    proxy_vid_to_topic = {}
    for topic_id in topics.keys():
        if topic_id in qrels:
            positives = [str(vid) for vid, grade in qrels[topic_id].items() if grade >= 2.0]
            if positives: 
                proxy_vid_to_topic[positives[0]] = topic_id
                
    query_xmls = {}
    shard_files = sorted(list(_PARQUET_DIR.glob("shard_*.parquet")))
    
    for shard in tqdm(shard_files, desc="Locating Query XMLs"):
        df = pq.read_table(shard, columns=["visual_id", "opt", "slt"]).to_pandas().dropna(subset=["opt", "slt"])
        mask = df["visual_id"].astype(str).isin(proxy_vid_to_topic.keys())
        for _, row in df[mask].iterrows():
            vid_str = str(row["visual_id"])
            t_id = proxy_vid_to_topic[vid_str]
            query_xmls[t_id] = {"opt": row["opt"], "slt": row["slt"]}
        del df
        gc.collect()
        if len(query_xmls) == len(proxy_vid_to_topic): break

    print(f"Found {len(query_xmls)} valid query definitions.")

    # ---------------------------------------------------------
    # PART 2: PHASE 3 - HYBRID INFERENCE
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"PHASE 3: HYBRID SEARCH (Alpha={ALPHA})")
    print("="*60)
    
    hybrid_run = {}
    for topic_id, xmls in tqdm(query_xmls.items(), desc="Extracting Candidates"):
        o_g, s_g = opt_to_pyg(xmls["opt"]), slt_to_pyg(xmls["slt"])
        if o_g is None or s_g is None: continue
        
        with torch.no_grad():
            o_batch, s_batch = Batch.from_data_list([o_g]).to(device), Batch.from_data_list([s_g]).to(device)
            q_emb = encoder(o_batch, s_batch, normalize=True).cpu().numpy()
            
        D_dense, I_dense = faiss_index.search(q_emb, TOP_K)
        dense_scores = {}
        for score, idx in zip(D_dense[0], I_dense[0]):
            if idx != -1:
                dense_scores[str(faiss_ids[idx])] = float(score)

        query_tokens = extract_math_tokens(xmls["slt"])
        sparse_scores = {}
        if query_tokens:
            docs, D_sparse = bm25_retriever.retrieve([query_tokens], k=TOP_K)
            for doc, score in zip(docs[0], D_sparse[0]):
                vid = str(doc.get("text", doc.get("id", ""))) if isinstance(doc, dict) else str(getattr(doc, "id", doc))
                if vid: sparse_scores[vid] = float(score)

        norm_dense, norm_sparse = min_max_normalize(dense_scores), min_max_normalize(sparse_scores)
        hybrid_scores = {}
        for vid in set(norm_dense.keys()) | set(norm_sparse.keys()):
            hybrid_scores[vid] = (ALPHA * norm_dense.get(vid, 0.0)) + ((1.0 - ALPHA) * norm_sparse.get(vid, 0.0))
            
        hybrid_run[topic_id] = dict(sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K])

    # ---------------------------------------------------------
    # PART 3: PREPARE FOR PHASE 4 (Extract Candidate XMLs)
    # ---------------------------------------------------------
    target_vids = set(proxy_vid_to_topic.keys())
    for candidates in hybrid_run.values(): target_vids.update(candidates.keys())

    xml_cache = {}
    for shard in tqdm(shard_files, desc="Caching Candidate XMLs"):
        df = pq.read_table(shard, columns=["visual_id", "opt", "slt"]).to_pandas().dropna(subset=["opt", "slt"])
        mask = df["visual_id"].astype(str).isin(target_vids)
        for _, row in df[mask].iterrows():
            xml_cache[str(row["visual_id"])] = {"opt": row["opt"], "slt": row["slt"]}
        del df
        gc.collect()
        if len(xml_cache) >= len(target_vids): break

    # ---------------------------------------------------------
    # PART 4: PHASE 4 - STRUCTURAL RE-RANKING
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 4: SOTA STRUCTURAL RE-RANKER")
    print("="*60)
    
    structural_run = {}
    for topic_id, candidates in tqdm(hybrid_run.items(), desc="Re-ranking"):
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
                d_tokens = extract_structural_paths(xml_cache[vid]["opt"]) + extract_structural_paths(xml_cache[vid]["slt"])
                if d_tokens:
                    doc_tokens_list.append(d_tokens)
                    valid_doc_vids.append(vid)
                    
        if not valid_doc_vids:
            structural_run[topic_id] = candidates
            continue
        
        n_docs = len(valid_doc_vids)
        df_counts = defaultdict(int)
        for tokens in doc_tokens_list:
            for t in set(tokens): df_counts[t] += 1
        idf = {t: math.log(1.0 + (n_docs - f + 0.5) / (f + 0.5)) for t, f in df_counts.items()}
        
        q_counts = Counter(q_tokens)
        q_total_tokens = sum(q_counts.values())
        max_q_score = sum(count * idf.get(t, 1.0) * get_depth_weight(t) for t, count in q_counts.items()) or 1.0
        
        structural_scores = {}
        for vid, d_tokens in zip(valid_doc_vids, doc_tokens_list):
            d_counts = Counter(d_tokens)
            d_total_tokens = sum(d_counts.values())
            
            overlap_score = 0.0
            for t, count in q_counts.items():
                matched = min(count, d_counts.get(t, 0))
                overlap_score += matched * idf.get(t, 1.0) * get_depth_weight(t)
                
            length_ratio = q_total_tokens / max(1.0, d_total_tokens)
            length_penalty = min(1.0, length_ratio) ** 0.25
            
            structural_scores[vid] = (overlap_score / max_q_score) * length_penalty
        
        struct_ranked = sorted(structural_scores.keys(), key=lambda v: structural_scores[v], reverse=True)
        dense_ranked = list(candidates.keys()) 
        
        fused_scores = defaultdict(float)
        for rank, vid in enumerate(dense_ranked): fused_scores[vid] += 1.0 / (K_RRF_DENSE + rank + 1)
        for rank, vid in enumerate(struct_ranked): fused_scores[vid] += 3.0 / (K_RRF_STRUCT + rank + 1)  
            
        structural_run[topic_id] = dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))

    with open(_FINAL_RUN_PATH, "w") as f:
        json.dump(structural_run, f)

    # ---------------------------------------------------------
    # PART 5: EVALUATION (Standard vs. Prime)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("TASK 2 EVALUATION (Standard vs. Prime Metrics)")
    print("="*60)
    
    qrels_dict = {str(tid): {str(vid): int(score) for vid, score in items.items()} for tid, items in qrels.items()}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {"ndcg_cut", "map_cut", "P", "bpref"}, relevance_level=2)
    
    metrics_std = evaluator.evaluate(structural_run)
    
    filtered_run = {}
    for tid, hits in structural_run.items():
        filtered_run[tid] = {}
        if tid in qrels_dict:
            for vid, score in hits.items():
                if vid in qrels_dict[tid]:  
                    filtered_run[tid][vid] = score
                    
    metrics_prime = evaluator.evaluate(filtered_run)

    def aggregate_metrics(metrics_dict):
        agg = defaultdict(float)
        n = len(metrics_dict)
        if n == 0: return agg
        for topic_results in metrics_dict.values():
            for metric, value in topic_results.items(): agg[metric] += value
        return {k: v / n for k, v in agg.items()}

    agg_std, agg_prime = aggregate_metrics(metrics_std), aggregate_metrics(metrics_prime)

    print(f"{'Metric':<15} | {'Standard (Hard Mode)':<22} | {'Prime (Filtered)':<20}")
    print("-" * 60)
    
    if "bpref" in agg_std:
        print(f"{'bpref':<15} | {agg_std['bpref']:<22.4f} | {agg_prime.get('bpref', 0):<20.4f}")
        print("-" * 60)

    for k in [5, 10, 100, 1000]:
        for metric_base in ["ndcg_cut", "map_cut", "P"]:
            metric_name = f"{metric_base}_{k}"
            if metric_name in agg_std:
                label = "nDCG" if metric_base == "ndcg_cut" else "MAP" if metric_base == "map_cut" else "P"
                print(f"{f'{label}@{k}':<15} | {agg_std[metric_name]:<22.4f} | {agg_prime.get(metric_name, 0.0):<20.4f}")
        if k != 1000: print("-" * 60)
    print("=" * 60)

    print(f"\nSaving detailed evaluation metrics to {_FINAL_METRICS_PATH.name}...")
    with open(_FINAL_METRICS_PATH, "w") as f:
        json.dump({
            "aggregate_standard": agg_std,
            "aggregate_prime": agg_prime,
            "per_topic_standard": metrics_std,
            "per_topic_prime": metrics_prime
        }, f, indent=4)
    print("Success! Pipeline execution complete.")

if __name__ == "__main__":
    main()