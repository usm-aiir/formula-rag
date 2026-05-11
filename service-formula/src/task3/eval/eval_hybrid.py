"""
Hybrid Search Evaluator (Dense + Sparse)

Fuses the Cross-Attention Dual Encoder (FAISS) with the exact-match 
keyword scanner (BM25) using Alpha-weighted Min-Max Normalization.
"""

import sys
import gc
import json
import torch
import faiss
import bm25s
import pytrec_eval
import numpy as np
import re
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.task3.model.formula_encoder import DualFormulaEncoder
from src.task3.dataset import load_topics, load_qrels
from src.data.formula_graph import opt_to_pyg, slt_to_pyg
from torch_geometric.data import Batch

# --- Paths ---
_CHECKPOINT_PATH = _PROJECT_ROOT / "checkpoints/task3/phase3_fusion/phase3_atten_fusion_ensemble_soup.pt"
_FAISS_INDEX_PATH = _PROJECT_ROOT / "checkpoints/task3/faiss_index/phase3_dense.faiss"
_FAISS_IDS_PATH = _PROJECT_ROOT / "checkpoints/task3/faiss_index/phase3_corpus_ids.npy"
_BM25_DIR = _PROJECT_ROOT / "checkpoints/task3/bm25_index"
_OUT_RUN_PATH = _PROJECT_ROOT / "data/processed/phase3_hybrid_run.json"
_PARQUET_DIR = _PROJECT_ROOT / "data/processed/formula_index"

# --- Fusion Parameters ---
ALPHA = 0.80  # 80% Dense GNN Structure, 20% Sparse BM25 Values
TOP_K = 1000
EVAL_SPLIT = "eval"

def min_max_normalize(scores_dict: dict) -> dict:
    """Forces an unbounded dictionary of scores into a 0.0 to 1.0 range."""
    if not scores_dict: return {}
    vals = list(scores_dict.values())
    min_v, max_v = min(vals), max(vals)
    
    if max_v == min_v:
        return {k: 1.0 for k in scores_dict}
        
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

def extract_math_tokens(xml_str: str) -> list:
    """
    Strips XML tags and extracts the actual mathematical text.
    Matches the exact logic used to build the BM25 index!
    """
    if not isinstance(xml_str, str):
        return []
    raw_tokens = re.findall(r'>\s*([^<]+?)\s*<', xml_str)
    return [t.strip() for t in raw_tokens if t.strip()]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*50)
    print("INITIALIZING HYBRID PIPELINE")
    print("="*50)

    # Load Dense Tools (GNN + FAISS)
    print(f"Loading Dual Encoder Soup from {_CHECKPOINT_PATH.name}...")
    encoder = DualFormulaEncoder.load(_CHECKPOINT_PATH, map_location=device).to(device)
    encoder.eval()

    print("Loading FAISS Index...")
    faiss_index = faiss.read_index(str(_FAISS_INDEX_PATH))
    faiss_ids = np.load(str(_FAISS_IDS_PATH))

    # Load Sparse Tools (BM25)
    print("Loading BM25 Sparse Index...")
    bm25_retriever = bm25s.BM25.load(str(_BM25_DIR), load_corpus=True)

    # Load Queries
    print(f"Loading '{EVAL_SPLIT}' queries...")
    topics = load_topics(EVAL_SPLIT)
    qrels = load_qrels(EVAL_SPLIT)
    
    # Use the proxy visual_ids to get the exact query XMLs from the Parquet shards. 
    print("\n--- Retrieving Queries ---")
    proxy_vid_to_topic = {}
    for topic_id in topics.keys():
        if topic_id in qrels:
            positives = [str(vid) for vid, grade in qrels[topic_id].items() if grade >= 2.0] # 0(non-relevant), 1(low), 2(medium), 3(high)
            if positives:
                proxy_vid_to_topic[positives[0]] = topic_id
                
    query_xmls = {}
    shard_files = sorted(list(_PARQUET_DIR.glob("shard_*.parquet")))
    
    for shard in tqdm(shard_files, desc="Locating Query XMLs"):
        df = pq.read_table(shard, columns=["visual_id", "opt", "slt"]).to_pandas()
        df = df.dropna(subset=["opt", "slt"])
        
        # Keep only the rows that match our proxies
        mask = df["visual_id"].astype(str).isin(proxy_vid_to_topic.keys())
        target_df = df[mask]
        
        for _, row in target_df.iterrows():
            vid_str = str(row["visual_id"])
            t_id = proxy_vid_to_topic[vid_str]
            query_xmls[t_id] = {"opt": row["opt"], "slt": row["slt"]}
            
        del df
        gc.collect()
        
        # Stop early if we found them all
        if len(query_xmls) == len(proxy_vid_to_topic):
            break
            
    print(f"Found {len(query_xmls)} valid query definitions.")
    
    run_results = {}

    print("\nRunning Hybrid Inference...")
    for topic_id, xmls in tqdm(query_xmls.items(), desc="Running Inference"):
        
        # Convert XML strings to PyTorch Geometric graphs
        o_g = opt_to_pyg(xmls["opt"])
        s_g = slt_to_pyg(xmls["slt"])
        
        # Skip if parsing completely fails for this query
        if o_g is None or s_g is None:
            continue
        
        # --- DENSE RETRIEVAL (Topology) ---
        with torch.no_grad():
            o_batch = Batch.from_data_list([o_g]).to(device)
            s_batch = Batch.from_data_list([s_g]).to(device)
            q_emb = encoder(o_batch, s_batch, normalize=True).cpu().numpy()
            
        D_dense, I_dense = faiss_index.search(q_emb, TOP_K)
        
        dense_scores = {}
        for score, idx in zip(D_dense[0], I_dense[0]):
            if idx != -1:
                vid = str(faiss_ids[idx])
                dense_scores[vid] = float(score)

        # --- SPARSE RETRIEVAL (Values) ---
        query_tokens = extract_math_tokens(xmls["slt"])
        sparse_scores = {}
        
        if query_tokens:
            docs, D_sparse = bm25_retriever.retrieve([query_tokens], k=TOP_K)
            
            # bm25s returns dictionaries; extract the ID.
            for doc, score in zip(docs[0], D_sparse[0]):
                if isinstance(doc, dict):
                    # bm25s puts the corpus string into the "text" key
                    vid = str(doc.get("text", doc.get("id", "")))
                elif hasattr(doc, "id"):
                    vid = str(doc.id)
                else:
                    vid = str(doc)
                    
                if vid:
                    sparse_scores[vid] = float(score)

        # --- ALPHA FUSION ---
        norm_dense = min_max_normalize(dense_scores)
        norm_sparse = min_max_normalize(sparse_scores)
        
        hybrid_scores = {}
        all_vids = set(norm_dense.keys()) | set(norm_sparse.keys())
        
        for vid in all_vids:
            d_score = norm_dense.get(vid, 0.0)
            s_score = norm_sparse.get(vid, 0.0)
            # Alpha Weighting: 80% Dense, 20% Sparse
            hybrid_scores[vid] = (ALPHA * d_score) + ((1.0 - ALPHA) * s_score)
            
        # Sort and keep the absolute best K results for this topic
        sorted_hybrid = dict(sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K])
        run_results[topic_id] = sorted_hybrid

    # Save Safety Net
    with open(_OUT_RUN_PATH, "w") as f:
        json.dump(run_results, f)
    print(f"\nSaved raw hybrid inference run to {_OUT_RUN_PATH.name}")

    # PyTrec_Eval
    print("\n" + "="*50)
    print(f"TASK 2 EVALUATION (Hybrid: Alpha={ALPHA})")
    print("="*50)
    
    qrels_str = {str(tid): {str(vid): int(score) for vid, score in items.items()} 
                 for tid, items in qrels.items()}
                 
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_str,
        {"ndcg_cut", "map_cut", "P", "bpref"},
        relevance_level=2,
    )
    
    metrics = evaluator.evaluate(run_results)
    
    metrics_agg = defaultdict(float)
    n = len(metrics)
    
    if n == 0:
        print("\nError: No queries were successfully evaluated! Check parsing logic.")
        return
        
    for topic_results in metrics.values():
        for metric, value in topic_results.items():
            metrics_agg[metric] += value
    metrics_agg = {k: v / n for k, v in metrics_agg.items()}

    if "bpref" in metrics_agg:
        print(f"  bpref      {metrics_agg['bpref']:.4f}")
        print(f"{'-'*30}")

    for k in [5, 10, 100, 1000]:
        ndcg_key = f"ndcg_cut_{k}"
        map_key = f"map_cut_{k}"
        p_key = f"P_{k}"
        
        if ndcg_key in metrics_agg:
            print(f"  nDCG@{k:<5} {metrics_agg[ndcg_key]:.4f}")
        if map_key in metrics_agg:
            print(f"  MAP@{k:<6} {metrics_agg[map_key]:.4f}")
        if p_key in metrics_agg:
            print(f"  P@{k:<8} {metrics_agg[p_key]:.4f}")
        
        if k != 1000:
            print(f"{'-'*30}")
            
    print("="*50)

if __name__ == "__main__":
    main()