"""
Dual-Modality Hard Negative Mining (Memory Safe Streaming).

This script mines hard negatives for the Phase 3 dual-encoder architecture. It should be run after Phase 2 (OPT-based Encoder), 
which produces the checkpoint used here. 
The output is a JSONL file containing query-positive-hard negative triplets, where each entry has the following format:
{
    "topic_id": "123",
    "query_opt": "...",
    "query_slt": "...",
    "pos_opt": "...",
    "pos_slt": "...",
    "hard_neg_opts": ["...", "...", ...],
    "hard_neg_slts": ["...", "...", ...]
}
"""

import json
import sys
import gc
from pathlib import Path
from tqdm import tqdm
import torch
import faiss
import numpy as np
import pyarrow.parquet as pq
from torch_geometric.data import Batch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.task3.model.formula_encoder import FormulaEncoder
from src.task3.dataset import load_topics, load_qrels
from src.data.formula_graph import opt_to_pyg

_TRAIN_SPLIT = "train"
_OUTPUT_FILE = _PROJECT_ROOT / "data/processed/phase3_hard_negatives.jsonl"
_PARQUET_DIR = _PROJECT_ROOT / "data/processed/formula_index"


def mine_phase3_hard_negatives_streaming(checkpoint_path: str, top_k: int = 50, device_str: str = "cuda"):
    device = torch.device(device_str)
    print(f"Loading Phase 2 Model from {checkpoint_path} on {device}...", flush=True)
    encoder = FormulaEncoder.load(checkpoint_path, map_location=device).to(device)
    encoder.eval()

    topics = load_topics(_TRAIN_SPLIT)
    qrels = load_qrels(_TRAIN_SPLIT)
    shard_files = sorted(list(_PARQUET_DIR.glob("shard_*.parquet")))

    # Identify one Proxy Visual ID per query to grab its XML later
    proxy_vid_to_topic = {}
    for topic_id in topics.keys():
        if topic_id in qrels:
            positives = [str(vid) for vid, grade in qrels[topic_id].items() if grade >= 2.0]
            if positives:
                proxy_vid_to_topic[positives[0]] = topic_id
                
    query_xmls = {} # Stores {"topic_id": {"opt": "...", "slt": "..."}}

    # ==========================================================
    # PASS 1: Streaming FAISS Build & Proxy Extraction
    # ==========================================================
    print("\n--- PASS 1: Streaming Encoder & FAISS Build ---", flush=True)
    emb_dim = encoder.output_dim
    faiss_index = faiss.IndexFlatIP(emb_dim)
    corpus_ids = []

    for shard in tqdm(shard_files, desc="Encoding Shards (RAM Safe)"):
        # Stream just the columns we need for this pass
        df = pq.read_table(shard, columns=["visual_id", "opt", "slt"]).to_pandas()
        df = df.dropna(subset=["opt", "slt"])

        # Batch encode to keep memory fragmented and small
        batch_size = 512
        shard_embs = []
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            
            valid_graphs = []
            valid_ids = []
            
            for _, row in batch_df.iterrows():
                vid_str = str(row["visual_id"])
                
                # If this is a proxy for a query, save its XMLs!
                if vid_str in proxy_vid_to_topic:
                    t_id = proxy_vid_to_topic[vid_str]
                    query_xmls[t_id] = {"opt": row["opt"], "slt": row["slt"]}
                
                # Parse graph
                g = opt_to_pyg(row["opt"])
                if g is not None:
                    valid_graphs.append(g)
                    valid_ids.append(vid_str)
                    
            if not valid_graphs:
                continue

            batch = Batch.from_data_list(valid_graphs).to(device)
            with torch.no_grad():
                embs = encoder(batch, normalize=True)[0].cpu().numpy()
                
            shard_embs.append(embs)
            corpus_ids.extend(valid_ids)

        if shard_embs:
            faiss_index.add(np.vstack(shard_embs))

        # Explicitly free memory before loading the next shard
        del df
        del shard_embs
        gc.collect()

    print(f"FAISS Index built! {faiss_index.ntotal:,} vectors encoded.", flush=True)
    corpus_ids_np = np.array(corpus_ids)

    # ==========================================================
    # PASS 2: Search & Identify Needed Target IDs
    # ==========================================================
    print("\n--- PASS 2: Retrieving Hard Negatives ---", flush=True)
    needed_vids = set()
    topic_hard_negatives = {}
    
    for topic_id in tqdm(query_xmls.keys(), desc="Searching FAISS"):
        q_opt = query_xmls[topic_id]["opt"]
        graph = opt_to_pyg(q_opt)
        if not graph:
            continue
            
        batch = Batch.from_data_list([graph]).to(device)
        with torch.no_grad():
            q_emb = encoder(batch, normalize=True)[0].cpu().numpy().reshape(1, -1)
            
        # KNN Search
        scores, indices = faiss_index.search(q_emb, top_k)
        
        hn_ids = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1: continue
            res_id = corpus_ids_np[idx]
            
            if res_id not in qrels[topic_id]:
                hn_ids.append(res_id)
                needed_vids.add(res_id) # Flag this ID for extraction
                
            if len(hn_ids) >= 5:
                break
                
        topic_hard_negatives[topic_id] = hn_ids
        
        # Flag all True Positives for extraction
        for vid, grade in qrels[topic_id].items():
            if grade >= 2.0: # 0 (Non-relevant), 1 (low), 2 (medium), 3 (high)
                needed_vids.add(str(vid))

    print(f"Identified {len(needed_vids)} unique target IDs for extraction.", flush=True)

    # ==========================================================
    # PASS 3: Targeted Dual-Modality Extraction
    # ==========================================================
    print("\n--- PASS 3: Extracting Target XMLs ---", flush=True)
    final_dict = {}
    
    for shard in tqdm(shard_files, desc="Streaming Extraction"):
        df = pq.read_table(shard, columns=["visual_id", "opt", "slt"]).to_pandas()

        df = df.dropna(subset=["opt", "slt"])
        
        # Filter down to ONLY the rows we care about to save RAM
        df["visual_id"] = df["visual_id"].astype(str)
        mask = df["visual_id"].isin(needed_vids)
        target_df = df[mask]
        
        for _, row in target_df.iterrows():
            final_dict[row["visual_id"]] = {
                "opt": row["opt"],
                "slt": row["slt"]
            }
            
        del df
        gc.collect()

    # ==========================================================
    # PASS 4: Assemble and Write JSONL
    # ==========================================================
    print("\n--- PASS 4: Writing Triplets ---", flush=True)
    triplets_mined = 0
    with open(_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for topic_id, hn_ids in topic_hard_negatives.items():
            if topic_id not in query_xmls: continue
            
            positives = [str(vid) for vid, grade in qrels[topic_id].items() if grade >= 2.0]
            
            for pos_id in positives:
                if pos_id in final_dict:
                    # Validate that we successfully extracted all Hard Negatives
                    valid_hns = [hn for hn in hn_ids if hn in final_dict]
                    if not valid_hns: continue
                    
                    record = {
                        "topic_id": topic_id,
                        "query_opt": query_xmls[topic_id]["opt"],
                        "query_slt": query_xmls[topic_id]["slt"],
                        "pos_opt": final_dict[pos_id]["opt"],
                        "pos_slt": final_dict[pos_id]["slt"],
                        "hard_neg_opts": [final_dict[hn]["opt"] for hn in valid_hns],
                        "hard_neg_slts": [final_dict[hn]["slt"] for hn in valid_hns]
                    }
                    f.write(json.dumps(record) + "\n")
                    triplets_mined += 1
                    
    print(f"\nSUCCESS: Mined {triplets_mined} Dual-Modality Hard Negative Triplets!")
    print(f"Saved to: {_OUTPUT_FILE}")

if __name__ == "__main__":
    mine_phase3_hard_negatives_streaming(
        checkpoint_path=str(_PROJECT_ROOT / "checkpoints/task3/phase2_best.pt")
    )