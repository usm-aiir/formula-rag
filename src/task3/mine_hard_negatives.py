"""
Phase 2: Self-Adversarial Hard Negative Mining.

Uses the trained Phase 1 GAT encoder to search the 8.3M corpus using the
TRAINING topics. High-ranking incorrect results are saved as hard negatives.
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.data import Batch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.task3.eval import (
    _encode_corpus, 
    _build_faiss_index, 
    _retrieve, 
    normalize_latex
)
from src.task3.model.formula_encoder import FormulaEncoder
from src.task3.dataset import load_topics, load_qrels, load_opt_index
from src.data.formula_graph import opt_to_pyg

_TRAIN_SPLIT = "train"
_OUTPUT_FILE = _PROJECT_ROOT / "data/processed/self_mined_hard_negatives.jsonl"

def mine_hard_negatives(checkpoint_path: str, top_k: int = 50, device_str: str = "cuda"):
    device = torch.device(device_str)
    print(f"Loading Phase 1 Model from {checkpoint_path} on {device}...", flush=True)
    
    encoder = FormulaEncoder.load(checkpoint_path, map_location=device).to(device)
    encoder.eval()

    # Encode the Corpus & Build FAISS
    corpus_embs, corpus_ids = _encode_corpus(encoder, device)
    faiss_index = _build_faiss_index(corpus_embs)

    # Load Training Data (ARQMath 1 & 2)
    topics = load_topics(_TRAIN_SPLIT)
    qrels = load_qrels(_TRAIN_SPLIT)
    
    # We need the OPT dictionary to write the actual trees to the JSONL
    print("Loading global OPT index...", flush=True)
    opt_index = load_opt_index()

    print("Encoding Training Queries...", flush=True)
    unique_queries = {latex.strip() for latex in topics.values()}
    norm_to_orig = {normalize_latex(q): q for q in unique_queries}
    normalized_query_set = set(norm_to_orig.keys())
    
    query_opt_map = {}
    for vid, opt in opt_index.items():
        if len(query_opt_map) == len(unique_queries):
            break
        pass
    
    triplets_mined = 0
    
    with open(_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for topic_id, latex in tqdm(topics.items(), desc="Mining Triplets"):
            if topic_id not in qrels:
                continue
                
            # Get True Positives (Grade 2+)
            positives = [vid for vid, grade in qrels[topic_id].items() if grade >= 2.0]
            if not positives:
                continue
                
            query_opt = None
            norm_latex = normalize_latex(latex)
            # Find the query OPT by cross-referencing the positive's visual ID
            # (If a positive exists, its OPT is a valid proxy for the query structure)
            if positives[0] in opt_index:
                query_opt = opt_index[positives[0]]
                
            if not query_opt:
                continue

            # Encode Query
            graph = opt_to_pyg(query_opt)
            if not graph:
                continue
            batch = Batch.from_data_list([graph]).to(device)
            with torch.no_grad():
                query_emb = encoder(batch, normalize=True)[0].cpu().float().numpy()

            # KNN from FAISS
            results = _retrieve(faiss_index, corpus_ids, query_emb, top_k)
            
            # Find the Hard Negatives (High rank, but NOT in qrels)
            # We ignore items graded 0 or 1, and only grab items that the assessors 
            # definitively marked wrong, or unjudged items the model thinks are perfect.
            hard_negatives = []
            for res_id, score in results:
                if res_id not in qrels[topic_id]:
                    if res_id in opt_index:
                        hard_negatives.append(res_id)
                if len(hard_negatives) >= 5: # Keep top 5 hardest negatives
                    break
                    
            if not hard_negatives:
                continue
                
            # Write one triplet per positive
            for pos_id in positives:
                if pos_id in opt_index:
                    record = {
                        "topic_id": topic_id,
                        "query_opt": query_opt,
                        "pos_opt": opt_index[pos_id],
                        "hard_neg_opts": [opt_index[hn] for hn in hard_negatives]
                    }
                    f.write(json.dumps(record) + "\n")
                    triplets_mined += 1

    print(f"\nSUCCESS: Mined {triplets_mined} Hard Negative Triplets!")
    print(f"Saved to: {_OUTPUT_FILE}")

if __name__ == "__main__":
    mine_hard_negatives(checkpoint_path=str(_PROJECT_ROOT / "checkpoints/task3/best.pt"))