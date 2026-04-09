"""
Dense FAISS Index Builder

Encodes the 8.3M Dual-Modality corpus using the Cross-Attention Dual Encoder 
and permanently saves the FAISS index and ID mappings to disk.
"""

import sys
import gc
import faiss
import torch
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Batch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.task3.model.formula_encoder import DualFormulaEncoder
from src.data.formula_graph import opt_to_pyg, slt_to_pyg

_PARQUET_DIR = _PROJECT_ROOT / "data/processed/formula_index"
_OUT_DIR = _PROJECT_ROOT / "checkpoints/task3/faiss_index"

# Point this to the trained model soup checkpoint
_CHECKPOINT_PATH = _PROJECT_ROOT / "checkpoints/task3/phase3_fusion/phase3_atten_fusion_ensemble_soup.pt"

def build_dense_index():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building FAISS Index on {device}...")

    # Load the Cross-Attention Model Soup
    print(f"Loading Dual Encoder from {_CHECKPOINT_PATH.name}...")
    encoder = DualFormulaEncoder.load(_CHECKPOINT_PATH, map_location=device).to(device)
    encoder.eval()

    # Initialize FAISS Index (Inner Product for Cosine Similarity)
    emb_dim = encoder.output_dim
    faiss_index = faiss.IndexFlatIP(emb_dim)
    corpus_ids = []
    
    shard_files = sorted(list(_PARQUET_DIR.glob("shard_*.parquet")))
    print(f"\nFound {len(shard_files)} shards. Starting streaming encoding...")

    # Stream and Encode
    for shard in tqdm(shard_files, desc="Encoding & Indexing (RAM Safe)"):
        df = pq.read_table(shard, columns=["visual_id", "opt", "slt"]).to_pandas()
        df = df.dropna(subset=["opt", "slt"])

        batch_size = 512
        shard_embs = []
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            valid_q_opts, valid_q_slts, valid_ids = [], [], []
            
            for _, row in batch_df.iterrows():
                o_g = opt_to_pyg(row["opt"])
                s_g = slt_to_pyg(row["slt"])
                
                if o_g is not None and s_g is not None:
                    valid_q_opts.append(o_g)
                    valid_q_slts.append(s_g)
                    valid_ids.append(str(row["visual_id"]))
                    
            if not valid_q_opts:
                continue

            o_batch = Batch.from_data_list(valid_q_opts).to(device)
            s_batch = Batch.from_data_list(valid_q_slts).to(device)
            
            with torch.no_grad():
                # L2 Normalize is built into the forward pass
                embs = encoder(o_batch, s_batch, normalize=True).cpu().numpy()
                
            shard_embs.append(embs)
            corpus_ids.extend(valid_ids)

        if shard_embs:
            faiss_index.add(np.vstack(shard_embs))

        del df
        del shard_embs
        gc.collect()

    print(f"\nFAISS Index built successfully! Total vectors: {faiss_index.ntotal:,}")

    # Save to Disk
    faiss_path = _OUT_DIR / "phase3_dense.faiss"
    ids_path = _OUT_DIR / "phase3_corpus_ids.npy"
    
    print(f"Saving FAISS index to {faiss_path.name}...")
    faiss.write_index(faiss_index, str(faiss_path))
    
    print(f"Saving Corpus IDs to {ids_path.name}...")
    np.save(str(ids_path), np.array(corpus_ids))
    
    print("\n" + "="*50)
    print("Dense Index Saved! Ready for Hybrid Search.")
    print("="*50)

if __name__ == "__main__":
    build_dense_index()