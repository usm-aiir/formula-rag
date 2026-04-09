"""
BM25 Sparse Index Builder

Extracts literal math tokens (numbers, variables) from Presentation MathML (SLT) 
and builds a highly optimized BM25 sparse index to complement the dense Dual Encoder.
"""

import sys
import re
import gc
import bm25s
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

_PARQUET_DIR = _PROJECT_ROOT / "data/processed/formula_index"
_OUT_DIR = _PROJECT_ROOT / "checkpoints/task3/bm25_index"

def extract_math_tokens(xml_str: str) -> List[str]:
    """
    Strips XML tags and extracts the actual mathematical text.
    Example: "<mi>x</mi><mn>2</mn>" -> ["x", "2"]
    """
    if not isinstance(xml_str, str):
        return []
        
    # Regex to capture anything between > and <
    # e.g., <mi>x</mi> captures 'x'
    raw_tokens = re.findall(r'>\s*([^<]+?)\s*<', xml_str)
    
    # Clean up whitespace and drop empty strings
    clean_tokens = [t.strip() for t in raw_tokens if t.strip()]
    return clean_tokens

def build_sparse_index():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(list(_PARQUET_DIR.glob("shard_*.parquet")))
    
    if not shard_files:
        print("Error: No parquet shards found!")
        return

    print(f"Found {len(shard_files)} shards. Extracting tokens...")
    
    corpus_tokens = []
    corpus_ids = []
    
    # Stream the shards and tokenize
    for shard in tqdm(shard_files, desc="Tokenizing MathML"):
        df = pq.read_table(shard, columns=["visual_id", "slt"]).to_pandas()
        df = df.dropna(subset=["slt"])
        
        for _, row in df.iterrows():
            tokens = extract_math_tokens(row["slt"])
            if tokens:  # Only add if it has actual tokens
                corpus_tokens.append(tokens)
                corpus_ids.append(str(row["visual_id"]))
                
        del df
        gc.collect()

    print(f"\nExtracted {len(corpus_tokens):,} valid tokenized formulas.")
    print("Building BM25 Index (This will take a few minutes and max out CPU)...")
    
    # Build the BM25 Index
    # We use 'l2' method which is standard for most retrieval tasks
    retriever = bm25s.BM25(method="robertson", k1=1.5, b=0.75)
    
    # bm25s requires a list of lists of strings
    retriever.index(corpus_tokens)
    
    # Save the index and the corresponding IDs
    print("\nSaving BM25 Index to disk...")
    retriever.save(str(_OUT_DIR), corpus=corpus_ids)
    
    print(f"Success! BM25 Index saved to {_OUT_DIR.name}")

if __name__ == "__main__":
    build_sparse_index()