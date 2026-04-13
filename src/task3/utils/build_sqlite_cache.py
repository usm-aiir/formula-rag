"""
Formula XML SQLite Cache Builder

Converts the analytical Parquet shards into a highly-indexed SQLite database
for instantaneous Key-Value lookups during live Retrieval and Re-ranking.

(XML lookups are a major bottleneck during inference, and this caching layer reduces retrieval time massively)
"""

import sys
import sqlite3
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

_PARQUET_DIR = _PROJECT_ROOT / "data/processed/formula_index"
_DB_PATH = _PROJECT_ROOT / "data/processed/formula_cache.db"

def build_cache():
    print("\n" + "="*50)
    print("BUILDING SQLITE XML CACHE")
    print("="*50)

    shard_files = sorted(list(_PARQUET_DIR.glob("shard_*.parquet")))
    if not shard_files:
        print(f"Error: No Parquet shards found in {_PARQUET_DIR}")
        return

    # Initialize SQLite Database
    print(f"Initializing database at {_DB_PATH.name}...")
    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()

    # Create table with visual_id as the Primary Key (automatically indexes it)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS formulas (
            visual_id TEXT PRIMARY KEY,
            opt TEXT,
            slt TEXT
        )
    ''')
    
    # Speed up insertions
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA journal_mode = MEMORY')

    # Extract and Insert Data
    total_inserted = 0
    
    for shard in tqdm(shard_files, desc="Migrating Shards to SQLite"):
        # Load shard
        df = pq.read_table(shard, columns=["visual_id", "opt", "slt"]).to_pandas()
        df = df.dropna(subset=["opt", "slt"])
        
        # Convert to list of tuples for fast insertion
        records = df[['visual_id', 'opt', 'slt']].astype(str).values.tolist()
        
        # Insert using executemany
        cursor.executemany('''
            INSERT OR IGNORE INTO formulas (visual_id, opt, slt) 
            VALUES (?, ?, ?)
        ''', records)
        
        total_inserted += len(records)
        conn.commit()

    conn.close()
    print("\n" + "="*50)
    print(f"Success! {total_inserted:,} formulas cached.")
    print(f"Live engine will now fetch XMLs in milliseconds.")
    print("="*50)

if __name__ == "__main__":
    build_cache()