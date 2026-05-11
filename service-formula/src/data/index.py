"""
Build the formula index Parquet shards from the three ARQMath TSV families.

The three TSV families (latex_representation_v3, opt_representation_v3,
slt_representation_v3) share identical row IDs and matching filenames
(1.tsv through 101.tsv).  This script streams them in lockstep, joins each
row on the shared `id` field, and writes one Parquet shard per file triplet.

Output
------
data/processed/formula_index/shard_001.parquet  …  shard_101.parquet

Usage
-----
    python -m src.data.index [--force]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMULAS_DIR = REPO_ROOT / "data/raw/arqmath/formulas"
LATEX_DIR = FORMULAS_DIR / "latex_representation_v3"
OPT_DIR   = FORMULAS_DIR / "opt_representation_v3"
SLT_DIR   = FORMULAS_DIR / "slt_representation_v3"
OUT_DIR   = REPO_ROOT / "data/processed/formula_index"

SCHEMA = pa.schema([
    pa.field("id",            pa.int64()),
    pa.field("post_id",       pa.int64()),
    pa.field("type",          pa.string()),
    pa.field("old_visual_id", pa.string()),
    pa.field("visual_id",     pa.string()),
    pa.field("latex",         pa.string()),
    pa.field("opt",           pa.large_string()),
    pa.field("slt",           pa.large_string()),
])


def _sorted_tsv_files(directory: Path) -> list:
    """Return TSV files sorted numerically by stem (1 < 2 < … < 101)."""
    return sorted(directory.glob("*.tsv"), key=lambda p: int(p.stem))


def _formula_value(row: list):
    """Return the formula string, or None if the row has an issue flag."""
    issue = row[7].strip()
    formula = row[8].strip() if len(row) > 8 else ""
    if issue:
        return None
    return formula or None


def build_shard(
    latex_path: Path,
    opt_path: Path,
    slt_path: Path,
    out_path: Path,
) -> int:
    """Stream one TSV file triplet in lockstep and write a Parquet shard."""
    ids, post_ids, types, old_vids, vids, latexes, opts, slts = [], [], [], [], [], [], [], []

    f_l = latex_path.open(newline="", encoding="utf-8")
    f_o = opt_path.open(newline="", encoding="utf-8")
    f_s = slt_path.open(newline="", encoding="utf-8")
    try:
        reader_l = csv.reader(f_l, delimiter="\t")
        reader_o = csv.reader(f_o, delimiter="\t")
        reader_s = csv.reader(f_s, delimiter="\t")
        next(reader_l)
        next(reader_o)
        next(reader_s)

        for row_l, row_o, row_s in zip(reader_l, reader_o, reader_s):
            lid, oid, sid = int(row_l[0]), int(row_o[0]), int(row_s[0])
            if lid != oid or lid != sid:
                raise ValueError(f"Row ID mismatch in {latex_path.name}: latex={lid}, opt={oid}, slt={sid}")
            ids.append(lid)
            post_ids.append(int(row_l[1]))
            types.append(row_l[3])
            old_vids.append(row_l[5])
            vids.append(row_l[6])
            latexes.append(row_l[8].strip())
            opts.append(_formula_value(row_o))
            slts.append(_formula_value(row_s))
    finally:
        f_l.close()
        f_o.close()
        f_s.close()

    table = pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "post_id": pa.array(post_ids, type=pa.int64()),
            "type": pa.array(types, type=pa.string()),
            "old_visual_id": pa.array(old_vids, type=pa.string()),
            "visual_id": pa.array(vids, type=pa.string()),
            "latex": pa.array(latexes, type=pa.string()),
            "opt": pa.array(opts, type=pa.large_string()),
            "slt": pa.array(slts, type=pa.large_string()),
        },
        schema=SCHEMA,
    )
    pq.write_table(table, out_path, compression="snappy")
    return len(ids)


def main(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latex_files = _sorted_tsv_files(LATEX_DIR)
    opt_files = _sorted_tsv_files(OPT_DIR)
    slt_files = _sorted_tsv_files(SLT_DIR)

    if not latex_files:
        print(f"ERROR: No TSV files found in {LATEX_DIR}")
        print("Run: bash scripts/setup.sh")
        sys.exit(1)

    if len(latex_files) != len(opt_files) or len(latex_files) != len(slt_files):
        print(f"WARNING: TSV file count mismatch — latex={len(latex_files)}, opt={len(opt_files)}, slt={len(slt_files)}")

    n_files = min(len(latex_files), len(opt_files), len(slt_files))
    total_rows, skipped = 0, 0

    for latex_f, opt_f, slt_f in tqdm(
        zip(latex_files[:n_files], opt_files[:n_files], slt_files[:n_files]),
        total=n_files,
        desc="Building formula index",
        unit="shard",
    ):
        stem = int(latex_f.stem)
        out_path = OUT_DIR / f"shard_{stem:03d}.parquet"
        if out_path.exists() and not force:
            total_rows += pq.read_metadata(out_path).num_rows
            skipped += 1
            continue
        total_rows += build_shard(latex_f, opt_f, slt_f, out_path)

    print(f"\nFormula index complete.\n  Shards written: {n_files - skipped}\n  Shards skipped: {skipped}\n  Total rows: {total_rows:,}\n  Output: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build formula index Parquet shards from ARQMath TSVs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing shards")
    main(force=parser.parse_args().force)
