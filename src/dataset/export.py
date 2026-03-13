"""
Export the assembled trimodal ARQMath dataset to HuggingFace format.

Loads the sharded Parquet files produced by 03_assemble_dataset.py,
re-encodes rendered formula images and post images as HuggingFace Image
features (embedding the actual bytes so the dataset is self-contained),
and either saves the result locally as an Arrow dataset or pushes it to
the HuggingFace Hub.

The formula index Parquet shards (formula_index/*.parquet) are uploaded as
a companion dataset containing OPT and SLT MathML representations — they
are referenced by formula_id in the main dataset.

Usage
-----
    # Save locally only
    python scripts/dataset/04_export_hf.py --output-dir data/hf_dataset

    # Push to HuggingFace Hub (requires HF_TOKEN env var or huggingface-cli login)
    python scripts/dataset/04_export_hf.py --push-to-hub your-username/arqmath-trimodal
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data/processed/dataset"
FORMULA_INDEX_DIR = REPO_ROOT / "data/processed/formula_index"


DATASET_CARD = """\
---
license: cc-by-4.0
task_categories:
  - question-answering
  - feature-extraction
language:
  - en
tags:
  - mathematics
  - formula-retrieval
  - multimodal
  - rag
  - arqmath
pretty_name: ARQMath Trimodal
size_categories:
  - 1M<n<10M
---

# ARQMath Trimodal

An image-modality expanded version of the ARQMath Math Stack Exchange corpus,
treating mathematical formulas as a first-class modality alongside text and images.

## Dataset Description

**Source:** Math Stack Exchange posts from 2010–2018, as distributed in the
[ARQMath](https://www.cs.rit.edu/~dprl/ARQMath/) benchmark collection.

**Trimodal structure per post:**

| Modality | Field | Description |
|---|---|---|
| Text | `body_text` | HTML-stripped post body; formula positions marked `[FORMULA_i]` |
| Formula | `formulas[].latex` | LaTeX string |
| Formula | `formulas[].formula_id` | `old_visual_id` — stable ID used in ARQMath qrels |
| Formula | `formulas[].rendered_typeset` | Typeset PNG rendered via matplotlib mathtext |
| Formula | `formulas[].rendered_graph` | Function graph PNG (where formula is plottable) |
| Image | `post_images` | Paths to images embedded in the original post |

## Companion Artifact

`formula_index/*.parquet` — one shard per source TSV file, containing full
OPT (Content MathML) and SLT (Presentation MathML) representations joinable
by `formula_id`.  Load with:

```python
import polars as pl
formula_index = pl.scan_parquet("formula_index/*.parquet")
```

## Usage

```python
from datasets import load_dataset

ds = load_dataset("your-username/arqmath-trimodal")

# Access a post
post = ds["train"][0]
print(post["body_text"])
for formula in post["formulas"]:
    print(formula["latex"])
    formula["rendered_typeset"].show()   # PIL Image
```

## Citation

If you use this dataset, please cite the original ARQMath collection:

```
@inproceedings{mansouri2022arqmath,
  title={ARQMath-3 (2022 CLEF Lab on Answer Retrieval for Questions on Math)},
  author={Mansouri, Behrooz and Zanibbi, Richard and Oard, Douglas W and Agarwal, Anurag},
  booktitle={CLEF},
  year={2022}
}
```
"""


def _load_dataset_info() -> dict:
    info_path = DATASET_DIR / "dataset_info.json"
    if info_path.exists():
        with info_path.open() as f:
            return json.load(f)
    return {}


def build_hf_dataset(output_dir: Path) -> None:
    """Load assembled Parquet shards and save as a HuggingFace dataset."""
    try:
        from datasets import Dataset, Features, Image, Sequence, Value
    except ImportError:
        print("ERROR: `datasets` library not installed.")
        print("Run: pip install datasets")
        sys.exit(1)

    shards = sorted(DATASET_DIR.glob("shard_*.parquet"))
    if not shards:
        print(f"ERROR: No Parquet shards found in {DATASET_DIR}")
        print("Run: python scripts/dataset/03_assemble_dataset.py")
        sys.exit(1)

    print(f"Loading {len(shards)} Parquet shards...")

    # Load with datasets directly from Parquet — Images are resolved lazily
    # as file paths during the cast step.
    ds = Dataset.from_parquet([str(s) for s in shards])

    print(f"  {len(ds):,} posts loaded")
    print("Casting image columns...")

    # The rendered_typeset / rendered_graph columns are relative path strings.
    # We resolve them to absolute paths before casting so HuggingFace can read
    # the files when the dataset is saved / pushed.
    def _resolve_image_paths(batch: dict) -> dict:
        def _abs(rel: str | None) -> str | None:
            if rel is None:
                return None
            p = REPO_ROOT / rel
            return str(p) if p.exists() else None

        resolved_formulas = []
        for formula_list in batch["formulas"]:
            resolved_formulas.append([
                {
                    **f,
                    "rendered_typeset": _abs(f.get("rendered_typeset")),
                    "rendered_graph":   _abs(f.get("rendered_graph")),
                }
                for f in (formula_list or [])
            ])
        batch["formulas"] = resolved_formulas

        resolved_images = []
        for img_list in batch["post_images"]:
            resolved_images.append([
                str(REPO_ROOT / p) for p in (img_list or [])
                if (REPO_ROOT / p).exists()
            ])
        batch["post_images"] = resolved_images

        return batch

    ds = ds.map(_resolve_image_paths, batched=True, batch_size=1000, desc="Resolving paths")

    # Define features with Image type so HuggingFace embeds the PNG bytes
    features = Features({
        "post_id":               Value("int64"),
        "post_type":             Value("string"),
        "parent_id":             Value("int64"),
        "accepted_answer_id":    Value("int64"),
        "title":                 Value("string"),
        "tags":                  Sequence(Value("string")),
        "score":                 Value("int64"),
        "creation_date":         Value("string"),
        "body_text":             Value("string"),
        "formulas": Sequence({
            "formula_id":        Value("string"),
            "latex":             Value("string"),
            "rendered_typeset":  Image(),
            "rendered_graph":    Image(),
        }),
        "post_images":           Sequence(Image()),
        "has_formulas":          Value("bool"),
        "has_post_images":       Value("bool"),
        "has_rendered_formulas": Value("bool"),
    })

    print("Casting to HuggingFace feature types (embeds image bytes)...")
    ds = ds.cast(features)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write dataset card
    (output_dir / "README.md").write_text(DATASET_CARD)

    print(f"Saving dataset to {output_dir}...")
    ds.save_to_disk(str(output_dir))

    info = _load_dataset_info()
    print(
        f"\nHuggingFace dataset saved."
        f"\n  Posts                    : {len(ds):,}"
        f"\n  Posts with formulas      : {info.get('posts_with_formulas', '?'):,}"
        f"\n  Posts with images        : {info.get('posts_with_post_images', '?'):,}"
        f"\n  Posts with rendered fmls : {info.get('posts_with_rendered_formulas', '?'):,}"
        f"\n  Output                   : {output_dir}"
    )
    return ds


def push_to_hub(ds, repo_id: str) -> None:
    """Push the dataset to the HuggingFace Hub."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set. Attempting unauthenticated push (may fail).")

    print(f"\nPushing to HuggingFace Hub: {repo_id}...")
    ds.push_to_hub(repo_id, token=token)
    print(f"  Dataset available at: https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trimodal ARQMath to HuggingFace format")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/hf_dataset",
        help="Local directory to save the HuggingFace dataset (default: data/hf_dataset)",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="HuggingFace Hub repo ID to push to, e.g. your-username/arqmath-trimodal",
    )
    args = parser.parse_args()

    ds = build_hf_dataset(args.output_dir)

    if args.push_to_hub:
        push_to_hub(ds, args.push_to_hub)


if __name__ == "__main__":
    main()
