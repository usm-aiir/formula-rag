# formula-rag

Trimodal retrieval-augmented generation for mathematics, treating formulas as a first-class modality alongside text and images.

## Setup

```bash
# 1. Download and extract the ARQMath dataset
bash scripts/setup.sh

# 2. Download post images
pip install -r requirements.txt
python -m src.dataset.download_images <posts_jsonl> <images_dir>
```

See `docs/DATA_ACQUISITION.md` for full details.

## Dataset Pipeline

Run in order:

```bash
# Build formula index Parquet shards from TSV files (~30-60 min)
python -m src.dataset.build_formula_index

# Render formula images — typeset PNGs + function graphs (priority posts first)
python -m src.dataset.render_formulas
python -m src.dataset.render_formulas --all   # full corpus, background job

# Assemble trimodal dataset: text + formulas + images → sharded Parquet
python -m src.dataset.assemble_dataset

# Export to HuggingFace datasets format
python -m src.dataset.export_hf --output-dir data/hf_dataset
python -m src.dataset.export_hf --push-to-hub your-username/arqmath-trimodal
```
