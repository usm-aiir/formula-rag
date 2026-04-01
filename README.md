# formula-rag

Bimodal (text + formula) retrieval benchmark for mathematics, built on the [ARQMath](https://www.cs.rit.edu/~dprl/ARQMath/) dataset.

## Setup

```bash
# 1. Download and extract the ARQMath dataset (~2.8 GB compressed)
bash scripts/setup.sh

# 2. Build the formula index (Parquet shards from ARQMath TSVs)
python -m src.data.index

# 3. Install dependencies
pip install -r requirements.txt
```

See [`docs/DATA_ACQUISITION.md`](docs/DATA_ACQUISITION.md) for details on the raw data, split strategy, and directory layout.

## Research Tasks

Each task isolates a different retrieval modality to establish baselines before combining them.

| Task | Description | Train data | Eval data | Model |
|------|-------------|------------|-----------|-------|
| **1** | Text retrieval (joint) | ARQMath Task 1 + Task 2 qrels | Task 1 Year 3 | MathBERTa text encoder |
| **2** | Text retrieval (text only) | ARQMath Task 1 qrels only | Task 1 Year 3 | MathBERTa text encoder |
| **3** | Formula retrieval | ARQMath Task 2 qrels | Task 2 Year 3 | GAT formula encoder |

## Task 1 — Text retrieval with MathBERTa (joint training)

Trains a MathBERTa dense encoder on ARQMath Task 1 answer retrieval, optionally including Task 2 formula pairs for joint text+formula contrastive learning. Uses GradCache for large effective-batch InfoNCE (255 in-batch negatives) with micro-batch GPU memory footprint.

```bash
python -m src.task1.train --config configs/task1.yaml

python -m src.task1.eval --checkpoint checkpoints/task1
python -m src.task1.eval --checkpoint checkpoints/task1 --quick-run
```

## Task 2 — Text retrieval with MathBERTa (text only)

Same architecture and code as Task 1, but trained on ARQMath Task 1 qrels only (no formula pairs). This isolates the text retrieval signal to measure whether joint training with formula data helps or hurts.

To run Task 2, set `include_task2: false` in the config (this is the current default in `configs/task1.yaml`):

```bash
python -m src.task1.train --config configs/task1.yaml
python -m src.task1.eval --checkpoint checkpoints/task1
```

To run Task 1 (joint), set `include_task2: true` in the config.

## Task 3 — Formula retrieval with GAT encoder

Trains a Graph Attention Network on ARQMath Task 2 formula retrieval. Formulas are represented as Operator Trees (Content MathML) converted to PyG graphs. Uses symmetric InfoNCE with learnable temperature and DDP for multi-GPU training.

```bash
python -m src.task3.train --config configs/task3.yaml

python -m src.task3.eval --checkpoint checkpoints/task3/best.pt
python -m src.task3.eval --checkpoint checkpoints/task3/best.pt --quick-run
```

## Project structure

```
configs/              Task-specific YAML configs
scripts/setup.sh      Downloads and extracts the ARQMath dataset
docs/                 Data acquisition guide

src/
  data/
    index.py          Build formula index Parquet shards from ARQMath TSVs
    formula_graph.py  Convert Content MathML (OPT) to PyG graph objects
  task1/
    data.py           Load topics, qrels, post corpus (reads raw XML for inline formulas)
    dataset.py        PyTorch Dataset for contrastive (query, positive) pairs
    model.py          MathBERTa encoder with mean pooling
    train.py          GradCache training loop
    eval.py           FAISS retrieval + pytrec_eval metrics
  task3/
    dataset.py        PyTorch Dataset for formula (query, positive) pairs
    model/
      gnn.py          GAT encoder for OPT graphs
      formula_encoder.py  Wrapper with checkpoint save/load
    train.py          DDP training loop
    eval.py           FAISS retrieval + pytrec_eval metrics
```

## Archive

The trimodal dataset pipeline (render, scrape, assemble, export) and image-related code are preserved on `archive/trimodal-pipeline`.
