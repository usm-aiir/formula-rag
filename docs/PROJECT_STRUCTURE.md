# Formula-Aware Multimodal RAG for Mathematics

## Proposed Directory Structure

```
formula-rag/
├── data/                          # All data artifacts
│   ├── raw/                       # Raw data (ARQMath download)
│   │   └── arqmath/               # ARQMath v1.3 (collection, formulas, topics, qrels, images)
│   ├── processed/                 # Cleaned, filtered, structured
│   │   ├── posts/                 # Individual multimodal post objects (JSON)
│   │   ├── chunks/                # Chunked for retrieval (by modality)
│   │   └── splits/                # train/val/test splits (ARQMath-1/2/3)
│   └── benchmark/                 # Final benchmark-ready format
│       ├── task1/                 # Answer retrieval: queries, corpus, qrels
│       ├── task2/                 # Formula retrieval: queries, corpus, qrels
│       └── task3/                 # Generation: prompts, references
│
├── src/                           # Source code — three directories, one per phase
│   ├── data_pipeline/             # Phase 1: build the extended ARQMath corpus
│   │   ├── parsing/               # Stream Posts XML → multimodal post JSONL
│   │   ├── collection/            # Download images from MSE posts
│   │   └── filtering/             # Post-level quality filters
│   ├── retrieval/                 # Phase 2: trimodal retrieval model
│   │   ├── encoders/              # Text, image, formula encoders
│   │   ├── index/                 # FAISS index building & querying
│   │   ├── train/                 # Contrastive training on ARQMath-1 qrels
│   │   └── evaluate/              # Retrieval metrics (nDCG', MAP, Recall@k)
│   └── generation/                # Phase 3: RAG pipeline & evaluation
│       ├── pipeline/              # Retrieve → prompt → generate
│       └── evaluate/              # Generation metrics (BERTScore, etc.)
│
├── configs/                       # Experiment configs
│   ├── data_collection.yaml
│   ├── retrieval.yaml
│   └── evaluation.yaml
│
├── scripts/                       # One-off or orchestration scripts
│   ├── collect_mse_data.py
│   ├── build_benchmark.py
│   └── run_baselines.py
│
├── notebooks/                     # Exploratory analysis
│   └── data_exploration.ipynb
│
├── docs/                          # Documentation
│   ├── PROJECT_STRUCTURE.md
│   ├── DATASET_DESIGN.md
│   └── BENCHMARK_SPEC.md
│
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Design Principles

- **Modality separation**: Data is stored and processed per modality so we can experiment with different encoders and fusion strategies.
- **Reproducibility**: Raw → processed → benchmark is a clear pipeline; configs drive experiments.
- **Three-phase structure**: `src/` mirrors the three project phases — data pipeline, retrieval, generation — making it easy to work on one phase without touching the others.
