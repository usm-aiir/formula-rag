# Formula RAG: Two-Stage Formula Retrieval Engine

This directory contains a two-stage mathematical retrieval engine designed to act as the formula-fetching component of our overarching Tri-RAG (Text, Formula, Image) architecture.

It queries the ARQMath corpus using a fast neural/sparse hybrid approach, followed by a highly precise structural subtree coverage re-ranker.

## Architecture Overview

The pipeline is split into two distinct execution phases to balance massive scale with surgical precision:

1. **Hybrid Retrieval**
   * **Method:** Fuses a Cross-Attention Dual-Encoder (Graph Neural Network evaluated via FAISS inner-product) with an exact-match keyword scanner (BM25s).
   * **Output:** Rapidly filters 8.3 million equations down to the Top `K` fuzzy candidates using an 80/20 Alpha-weighted Min-Max normalization.

2. **Structural Re-ranker**
   * **Method:** Extracts the raw MathML (OPT/SLT) of the `K` candidates and shreds them into max-depth-4 structural paths. It applies IDF-weighted subtree coverage scoring with deterministic canonicalization for commutative operators.
   * **Output:** Yields the final retrieval results 

## Key Files & Structure

    src/task3/eval/
    ├── eval_hybrid.py                     # Dense + Sparse Top-1000 Retrieval
    ├── phase4_structural_reranker.py      # Subtree Coverage Re-ranker
    └── visualize_results.py               # Streamlit Dashboard for manual inspection

    data/processed/                        # (To view data files, run eval scripts)
    ├── phase3_hybrid_run.json             # Intermediary output from Phase 3
    ├── phase4_structural_run.json         # Final retreval rankings from Phase 4
    └── phase4_structural_metrics.json     # Aggregated and per-topic PyTrec metrics

## Execution Guide

To run the pipeline end-to-end, execute the following scripts in order from the project root. Ensure your virtual environment is active and FAISS/PyTorch Geometric are properly compiled. A trained Dual-Formula GNN Encoder checkpoint from `src/task3/train_fusion.py` is required.

### 1. Run the Hybrid Retriever

Generates the initial Top `K` candidates. This requires the pre-computed FAISS and BM25 indices.
```

    python -m src.task3.eval_hybrid

```

*Outputs: `data/processed/phase3_hybrid_run.json`*

### 2. Run the Structural Re-ranker

Applies the Tangent-style structural override to find the exact mathematical matches.
```

    python -m src.task3.phase4_structural_reranker
    
```

*Outputs: `data/processed/phase4_structural_run.json` and metrics.*

### 3. Launch the Evaluation Dashboard

To visually inspect the top retrieval results and view confidence metrics/deltas before passing them to the LLM:
```

    streamlit run src/task3/visualize_results.py

```

## Integration Notes for the Tri-RAG Pipeline


* **Query Translation (Crucial):** This engine expects **MathML (OPT & SLT)** to perform its structural matching, not raw LaTeX. When a user inputs a query (or an OCR tool extracts one), it MUST be passed through a compiler (e.g., `LaTeXML` or `latex2mathml`) before being handed to `eval_hybrid.py` or another retrieval script.

* **Context Window Injection:** The output of `phase4_structural_run.json` will give you the `visual_id` of all retrieved formulas. You can use these IDs to fetch the surrounding text context from the corpus and inject the combined Text + MathML string directly into the generative LLM's prompt.

* **Model Routing:** If the user's prompt is heavily structural (e.g., "Solve this integral..."), route heavily toward this engine's output. If the prompt is conceptual (e.g., "Who invented this formula?"), rely more on the standard text-retriever.