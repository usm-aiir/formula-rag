# Formula RAG: Two-Stage Formula Retrieval Engine

This directory contains a two-stage mathematical retrieval engine designed to act as the formula-fetching component of our overarching Tri-RAG (Text, Formula, Image) architecture. 

It queries the ARQMath corpus using a fast neural/sparse hybrid approach, followed by a structural subtree coverage re-ranker.

## Architecture Overview

The pipeline is split into two distinct execution phases:

1. **Hybrid Retrieval**
   * **Method:** Fuses a Cross-Attention Dual-Encoder (Graph Neural Network evaluated via FAISS inner-product) with an exact-match keyword scanner (BM25s).
   * **Output:** Rapidly filters 8.3 million equations down to the Top K fuzzy candidates using an 80/20 Alpha-weighted Min-Max normalization.

2. **Structural Re-ranker**
   * **Method:** Fetches the raw MathML (OPT/SLT) of the top K candidates from a heavily indexed SQLite cache and shreds them into max-depth-4 structural paths. It applies IDF-weighted subtree coverage scoring with deterministic canonicalization for commutative operators.
   * **Output:** Yields the final retrieval results.

## Prerequisites: Installing the Semantic Compiler (LaTeXML)

This engine requires **LaTeXML** to perform high-precision semantic parsing, allowing for the conversion from LaTeX to SLT and OPT formula representations. Unlike standard visual compilers, LaTeXML generates the Content MathML (OPT) required for structural re-ranking.

### User-Space Installation (No Sudo Required)
If you are on a shared research server like `mirage`, install LaTeXML into your home directory using Perl's local manager:

```bash
# 1. Install local Perl manager
curl -L [https://cpanmin.us](https://cpanmin.us) | perl - -l ~/perl5 App::cpanminus local::lib

# 2. Configure environment (Add to .bashrc)
eval $(perl -I ~/perl5/lib/perl5/ -Mlocal::lib)
echo 'eval $(perl -I ~/perl5/lib/perl5/ -Mlocal::lib)' >> ~/.bashrc

# 3. Install LaTeXML (Skip tests to bypass missing TeX styling dependencies)
cpanm --notest LaTeXML
```

## Key Files & Structure

    src/task3/utils
    ├── build_sqlite_cache.py          # Migrates Parquet shards to SQLite for instant lookups
    ├── formula_retriever.py           # Core inference class (loads ML models into VRAM)
    └── api_server.py                  # FastAPI microservice for the Tri-RAG routing agent

    src/task3/eval
    ├── end_to_end_eval.py             # Offline evaluation script for ARQMath metrics
    └── visualize_results.py           # Streamlit Dashboard for manual inspection

    data/processed/                    
    ├── formula_cache.db               # Fast-access SQLite database for MathML strings
    ├── end_to_end_run.json            # Output from the offline evaluation script
    └── end_to_end_metrics.json        # Aggregated and per-topic PyTrec metrics

## Execution Guide: Live API Deployment

To run the pipeline as a live API for the Tri-RAG system, execute the following steps from the project root. Ensure your virtual environment is active, FAISS/PyTorch Geometric are properly compiled, and `fastapi`/`uvicorn` are installed. In addition, make sure a model checkpoint, faiss index, mb25 index, and formula cache SQLite DB are present.

### 1. Build the Fast-Access Cache (One-Time Setup)
Parquet files are too slow for live Key-Value lookups. Run this script once to compile the 8.3 million MathML strings into an indexed SQLite database.

    python src/task3/build_sqlite_cache.py

### 2. Start the API Server
Boot the FastAPI microservice. This will securely load the Dual-Encoder GNN and the 2GB FAISS index into VRAM during the startup lifespan, preventing memory leaks and loading times on individual queries.

    python src/task3/api_server.py

### 3. Query the Engine
The server listens on port 8567. Send a POST request containing the user's raw LaTeX query:

    curl -X POST "http://127.0.0.1:8567/search" \
         -H "Content-Type: application/json" \
         -d '{"query": "\\int_{0}^{\\infty} e^{-x^2} dx", "top_k": 5}'

## Offline Evaluation & Debugging

* **Run End-to-End Metrics:** To test the engine against the ARQMath qrels (answer key):

    python src/task3/end_to_end_eval.py

* **Launch the Visual Dashboard:** To visually inspect the top retrieval results and view confidence metrics/deltas:

    streamlit run src/task3/visualize_results.py

## Integration Notes for the Tri-RAG Pipeline

* **Query Translation (Automated):** This engine strictly expects **MathML (OPT & SLT)** to perform its deep structural matching. The `formula_retriever.py` class now natively handles this by compiling incoming LaTeX using the `latex2mathml` library. The Tri-RAG router only needs to pass standard LaTeX strings to the `/search` endpoint.
* **Context Window Injection:** The API returns a ranked JSON array containing the `visual_id` and confidence `score` of the matched formulas. Use these IDs to fetch the surrounding text context from the corpus and inject the combined Text + MathML string directly into the generative LLM's prompt.
* **Model Routing:** If the user's prompt is heavily structural (e.g., "Solve this integral..."), route heavily toward this engine's output. If the prompt is conceptual (e.g., "Who invented this formula?"), rely more on the standard text-retriever.
