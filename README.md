# Formula RAG

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Orchestrator](#orchestrator)
   - [tri_search.py](#tri_searchpy)
   - [clip_handler.py](#clip_handlerpy)
   - [gnn_handler.py](#gnn_handlerpy)
   - [formula_utils.py](#formula_utilspy)
   - [visualize_rag.py](#visualize_ragpy)
5. [Service: Text](#service-text)
   - [text_handler.py](#text_handlerpy)
6. [Service: Formula](#service-formula)
   - [gnn_handler.py](#service-formula-gnn_handlerpy)
   - [formula_handler.py](#formula_handlerpy)
   - [src/task3/utils/formula_retriever.py](#srctask3utilsformula_retrieverpy)
7. [Service: Image (CLIP)](#service-clip)
   - [image_handler.py](#image_handlerpy)
   - [clipvb_handler.py](#clipvb_handlerpy)
   - [longclip_handler.py](#longclip_handlerpy)
   - [dataset_handler.py](#dataset_handlerpy)


## Architecture Overview

This system answers math questions by searching three types of content at once: plain text, math formulas, and images of math. A central orchestrator script ties all three together and produces a final answer using a local language model.

### What it does

You run the orchestrator directly. It takes a question, searches across all three content types at the same time, collects the results, and feeds the most relevant ones into a language model. The language model reads the retrieved content and produces a final answer.

### The three search modules

**Text** searches a large index of math forum posts. It uses a fine-tuned sentence embedding model and OpenSearch to find posts that match the meaning of the question.

**Formula** searches math formulas stored as graphs. It uses a graph neural network that understands the structure of a formula, not just its text. It also has a BM25 keyword pass and a structural re-ranker on top to sharpen the results. This is the most complex of the three.

**Image** searches images of math using CLIP, a model that can compare text to images. There are three CLIP variants available: a base CLIP model, a math-specific fine-tuned model called MathVB, and LongCLIP which handles longer text inputs. Each one has its own FAISS index built from a dataset of math images.

### How the orchestrator works

The orchestrator (`tri_search.py`) imports the three search modules directly as Python code. There are no servers or network calls at runtime. All three searches run in parallel using a thread pool. Once results come back, they are merged and passed to a locally running Llama 3.1 8B model, which generates the final answer.

### Why there are two virtual environments

The image module (`service-clip`) requires specific pinned versions of PyTorch and Transformers that conflict with what the rest of the codebase needs. Because of this it lives in its own virtual environment at `service-clip/.clipvenv`. Everything else, the orchestrator, text module, and formula module, shares a single virtual environment at `.venv` in the repo root.



## Orchestrator

The orchestrator is the entry point for the whole system. It lives in the `orchestrator/` directory. You run it directly from the command line or call its functions from your own script.

Its job is to coordinate the three search modules, merge their results, and feed everything to a local language model.

### tri_search.py

The heart of the rag, imports everything and runs a search. Note, memory fails will likely occur, this is a bug I did not
get around to fixing. Simply ensure all indexes and models per model are cleared before loading another or look into
other memory management stratagies. Right now it will attempt to do it in parrell, I thought this was smart at first, and
it would be if I had 128gb of ram. 

**Running it from the command line:**

```bash
cd orchestrator
python tri_search.py "What is the derivative of x squared?"
```

If you run it without an argument it drops into an interactive prompt.

**Using it from code:**

```python
from tri_search import load_model, rag_query

load_model()  # loads Llama 3.1 8B into memory, only needs to happen once

answer = rag_query("Solve x^2 - 5x + 6 = 0")
print(answer)
```

**What `rag_query` does step by step:**

1. Runs `text_sources`, `formula_sources`, and `clipvb_sources` in parallel using a thread pool.
2. Each function returns a string of scraped post text relevant to the query.
3. Those three strings get passed to `build_prompt`, which assembles them into a structured prompt.
4. The prompt is passed to `prompt_model`, which runs it through the loaded Llama pipeline.

**Note, this is the main memory problem I mentioned above**
```python
with ThreadPoolExecutor(max_workers=3) as executor:
    fut_text     = executor.submit(text_sources,   query, top_k_text)
    fut_formulas = executor.submit(formula_sources, query, top_k_formulas)
    fut_clipvb   = executor.submit(clipvb_sources,  query, top_k_images)

prompt = build_prompt(
    query,
    text_context=fut_text.result(),
    formula_context=fut_formulas.result(),
    image_context=fut_clipvb.result(),
)
return prompt_model(prompt)
```

**The model:** Llama 3.1 8B Instruct loaded via `transformers`. If a LoRA adapter exists at `orchestrator/models/mathmex-llama-dpo-adapter/` it is merged in automatically. Set `LLAMA_MODEL_NAME` in `orchestrator/.env` to use a different checkpoint. Set `HUGGINGFACE_TOKEN` if the model requires authentication. This is due to be changed, I just kept the same model used by mathmex chat, 
but I am positive other models will preform better. 

---

### clip_handler.py

A thin wrapper around `service-clip/longclip_handler.py`. It adds `service-clip` to `sys.path` then imports the module directly. No HTTP involved.

```python
from clip_handler import fetch_clip_results

hits = fetch_clip_results("eigenvalue decomposition", k=5)
for hit in hits:
    print(hit["rank"], hit["title"])
    print(hit["scraped_text"][:200])
```

Each result has: `rank`, `score`, `image_id`, `source`, `title`, `url`, `file_path`, `scraped_text`.

---

### gnn_handler.py

wrapper around `service-formula/gnn_handler.py`. Because both files share the same name, it loads the service module via `importlib.util.spec_from_file_location` to avoid a circular import.

```python
from gnn_handler import fetch_gnn_results

hits = fetch_gnn_results(r"What is $\frac{d}{dx} x^2$?", k=5)
for hit in hits:
    print(hit["rank"], hit["latex"])
    print(hit["scraped_text"][:200])
```

It first calls `formula_utils.extract_formulas()` to pull LaTeX out of the query string. If no formulas are found it returns an empty list immediately. Each result has: `rank`, `score`, `visual_id`, `latex`, `post_id`, `url`, `scraped_text`.

---

### formula_utils.py

Shared helpers for working with LaTeX. Used by both the orchestrator and the formula service.

```python
from formula_utils import extract_formulas, latex_to_mathml, trim_math_delimiters

# Pull all LaTeX out of a question string
formulas = extract_formulas(r"Solve $x^2 - 4 = 0$ and $\frac{d}{dx} x^2$")
# -> ["x^2 - 4 = 0", "\\frac{d}{dx} x^2"]

# Strip surrounding $ delimiters
clean = trim_math_delimiters("$x^2$")
# -> "x^2"

# Convert LaTeX to a MathML string (used by the formula service)
mathml = latex_to_mathml(r"\frac{x^2}{2}")
```

`extract_formulas` handles `$...$`, `$$...$$`, and bare LaTeX expressions. It filters out single characters, plain numbers, and whitespace-only strings.

---

### visualize_rag.py

A Streamlit dashboard for exploring what each search module returns before the LLM sees it. It is useful when you want to check retrieval quality without running the full pipeline. 

Lucas here... Ill be brutally honest this whole streamlit dashboard was vibe coded, I have zero clue how it works. So here is chatgpt explaining it. All I know is it does what it is suppose to do and it helped debugging a ton during development

```bash
cd orchestrator
streamlit run visualize_rag.py
```

Streamlit works by re-running the entire script from top to bottom every time you interact with the page. There is no server loop you write yourself. You just write Python, and Streamlit turns widgets into UI and reruns on each interaction.

**Sidebar controls**

The sidebar is built with `st.sidebar` calls at the top of the script. They render before anything else and their values are available as Python variables for the rest of the run:

```python
query         = st.sidebar.text_area("Enter a math question:", ...)
top_k_text    = st.sidebar.slider("Text results (k)", 0, 20, 5)
top_k_formulas = st.sidebar.slider("Formula results (k)", 0, 20, 5)
top_k_images  = st.sidebar.slider("Image results (k)", 0, 10, 3)
run_llm       = st.sidebar.checkbox("Run LLM and show final answer", value=False)
run_btn       = st.sidebar.button("Run Retrieval", ...)
```

Nothing runs until you click **Run Retrieval**. The `if run_btn and query.strip():` block at the bottom is what gates all the search logic.

**The four tabs**

When you click the button, four tabs appear: Text, Formulas, Images, and RAG Answer.

```python
tab_text, tab_formula, tab_image, tab_rag = st.tabs(
    ["📄 Text", "∑ Formulas", "🖼 Images", "🤖 RAG Answer"]
)
```

Each tab runs its search and stores context strings that the RAG tab later reads. If you set a slider to `0`, that tab skips its search entirely and shows an info message instead.

**Text tab**

Calls `text_sources(query, top_k=top_k_text)` from `tri_search.py`, which goes through `TextHandler` and OpenSearch. The returned string is a block of scraped post text. It gets rendered raw with `st.markdown`. It also gets saved to `text_context` for the RAG tab.

**Formula tab**

Calls `fetch_gnn_results(query, k=top_k_formulas)` from `gnn_handler.py`. Results are rendered with a helper `_render_formula_hits`. Each hit is laid out in three columns: rank and score on the left, the matched LaTeX rendered with `st.latex()` in the middle, and a link to the original post on the right.

```python
col_rank, col_math, col_meta = st.columns([1, 5, 2])
with col_math:
    st.latex(latex)  # renders the formula as math in the browser
```

After rendering, the scraped post text from each hit is joined into `formula_context` for the RAG tab.

**Image tab**

Calls `fetch_clip_results(query, k=top_k_images)` from `clip_handler.py`. Results are rendered with `_render_image_hits`, which lays them out in a 3-column grid. It tries to display the actual image file from `hit["file_path"]`. If the file does not exist on disk, it shows a caption saying so instead of crashing.

```python
if image_path and Path(image_path).exists():
    st.image(image_path, caption=title, use_container_width=True)
else:
    st.caption(f"*(image not on this machine: {image_path})*")
```

**RAG Answer tab**

Takes the three context strings from the other tabs and calls `build_prompt()` to assemble the full prompt. The prompt is shown inside a collapsible `st.expander` so you can inspect exactly what the model will receive.

```python
with st.expander("View assembled prompt", expanded=False):
    st.code(prompt, language="text")
```

If **Run LLM** is checked in the sidebar, it calls `tri_search.load_model()` and `tri_search.prompt_model(prompt)`. Loading happens lazily here, meaning the model only gets loaded if you actually check that box. It can take a few minutes on the first run. The answer is displayed with `st.success` and `st.markdown`.

If the box is not checked, the tab just shows the assembled prompt and waits.

---



## Service: Text

Handles all things text retreval 

---

### text_handler.py

All search logic is inside the `TextHandler` class. Both the embedding model and the OpenSearch client are loaded lazily

**Basic usage:**

```python
from text_handler import TextHandler

handler = TextHandler()

# Returns a formatted string of scraped post text, ready for the LLM
context = handler.retrieve_relevant_text("What is a Taylor series?", top_k=5)
print(context)
```

You can also run it directly:

```bash
cd service-text
python text_handler.py
```

**What `search()` does:**

1. Strips HTML tags and `$` signs from the query using BeautifulSoup. This avoids passing raw LaTeX into the embedding model.
2. Encodes the cleaned query into a vector using the fine-tuned sentence transformer.
3. Sends a `knn` query to OpenSearch across all five indices at once. The `knn` query finds the `top_k` nearest neighbours by cosine similarity in the `body_vector` field.
4. Returns a list of `TextRetrievalResult` objects, one per hit.

```python
search_query = {
    "query": {
        "bool": {
            "must": [{"knn": {"body_vector": {"vector": query_vector, "k": top_k}}}]
        }
    }
}
```

The five indices searched are:
- `mathmex_math-overflow`
- `mathmex_math-stack-exchange`
- `mathmex_mathematica`
- `mathmex_wikipedia`
- `mathmex_youtube`

**What `retrieve_relevant_text()` does:**

This is the method the orchestrator actually calls. It calls `search()` to get ranked hits, then for each hit it takes the `doc_id` (which is a URL) and calls `scrape_post_url()` to fetch the live post content. If scraping fails, it falls back to the `body_text` field stored in OpenSearch. Results are joined into one string with `[Source N]` labels.

The OpenSearch client is configured from environment variables in `service-text/.env`. If you do not have the credentials this will not work. 


### schemas/TextRetrievalResult.py

A simple Pydantic model used as the return type for `TextHandler.search()`: This was a legacy data class wrapper that was migrated
over from the mathmex chat project. It should be switched to a @dataclass object or something simular. However, this works well,
a different tool just may be better

```python
class TextRetrievalResult(BaseModel):
    doc_id: str   # the URL of the source post
    score:  float # cosine similarity score from OpenSearch
    rank:   int   # 1-indexed position in the result list
    text:   str   # body_text field from the OpenSearch document
```


---

### utils/scrape_url.py

Scrapes a Stack Exchange post URL and returns the question body plus up to two answers as plain text.

```python
from utils.scrape_url import scrape_post_url

text = scrape_post_url("https://math.stackexchange.com/q/12345")
# Returns: "Question:\n...\n\nAnswer:\n..."
```

It uses BeautifulSoup to select `.question .s-prose` for the question and `.answer .s-prose` for answers. It only takes the first two answers. If the request fails or the page structure does not match, it returns an empty string and logs a warning instead of raising.
To get more info on `.answer .s-prose` and `.question .s-prose`, go to any math stack exchange post and inspect it. Look at the div's
and their div class names. you will find the important QnA we need fall under the div classes `.answer .s-prose` and `.question .s-prose`.

The same utility also exists in `orchestrator/utils/scrape_url.py`. Both copies are identical. If you change one, change the other. 
TODO: This is a major flaw that should be addressed. I apologize for the oversite and not getting this done. Not clean on my part

---

### eval_arqmath.py

Evaluates the text handler against the ARQMath Task 1 benchmark.

```bash
cd service-text
python eval_arqmath.py --qrel arqmath_task1.qrel
```

It reads the ARQMath topics XML, runs each query through `TextHandler.search()`, extracts the numeric post ID from each result URL, and scores using `ranx`. The qrel file must be in standard 4-column TREC format.


## Service: Formula

The formula service searches a corpus of 8.3 million math formulas. It lives in `service-formula/`. The orchestrator calls it by importing `service-formula/gnn_handler.py` directly.

There are two separate retrieval backends in this service. The GNN pipeline (`gnn_handler.py` + `FormulaRetriever`) is the primary one and the one the orchestrator uses. The TangentCFT pipeline (`formula_handler.py`) is a secondary backend that uses a different approach and is kept for comparison or fallback. For tangentcft, you need to clone the formula-search repo into your parent directory to get the code needed. This formula search code used is on the mathmex intergration branch. Also, `latexml` must be installed on the system. `FormulaRetriever` calls `latexmlmath` as a subprocess to convert LaTeX to MathML.

### gnn_handler.py

This is the entry point the orchestrator calls. It wraps `FormulaRetriever` and the parquet formula data behind a simple `search()` function.

Both the retriever and formula data are loaded lazily and cached as module-level globals, so the first call is slow, but the rest will speed up after that. 

```python
# called by the orchestrator via gnn_handler.fetch_gnn_results()
hits = search(r"\frac{d}{dx} x^2", k=10)
# returns: [{"rank": 1, "visual_id": "...", "latex": "...", "post_id": 12345, "score": 0.94}, ...]
```

**How `search()` works:**

1. Calls `FormulaRetriever().search(latex_query)` to get back a list of `(visual_id, score)` tuples.
2. Looks up each `visual_id` in the in-memory formula data dict, which was loaded from sharded parquet files at `data/processed/formula_index/`.
3. Returns a list of dicts with `rank`, `visual_id`, `latex`, `post_id`, and `score`.

**The parquet data:**

The parquet files are columnar. Each shard is read with `pyarrow` and iterated row-wise by zipping the three columns together:

```python
for vid, latex, post_id in zip(
    table["visual_id"].to_pylist(),
    table["latex"].to_pylist(),
    table["post_id"].to_pylist(),
):
    _formula_data[str(vid)] = {"latex": str(latex), "post_id": int(post_id)}
```

---

### src/task3/utils/formula_retriever.py

This is the core of the formula service. `FormulaRetriever` runs a four-phase pipeline every time `.search()` is called.

**Initialisation** (happens once, again it will be slow):

```python
retriever = FormulaRetriever()
```

This loads three things into memory: the GNN dual-encoder from `checkpoints/task3/phase3_fusion/`, the FAISS dense index from `checkpoints/task3/faiss_index/`, and the BM25 sparse index from `checkpoints/task3/bm25_index/`. This is the most memory intensive part of the entire rag. 

**The four phases of `.search(latex_query)`:**

**Phase 1 — LaTeX parsing.**
The query string is passed to `latexmlmath` as a subprocess. This produces two MathML representations: Presentation MathML (SLT) which describes how the formula looks, and Content MathML (OPT) which describes what it means. If this step fails (e.g. `latexml` is not installed), the function returns empty results immediately.

**Phase 2 — Dense GNN search.**
The OPT and SLT trees are converted to PyTorch Geometric graph objects. The GNN dual-encoder embeds them into a single query vector. FAISS searches the 8.3M-formula index for the top 1000 nearest neighbours by cosine similarity.

**Phase 3 — Hybrid alpha fusion.**
BM25 runs a token-level sparse search on MathML tokens from the SLT tree. The dense scores and sparse scores are min-max normalised and then combined with a weighted sum: `score = 0.80 * dense + 0.20 * sparse`. The top 1000 results after fusion move to phase 4.

**Phase 4 — Structural re-ranking.**
The full MathML XML for each of the top 1000 candidates is fetched from a SQLite cache at `data/processed/formula_cache.db`. Structural path tokens are extracted from both the query and each candidate. These paths encode tree traversal patterns at multiple abstraction levels (exact, variable-normalised, fully-generalised). A custom IDF-weighted overlap score is computed, then the dense ranking and the structural ranking are fused using Reciprocal Rank Fusion (RRF). The final top-k results are returned.

```python
retriever = FormulaRetriever()
results = retriever.search(r"\int_{0}^{\infty} e^{-x^2} dx", final_top_k=5)
# returns: [("visual_id_string", fusion_score), ...]
```

### formula_handler.py

The secondary backend. It wraps TangentCFT, an older formula retrieval system that works by converting LaTeX to Symbol Layout Trees (SLTs) and searching a pre-encoded TSV index. Again, you need formula-search repo to be cloned for any of this to work. 

The orchestrator does not call this directly. It exists for benchmarking and as a fallback. If you want to use it, you instantiate it with a list of LaTeX strings and call `retrieve_similar_formulas()`:

```python
from formula_handler import FormulaHandler

handler = FormulaHandler([r"\frac{d}{dx} x^2", r"\int x dx"])
results = handler.retrieve_similar_formulas(top_k=10)
```

Each result dict includes `searched_formula`, `returned_formula`, `post_id`, `thread_id`, `rank`, and `score`. It requires the `formula-search/` directory to be present and a SQLite mapping DB at `formula-search/data/tsv_index.sqlite`.

---

## Service: CLIP

The image service searches a dataset of math images. It lives in `service-clip/` and runs in its own virtual environment at `service-clip/.clipvenv` because its PyTorch version conflicts with the rest of the repo.

There are three independent image search handlers, each built around a different CLIP variant. All three follow the same pattern: build a FAISS index once, then search it at query time.

The orchestrator uses `longclip_handler` (LongCLIP) by default, via `clip_handler.py` in the orchestrator directory.

---

### dataset_handler.py

All three image handlers share this module to iterate the image dataset on disk. It reads TSV metadata files from three sources: Math Stack Exchange, MathOverflow, and Mathematica. This dataset lives in /home/behrooz.mansouri/MathImages. Create a soft link to this to use the data.

Each TSV row maps to a `MathImageEntry` dataclass:

```python
@dataclass
class MathImageEntry:
    source:      str   # "MSE" | "MathOverflow" | "Mathematica"
    image_id:    str   # "<post_id>_<image_index>", e.g. "97_2"
    post_id:     str
    image_index: int
    title:       str   # post title
    url:         str   # link to source post
    image_path:  Path  # absolute path to the .png file
```

The dataset root is hardcoded to `/home/lucas.matheson/MathImages`. Obviously, that is my (Lucas - thats me) account, you need to change that. If you move the dataset, update `MATHIMAGES_ROOT` in `dataset_handler.py`. By default `iter_dataset()` skips entries whose image file does not exist on disk.

### image_handler.py

Uses the base OpenAI CLIP model (`ViT-B-32`). If a fine-tuned checkpoint exists at `checkpoints/clip_finetune/best.pt` it loads that instead.

**Build the index:**

 Do this before running anything

```bash
cd service-clip
source .clipvenv/bin/activate
python image_handler.py --build
```

This iterates every image in the dataset, encodes it to a 512-dimensional L2-normalised vector, and saves the FAISS index to `data/clip_index/mathimages.index` along with a `metadata.json` sidecar.

**Search:**

```python
from image_handler import search

results = search("a triangle with labeled angles", k=5)
# or search with an image file:
results = search("/path/to/query.png", k=5)
```

Each result has `rank`, `score`, `image_id`, `source`, `title`, `url`.

---

### clipvb_handler.py

Uses LLM2CLIP: a ViT-L-14-336 image model paired with a LLaMA-3-8B text encoder. This is the best-performing model from the MATVB benchmark and the one the orchestrator calls by default.

The text encoder is big (~16 GB). Loading it the first time is slow. Both the image model and text model are cached in module-level globals so they only load once per process.

**Build the index:**

```bash
cd service-clip
source .clipvenv/bin/activate
python clipvb_handler.py --build
```

Index saves to `data/llm2clip_index/`. SVG images are converted to PNG on the fly using `cairosvg` before encoding.

**Search:**

```python
from clipvb_handler import search

results = search("eigenvalue decomposition", k=5)
```

Each result has `rank`, `score`, `image_id`, `source`, `title`, `url`, `file_path`.

---

### longclip_handler.py

Uses LongCLIP-L, the main model requested to be used. This is a CLIP variant fine-tuned to handle longer text inputs (up to 248 tokens instead of the standard 77). Useful when the query is a full question rather than a short phrase.

Loads a fine-tuned checkpoint from `checkpoints/longclip/longclip-L-finetuned.pt` if it exists, otherwise falls back to the base `longclip-L.pt`.

Download the base checkpoint:

```bash
huggingface-cli download BeichenZhang/LongCLIP-L longclip-L.pt \
    --local-dir service-clip/checkpoints/longclip/
```

**Build the index:**

```bash
cd service-clip
source .clipvenv/bin/activate
python longclip_handler.py --build
```

Index saves to `data/longclip_index/`.

**Search:**

```python
from longclip_handler import search

results = search("Solve the integral of e to the power of negative x squared from 0 to infinity", k=5)
```