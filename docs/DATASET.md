# ARQMath-Trimodal Dataset — Curation Notes

This document describes how the expanded ARQMath dataset was constructed, what
decisions were made at each stage, what was included or excluded, and why.

---

## 1. Source Material

The dataset is built on top of **ARQMath** (Answer Retrieval for Questions in
Math), a community challenge dataset derived from [Math Stack Exchange](https://math.stackexchange.com).

| Source artifact | Description |
|---|---|
| `Posts.V1.3.xml` | Full post dump: 2.47M questions and answers with HTML bodies, scores, tags, dates |
| `latex_representation_v3.tsv` (×101) | LaTeX string for every formula occurrence in the corpus |
| `opt_representation_v3.tsv` (×101) | Content MathML (OPT — Operator Tree) via LaTeXML |
| `slt_representation_v3.tsv` (×101) | Presentation MathML (SLT — Symbol Layout Tree) via LaTeXML |
| `qrels/task1/` | ARQMath Task 1 relevance judgements (answer retrieval) — used to define "priority posts" for rendering |
| `data/raw/arqmath/images/` | Post-embedded images scraped from HTML `<img>` tags |

The three TSV families are produced by the ARQMath organisers and cover the
same 28.3M formula occurrences with one row per occurrence, in the same
sequential order across all three files for each shard. This allows them to be
streamed in lockstep without any random-access join.

---

## 2. Pipeline Overview

The dataset is built in four sequential stages:

```
TSV shards (LaTeX + OPT + SLT)
        │
        ▼
[1] src/dataset/index.py      →  data/processed/formula_index/  (101 Parquet shards)
        │
        ▼
[2] src/dataset/render.py     →  data/processed/rendered_formulas/  (typeset + graph PNGs)
        │
        ▼
[3] src/dataset/assemble.py   →  data/processed/dataset/  (25 Parquet shards)
        │
        ▼
[4] src/dataset/export.py     →  HuggingFace datasets format (optional Hub upload)
```

Post images are scraped independently by `src/dataset/scrape.py` and land in
`data/raw/arqmath/images/` before assembly.

---

## 3. Stage 1 — Formula Index

**Script:** `src/dataset/index.py`

### What it does

The three TSV families (latex, OPT, SLT) each have 101 shard files named
`1.tsv` through `101.tsv`. Each shard's three files are zipped together row by
row (they share a sequential `id` field), producing one Parquet shard per
triplet.

The resulting formula index has **28,320,920 rows** across 101 shards.

### Schema

| Column | Type | Notes |
|---|---|---|
| `id` | int64 | Global formula sequence number, unique across corpus |
| `post_id` | int64 | Post this formula occurrence belongs to |
| `type` | string | `"question"` / `"answer"` / `"comment"` |
| `old_visual_id` | string | Visual ID used in ARQMath qrels (stable across ARQMath-1/2/3) |
| `visual_id` | string | Corrected visual ID (ARQMath-3 format) |
| `latex` | string | LaTeX source string |
| `opt` | large_string | Content MathML (OPT); null when LaTeXML conversion failed |
| `slt` | large_string | Presentation MathML (SLT); null when LaTeXML conversion failed |

### Why Parquet?

The original TSV files total several hundred GB uncompressed. Parquet gives
columnar projection (reads only needed columns), lossless compression (~5–10×
smaller on disk), and efficient batch reads for the downstream join in Stage 3.
SQLite was considered but rejected: a 28M-row database with three string
columns is slower to build and harder to stream than sharded Parquet files.

### Formula types

Formulas from `"comment"` posts are excluded from rendering (Stage 2) because
comments are not part of the answer retrieval task and tend to contain informal,
low-quality LaTeX.

---

## 4. Stage 2 — Formula Rendering

**Script:** `src/dataset/render.py`

### What is rendered

Two image types are produced per unique formula (keyed by `old_visual_id`):

1. **`{id}_typeset.png`** — Typeset mathematics using matplotlib's mathtext
   engine. Renders the LaTeX as it would appear in a document: proper symbols,
   fractions, integrals, Greek letters, etc.

2. **`{id}_graph.png`** — A function graph plotted over \[-10, 10\] using sympy
   and matplotlib, produced only when the formula can be parsed as a
   plottable single-variable function (see graph filtering below).

### Priority-first rendering

Rendering 28M formulas in full takes a very long time. By default the script
renders only formulas belonging to **priority posts**:

- **qrel-judged posts** — posts that appear in the ARQMath Task 1 relevance
  judgements. These are the posts used for training and evaluating the
  retrieval system, making their formula images the most directly useful.
- **image posts** — posts that have at least one associated post image
  (scraped in `scrape.py`). These are needed for the trimodal alignment
  training signal.

The `--all` flag disables this filter and renders the full corpus as a
background job.

### What was rendered (final counts)

| Image type | Count |
|---|---|
| Typeset PNGs | 414,206 |
| Graph PNGs | 3,775 |

---

## 5. Quality Filtering — What Was Excluded and Why

All filters below are applied both during rendering (preventing new bad renders
from being written) and during `--prune` (removing any bad renders already on
disk).

### 5.1 Trivially simple formulas

**Excluded.** These formulas render as a single symbol with no mathematical
structure. They provide no useful training signal for a multimodal model —
the rendered image is visually indistinguishable from a plain text character
or a simple icon.

Exclusion is hierarchical. Starting from the stripped LaTeX:

| Category | Examples | Rule |
|---|---|---|
| Too short | `a`, `M`, `42` | Stripped length < 3 characters |
| Single Latin char (with decoration) | `x_1`, `T^*`, `A^T`, `f^{-1}`, `g'`, `\hat{x}`, `\vec{v}`, `\overline{z}` | After stripping all decorators (superscripts, subscripts, primes, accents), only one letter remains |
| Formatted single char | `\rm B`, `\mathbf{x}`, `\:0\:` | After stripping font/spacing commands, core is a single char |
| Single Greek letter | `\phi`, `\alpha`, `\Omega`, `\phi_n`, `\alpha^2` | After stripping decorators, a single `\greek` command remains |
| Special math constants | `\infty`, `\nabla`, `\partial`, `\ell`, `\aleph` | A single "one-thing" symbol with no relationships |

> **Design note:** The same `old_visual_id` can appear in the formula index
> with *different* LaTeX strings across posts (one author writes `B`, another
> writes `\rm B` for the same visual formula). The filter operates on the
> semantic core after stripping all wrappers, so both representations are
> handled uniformly regardless of which LaTeX version was found first.

### 5.2 Formulas that matplotlib cannot typeset (unparseable)

**Excluded.** When matplotlib's mathtext engine fails to parse a formula, it
silently falls back to rendering the *raw LaTeX source text* as a string. This
produces an image showing characters like `\begin{array}{lcr}...` or `\color
{red}` rather than any actual typeset mathematics — not useful as visual data.

Pre-validation is performed with `matplotlib.mathtext.MathTextParser` before
writing any file. Formulas that fail this check are silently skipped.

The two most common categories of failure (caught by fast regex before calling
the parser):

| Pattern | Reason for failure |
|---|---|
| `\begin{...}` | LaTeX environments (matrices, aligned equations, arrays) are not supported by matplotlib's subset of LaTeX |
| `\color{...}` | Color macros are not supported |
| `\text{...}` | Text mode inside math causes parser failure in a significant fraction of cases |

These three patterns together account for the majority (~50%) of all
unparseable formulas. Remaining failures are caught by the full parser call.

**Count removed:** ~9,017 typeset renders deleted by this filter during the
initial prune pass.

### 5.3 Graph rendering exclusions

Graph rendering is attempted only when a formula passes **all** of the
following:

| Check | Rationale |
|---|---|
| Contains a bare `x` or `y` variable | Must have something to plot against |
| Contains at least one operator or named function | Bare variable names (`x` alone) are not plottable functions |
| LaTeX length ≤ 200 characters | Very long formulas are almost never simple functions |
| `sympy.parsing.latex.parse_latex` succeeds | Formula must be symbolically parseable |
| Result is **not** a `sympy.Eq` or `Relational` | Equations like `x^2 - 2 = 0` are constraints, not functions to plot |
| Exactly **one** free symbol after parsing | Multi-variable expressions cannot be plotted on a 2D axis |
| `count_ops() ≤ 60` | Very complex expressions are unlikely to produce clean, informative graphs |
| ≥ 20 finite, non-divergent plot points in \[-10, 10\] | Ensures the graph is actually visible and informative |

The original `sympy.Relational` import bug (wrong module path) caused all
graph renders to silently fail on the first run. This was fixed by importing
from `sympy.core.relational` instead of `sympy`.

**Count produced:** 3,775 graph PNGs from 75,226 candidates with graph
potential (~5% yield, reflecting the fact that most ARQMath formulas are
equations or multi-variable expressions, not plottable single-variable
functions).

---

## 6. Post Images (Scraped)

**Script:** `src/dataset/scrape.py`

Post bodies in `Posts.V1.3.xml` occasionally reference external images via
HTML `<img>` tags (e.g., hand-drawn diagrams, spreadsheet screenshots,
plots from external tools). These are genuine mathematical images contributed
by users and represent real visual content that cannot be recovered from LaTeX
alone.

The scraper extracts all `<img src="...">` URLs from post HTML, downloads them
asynchronously (using `httpx` with HTTP/2), converts to PNG where needed, and
stores them under `data/raw/arqmath/images/{post_id}/{index}.png`.

| Metric | Value |
|---|---|
| Posts with images (directories) | 4,949 |
| Image files successfully downloaded | 3,569 |

The gap between directories (4,949) and files (3,569) reflects posts where the
referenced image URL returned a 404, was behind authentication, or had been
removed from the host server since the post was written — common for
third-party image hosts over a multi-year corpus.

---

## 7. Stage 3 — Dataset Assembly

**Script:** `src/dataset/assemble.py`

Joins each post from `posts.jsonl` with:

- Formula metadata (formula IDs and LaTeX) from the formula index, via an
  in-memory Polars `group_by` on `old_visual_id`
- Rendered image paths (typeset + graph PNGs) resolved from `rendered_formulas/`
- Post image paths resolved from `data/raw/arqmath/images/`

OPT and SLT MathML representations are intentionally **not** included in the
assembled dataset shards — they are large strings that inflate memory usage
dramatically and are already available in the formula index for lookup. The
assembled dataset carries only the LaTeX (human-readable) and image paths.

The output is 25 Parquet shards of ~100,000 posts each.

### Dataset statistics

| Metric | Value |
|---|---|
| Total posts | 2,466,080 |
| Posts with at least one formula | 2,188,112 (88.7%) |
| Posts with rendered formula images | 1,843,722 (74.8%) |
| Posts with scraped post images | 4,949 (0.2%) |

---

## 8. Assembled Dataset Schema

One row per post:

| Field | Type | Description |
|---|---|---|
| `post_id` | int64 | Math Stack Exchange post ID |
| `post_type` | string | `"question"` or `"answer"` |
| `parent_id` | int64 \| null | For answers: the question post ID |
| `accepted_answer_id` | int64 \| null | For questions: ID of accepted answer (if any) |
| `title` | string \| null | Question title (null for answers) |
| `tags` | list\<string\> \| null | Stack Exchange tags (null for answers) |
| `score` | int64 | Community vote score |
| `creation_date` | string | ISO-8601 creation timestamp |
| `body_text` | string | HTML-stripped post body; formula positions replaced with `[FORMULA_i]` tokens |
| `formulas` | list\<struct\> | See formula struct below |
| `post_images` | list\<string\> | Relative paths to scraped post images |
| `has_formulas` | bool | Convenience flag |
| `has_post_images` | bool | Convenience flag |
| `has_rendered_formulas` | bool | At least one formula has a typeset PNG |

**Formula struct:**

| Field | Type | Description |
|---|---|---|
| `formula_id` | string | `old_visual_id` from the ARQMath formula index |
| `latex` | string | LaTeX source |
| `rendered_typeset` | string \| null | Relative path to `{id}_typeset.png` |
| `rendered_graph` | string \| null | Relative path to `{id}_graph.png` |

---

## 9. Reproducibility

Run in order from the repository root:

```bash
# 0. Download and extract ARQMath data
bash scripts/setup.sh

# 1. Build formula index Parquet shards (~30–60 min)
python -m src.dataset.index

# 2. Scrape post images (async, ~minutes)
python -m src.dataset.scrape data/processed/posts.jsonl data/raw/arqmath/images

# 3. Render formula images — priority posts first
python -m src.dataset.render
# Optionally render the full 28M formula corpus:
python -m src.dataset.render --all

# 4. Prune bad renders (trivial / unparseable)
python -m src.dataset.render --prune

# 5. Generate graph renders
python -m src.dataset.render --regraph

# 6. Assemble trimodal dataset
python -m src.dataset.assemble

# 7. Export to HuggingFace (optional)
python -m src.dataset.export --output-dir data/hf_dataset
python -m src.dataset.export --push-to-hub your-username/arqmath-trimodal
```

All scripts are idempotent: re-running skips already-completed work.
