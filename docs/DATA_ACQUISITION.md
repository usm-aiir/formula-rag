# Data Acquisition Guide

## Overview

This project uses the **ARQMath dataset (v1.3)** as its primary data source — a curated subset of Math Stack Exchange (MSE) with human relevance judgments and pre-built formula indexes.

We then **extend it with images** extracted from the underlying MSE posts, which ARQMath does not include.

---

## ARQMath Task Selection

This project uses all three ARQMath tasks, each with a distinct role:

| Task | Description | Role in this project |
|------|-------------|----------------------|
| **Task 1 — Answer Retrieval** | Given a question, retrieve relevant answer posts | **Primary retrieval benchmark.** qrels used for retrieval model training (ARQMath-1/2) and held-out evaluation (ARQMath-3). |
| **Task 2 — Formula Retrieval** | Given a formula from a question post, retrieve similar formulas | **Formula encoder pre-training + image extension validation.** qrels used to pre-train the formula encoder before multimodal alignment. Also the natural task for showing that our image extension benefits formula retrieval — the extension is corpus-level and applies to all posts, so evaluating it on Task 2 is appropriate. |
| **Task 3 — Open Domain QA** | Given a question, return a single generated answer (may be generated, may come from outside ARQMath) | **Generation benchmark.** End-to-end RAG pipeline evaluation. |

---

## Step 1: Download ARQMath

Run the provided download script from the project root:

```bash
bash scripts/download_arqmath.sh
```

This downloads ~2.8 GB of compressed data (~7 GB uncompressed) to `data/raw/arqmath/`.

### Optional flags

The formula indexes are large. If you want to start quickly without the structure-aware representations:

```bash
# Skip Operator Trees (saves 587 MB) and Symbol Layout Trees (saves 835 MB)
bash scripts/download_arqmath.sh --no-opt --no-slt

# Skip only SLT (saves 835 MB)
bash scripts/download_arqmath.sh --no-slt
```

You can always re-run the script later to add them; files that already exist are skipped.

---

## What Gets Downloaded

### `data/raw/arqmath/collection/`

| File | Size | Description |
|------|------|-------------|
| `Posts.V1.3.zip` | 895 MB | Main MSE corpus: all questions + answers (2010–2018). Posts include text and LaTeX formulas in `<span class="math-container">` tags. |
| `PostLinks.V1.3.xml` | 29 MB | Related and duplicate question links (link types: 1=related, 3=duplicate). Useful for cross-question relevance. |
| `Tags.V1.3.xml` | 169 KB | Tag vocabulary with counts. Useful for filtering by topic (e.g., `calculus`, `geometry`). |
| `README_DATA.md` | — | Official ARQMath data documentation. |

After download, unzip the corpus:

```bash
cd data/raw/arqmath/collection
unzip Posts.V1.3.zip
```

`Posts.V1.3.xml` will be ~4.1 GB uncompressed. It contains both questions (`PostTypeId=1`) and answers (`PostTypeId=2`) in a flat XML structure. Key fields:

- `Id` — unique post ID
- `PostTypeId` — 1=question, 2=answer
- `ParentId` — (answers only) the question's post ID
- `AcceptedAnswerId` — (questions only) ID of the accepted answer
- `Body` — HTML content with embedded LaTeX in `<span class="math-container">` tags
- `Score`, `CreationDate`, `Tags`

### `data/raw/arqmath/formulas/`

| File | Size | Description |
|------|------|-------------|
| `latex_representation_v3.zip` | 406 MB | LaTeX formula index (TSV). Columns: `formula_id, post_id, thread_id, post_type, latex_string`. |
| `opt_representation_v3.zip` | 587 MB | Operator Tree representation (Content MathML-based). Structure-aware; captures math semantics. |
| `slt_representation_v3.zip` | 835 MB | Symbol Layout Tree representation (Presentation MathML-based). Captures visual appearance. |
| `README_formulas_V3.0.md` | — | Documentation for all formula indexes. |

After download, unzip the formula indexes you need:

```bash
cd data/raw/arqmath/formulas
unzip latex_representation_v3.zip
unzip opt_representation_v3.zip   # optional
unzip slt_representation_v3.zip   # optional
```

**Which representation to use when:**
- **LaTeX** — Baseline; use for text-as-formula models or as a fallback.
- **OPT (Operator Tree)** — Encodes semantic structure (what operations are performed). Best for structural similarity.
- **SLT (Symbol Layout Tree)** — Encodes visual structure (how formulas look). Good for appearance-based matching.

### `data/raw/arqmath/topics/`

Topics are the **query posts** for each task, drawn from MSE posts made *after* the corpus snapshot (no leakage).

**Task 1 topics** (answer retrieval queries — full question posts):

| File | Topics | Source year |
|------|--------|-------------|
| `task1/arqmath1/Topics_V2.0.xml` | ~77 | MSE posts from 2019 |
| `task1/arqmath2/Topics_Task1_2021_V1.1.xml` | ~71 | MSE posts from 2020 |
| `task1/arqmath3/Topics_Task1_2022_V0.1.xml` | ~78 | MSE posts from 2021 |

**Task 2 topics** (formula retrieval queries — a specific formula from a question post):

| File | Topics | Source year |
|------|--------|-------------|
| `task2/arqmath1/Topics_V1.1.xml` | ~71 | Formulas from 2019 posts |
| `task2/arqmath2/Topics_Task2_2021_V1.1.xml` | ~71 | Formulas from 2020 posts |
| `task2/arqmath3/Topics_Task2_2022_V0.1.xml` | ~78 | Formulas from 2021 posts |

### `data/raw/arqmath/qrels/`

Qrels are in TREC format: `topic_id  0  doc_id  relevance_score`. Relevance is graded 0–3.

**Task 1 qrels** (answer post relevance):

| File | Description |
|------|-------------|
| `task1/arqmath1/qrel_task1_2020_official` | ARQMath-1 official |
| `task1/arqmath2/qrel_task1_2021_official.tsv` | ARQMath-2 official |
| `task1/arqmath2/qrel_task1_2021_all.tsv` | ARQMath-2 with additional assessments |
| `task1/arqmath3/qrel_task1_2022_official.tsv` | ARQMath-3 official |
| `task1/arqmath3/qrel_task1_2022_all.tsv` | ARQMath-3 with additional assessments |

**Task 2 qrels** (formula relevance, keyed by visual group ID):

| File | Description |
|------|-------------|
| `task2/arqmath1/qrel_task2_2020_official.tsv` | ARQMath-1 official |
| `task2/arqmath1/qrel_task2_2020_all.tsv` | ARQMath-1 full |
| `task2/arqmath2/qrel_task2_2021_official.tsv` | ARQMath-2 official |
| `task2/arqmath2/qrel_task2_2021_all.tsv` | ARQMath-2 full |
| `task2/arqmath3/qrel_task2_2022_official.tsv` | ARQMath-3 official |
| `task2/arqmath3/qrel_task2_2022_all.tsv` | ARQMath-3 full |

**Use the `_all` qrels when available** — they include secondary assessments and increase coverage.

### `data/raw/arqmath/eval_scripts/arqmath3/`

Official ARQMath-3 evaluation scripts:

| Script | Task | Purpose |
|--------|------|---------|
| `task1_get_results.py` | 1 | Computes nDCG', MAP, P@10 |
| `arqmath_to_prime_task1.py` | 1 | Converts run files to `trec_eval` format |
| `task2_get_results.py` | 2 | Computes nDCG', MAP, P@10 for formula retrieval |
| `de_duplicate_2022.py` | 2 | De-duplicates formula runs by visual group |

---

## Step 2: Understand the Split Strategy

The ARQMath corpus is temporally clean — topics (queries) are drawn from posts made *after* the corpus snapshot, so there is no data leakage:

| Data | Time period | Task 1 role | Task 2 role | Task 3 role |
|------|-------------|-------------|-------------|-------------|
| `Posts.V1.3` (corpus) | 2010–2018 | Retrieval corpus | Formula corpus | Generation context |
| ARQMath-1 topics + qrels | 2019 queries | Retrieval training | Formula encoder pre-training | — |
| ARQMath-2 topics + qrels | 2020 queries | Retrieval validation | Formula encoder validation | — |
| ARQMath-3 topics + qrels | 2021 queries | Retrieval test (held-out) | Formula retrieval test (held-out) | Generation test (held-out) |

The retrieval model is trained on ARQMath-1 topics (both Task 1 and Task 2 qrels), validated on ARQMath-2, and evaluated on ARQMath-3. ARQMath-3 topics are never seen during training or validation.

---

## Step 3: Add Images (Our Extension)

ARQMath covers text and formulas but **not images**. The underlying MSE posts can contain images embedded as `<img>` tags in the HTML body.

To add images:

1. **Parse `Posts.V1.3.xml`** and extract `<img src="...">` from the `Body` field.
2. **Filter** — images are only present in a subset of posts; note the `post_id`s that have images.
3. **Download images** — MSE images are hosted at `https://i.stack.imgur.com/`. Download and store locally.

This is done in `src/data/collection/` (coming in the next step). The image extraction script will produce:

```
data/raw/arqmath/images/
  {post_id}/
    {image_index}.{ext}
```

along with a metadata file mapping image URLs to local paths.

---

## Directory Structure After Download

```
data/raw/arqmath/
├── collection/
│   ├── Posts.V1.3.zip        (keep for reproducibility)
│   ├── Posts.V1.3.xml        (unzipped, 4.1 GB)
│   ├── PostLinks.V1.3.xml
│   ├── Tags.V1.3.xml
│   └── README_DATA.md
├── formulas/
│   ├── latex_representation_v3.zip
│   ├── latex_representation_v3/
│   ├── opt_representation_v3.zip  (optional)
│   ├── opt_representation_v3/     (optional)
│   ├── slt_representation_v3.zip  (optional)
│   ├── slt_representation_v3/     (optional)
│   └── README_formulas_V3.0.md
├── topics/
│   ├── task1/
│   │   ├── arqmath1/Topics_V2.0.xml
│   │   ├── arqmath2/Topics_Task1_2021_V1.1.xml
│   │   └── arqmath3/Topics_Task1_2022_V0.1.xml
│   └── task2/
│       ├── arqmath1/Topics_V1.1.xml
│       ├── arqmath2/Topics_Task2_2021_V1.1.xml
│       └── arqmath3/Topics_Task2_2022_V0.1.xml
├── qrels/
│   ├── task1/
│   │   ├── arqmath1/qrel_task1_2020_official
│   │   ├── arqmath2/qrel_task1_2021_official.tsv
│   │   ├── arqmath2/qrel_task1_2021_all.tsv
│   │   ├── arqmath3/qrel_task1_2022_official.tsv
│   │   └── arqmath3/qrel_task1_2022_all.tsv
│   └── task2/
│       ├── arqmath1/qrel_task2_2020_official.tsv
│       ├── arqmath1/qrel_task2_2020_all.tsv
│       ├── arqmath2/qrel_task2_2021_official.tsv
│       ├── arqmath2/qrel_task2_2021_all.tsv
│       ├── arqmath3/qrel_task2_2022_official.tsv
│       └── arqmath3/qrel_task2_2022_all.tsv
├── eval_scripts/
│   └── arqmath3/
│       ├── task1_get_results.py
│       ├── arqmath_to_prime_task1.py
│       ├── task2_get_results.py
│       └── de_duplicate_2022.py
└── images/           (populated after image extraction step)
    └── {post_id}/
```

---

## Citation

If you use this data, cite the ARQMath-3 overview paper:

> Mansouri, B., Zanibbi, R., Oard, D.W., and Agarwal, A. (2022).
> Overview of ARQMath-3: Third CLEF Lab on Answer Retrieval for Questions on Math.
> *CLEF 2022 Working Notes*, pp. 1–27.
