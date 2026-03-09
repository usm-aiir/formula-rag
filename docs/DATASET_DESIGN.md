# Dataset & Benchmark Design

## 1. Data Source: ARQMath + Image Extension

**Primary source:** [ARQMath v1.3](https://www.cs.rit.edu/~dprl/ARQMath/) — a curated benchmark built on Math Stack Exchange (MSE), with:
- ~1.1M MSE posts (2010–2018) as the retrieval corpus
- ~226 human-annotated query topics across three editions (ARQMath-1/2/3)
- Human relevance judgments (qrels) for Task 1 (answer retrieval)
- Pre-built formula indexes in LaTeX, OPT (Operator Trees), and SLT (Symbol Layout Trees)

**Our extension:** ARQMath does not include images. We extract and add images from the underlying MSE post HTML, giving us a trimodal corpus (text + formulas + images).

**Why ARQMath over raw MSE:**
- Human qrels provide reliable relevance labels for retrieval training and evaluation
- Formula structure (OPT/SLT) is already parsed and indexed
- Established benchmark enables comparison with prior work
- Temporal split (corpus 2010–2018, topics 2019–2021) is clean by design

**ARQMath tasks used:**
- **Task 1 (Answer Retrieval)** — retrieval evaluation, provides qrels
- **Task 3 (Open Domain QA)** — generation evaluation, allows generated answers
- Task 2 (Formula Retrieval) is intentionally excluded; it uses isolated formula queries which don't fit the multimodal RAG setting

---

## 2. Inclusion Criteria

The base corpus is all ARQMath posts (2010–2018). For the image-extended benchmark subset, a post qualifies if:

| Criterion | Rationale |
|-----------|-----------|
| Has ≥1 image in question OR answer | Ensures the image modality is present |
| Contains ≥1 math expression (LaTeX) | Formula modality is always present in ARQMath posts |
| Has an accepted answer (or high-score answer) | Provides ground-truth for Task 3 generation eval |
| Not deleted | Quality filter |

**Optional filters (for subset experiments):**
- Minimum score threshold
- Specific tags (e.g., `geometry`, `calculus`, `linear-algebra`)
- Posts that appear in ARQMath topic threads (highest-quality subset)

---

## 3. Multimodal Post Schema

Each example is a **multimodal object**, not a flattened string:

```json
{
  "id": "mse_12345",
  "question": {
    "text": "How do I prove that...",
    "formulas": [
      {"latex": "\\frac{a}{b}", "position": 0, "context": "inline"},
      {"latex": "\\int_0^1 f(x)\\,dx", "position": 1, "context": "block"}
    ],
    "images": [
      {"url": "...", "alt": "...", "type": null}
    ]
  },
  "answer": {
    "text": "...",
    "formulas": [...],
    "images": [...]
  },
  "metadata": {
    "score": 15,
    "accepted": true,
    "tags": ["calculus", "integration"]
  }
}
```

**Key design choice:** Formulas and images are **extracted and indexed separately** so retrieval can operate per modality.

---

## 4. Benchmark Structure

The benchmark covers all three ARQMath tasks, each with a distinct role.

### 4.1 Answer Retrieval (ARQMath Task 1)

**Input:** Question (text + formulas + images)  
**Output:** Ranked list of answer posts  
**Ground truth:** ARQMath qrels (human relevance judgments, graded 0–3)  
**Metrics:** nDCG', MAP, P@10 (ARQMath standard); modality-specific Recall@k (our extension)

**Format:**
```
benchmark/retrieval/
├── queries.jsonl       # {query_id, question_text, question_formulas, question_images}
│                       # derived from ARQMath Task 1 topic XML files
├── corpus/             # Answer post chunks with modality tags
└── qrels.tsv           # ARQMath qrels (topic_id, 0, answer_id, relevance)
```

### 4.2 Formula Retrieval (ARQMath Task 2)

**Input:** A formula from a question post (with its surrounding post context, including any images)
**Output:** Ranked list of relevant formulas (with their source posts)
**Ground truth:** ARQMath Task 2 qrels (keyed by visual group ID)
**Metrics:** nDCG', MAP, P@10 (ARQMath standard)

**Two roles:**
1. **Formula encoder pre-training** — ARQMath-1/2 qrels train the formula encoder before multimodal alignment (Stage 1 training)
2. **Image extension validation** — ARQMath-3 held-out evaluation shows whether adding image context to posts improves formula retrieval

```
benchmark/task2/
├── queries.jsonl       # {query_id, formula_latex, source_post_id, post_images}
├── corpus/             # All formulas with source post context
└── qrels.tsv           # ARQMath Task 2 qrels
```

### 4.3 Generation (ARQMath Task 3)

**Input:** Question + retrieved evidence (from Task 1)  
**Output:** Single generated answer  
**Ground truth:** Top-relevant answer(s) from qrels; optionally human evaluation  
**Metrics:** BERTScore or similar similarity to reference; human evaluation on a subset

**Format:**
```
benchmark/generation/
├── prompts.jsonl       # {query_id, question, retrieved_chunks}
├── references.jsonl    # {query_id, reference_answer} (from qrels top answers)
└── rubric.md           # Evaluation criteria
```

---

## 5. Chunking Strategy

**Challenge:** A single post has text, formulas, and images interleaved. How do we chunk for retrieval?

**Options:**

| Strategy | Pros | Cons |
|----------|------|------|
| **Sentence-level** | Fine-grained | May split formulas awkwardly |
| **Paragraph-level** | Natural boundaries | Variable length |
| **Modality-specific** | Clean separation for per-modality retrieval | Loses local context |
| **Hybrid** | Text chunks + standalone formula chunks + image chunks | More complex indexing |

**Recommendation:** Start with **modality-specific chunks** — each chunk is either text-only, formula-only, or image-only. This aligns with the three-branch retrieval design. Cross-modal context can be preserved via metadata (e.g., chunk A and B are from the same paragraph).

---

## 6. Splits & Scale

The temporal split is fixed by the ARQMath edition structure:

| Split | Topics | Queries | Corpus posts | Purpose |
|-------|--------|---------|--------------|---------|
| **Train** | ARQMath-1 (2019) | ~77 | All 2010–2018 | Retrieval model training, generation fine-tuning |
| **Val** | ARQMath-2 (2020) | ~71 | All 2010–2018 | Hyperparameter tuning |
| **Test** | ARQMath-3 (2021) | ~78 | All 2010–2018 | Final held-out evaluation |

The retrieval corpus (all 2010–2018 posts) is shared across all splits — the model always searches the same corpus. Only the query topics differ by split.

**Scale for image extension:** Not all corpus posts have images. We'll build a filtered index of image-containing posts alongside the full corpus index. The ~226 ARQMath topics are relatively few; if more training signal is needed, we can supplement with pseudo-labeled (query, relevant answer) pairs from the broader MSE corpus using same-thread heuristics.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Many MSE images are low-value (screenshots of errors) | Filter during image extraction; skip posts where all images are tiny or non-mathematical |
| Formula extraction from HTML is noisy | Use robust LaTeX parsers; validate on a sample |
| Dataset construction dominates effort | Automate pipeline early; document schema clearly |

---

## 8. Open Questions

1. **Chunk granularity:** Modality-specific vs. hybrid chunks — which to implement first?
2. **Formula representation:** Store as LaTeX string only, or also OPT/SLT trees? (Trees needed for structure-aware encoder; LaTeX sufficient for baseline.)
3. **Negative sampling for retrieval training:** In-batch negatives from ARQMath corpus? Hard negatives mined from BM25?
4. **Image relevance labels:** ARQMath qrels cover answer posts, not individual images within posts. How do we assign relevance to specific images?
