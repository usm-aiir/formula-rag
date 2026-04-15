"""
Tri-Search RAG Visualizer Dashboard

A Streamlit app to interactively explore retrieval results from all three
modalities: text (OpenSearch), formula (TangentCFT), and image (CLIP).

Run from the tri-search directory:
    streamlit run visualize_rag.py
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import List

import streamlit as st

_TRISEARCH_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TRISEARCH_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_TRISEARCH_DIR))

from dotenv import load_dotenv

load_dotenv(str(_TRISEARCH_DIR / ".env"))

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tri-Search RAG Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Tri-Search RAG Visualizer")
st.markdown(
    "Interactively explore what each retrieval modality returns for a query "
    "before it reaches the LLM."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Query")
query = st.sidebar.text_area(
    "Enter a math question:",
    value=r"Solve the quadratic equation: $x^2 - 5x + 6 = 0$",
    height=100,
)

st.sidebar.header("Retrieval Settings")
top_k_text = st.sidebar.slider("Text results (k)", 0, 20, 5)
top_k_formulas = st.sidebar.slider("Formula results (k)", 0, 20, 5)
top_k_images = st.sidebar.slider("Image results (k)", 0, 10, 3)

st.sidebar.header("Active Formula Handler")
formula_backend = st.sidebar.radio(
    "Formula retrieval backend:",
    options=["TangentCFT", "GNN (Task 3)"],
    index=0,
    help="TangentCFT uses the formula-search repo. GNN requires the Task 3 checkpoint pipeline.",
)

run_llm = st.sidebar.checkbox(
    "Run LLM and show final answer",
    value=False,
    help="Requires the Llama model to be downloadable. Slow on first run.",
)

run_btn = st.sidebar.button("Run Retrieval", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_formula_hits(hits: List[dict], top_k: int = 50) -> None:
    if not hits:
        st.info("No formula matches found.")
        return
    shown = 0
    for hit in hits:
        if shown >= top_k:
            break
        latex = hit.get("returned_formula") or hit.get("latex") or ""
        if not latex or len(latex.strip()) <= 1 or latex.strip().isalpha():
            continue
        shown += 1
        score = hit.get("score", 0.0)
        rank = hit.get("rank", "?")
        vid = hit.get("mapping_visual_id") or hit.get("visual_id") or ""
        post_id = hit.get("post_id") or ""

        with st.container():
            col_rank, col_math, col_meta = st.columns([1, 5, 2])
            with col_rank:
                st.markdown(f"**Rank {rank}**")
                st.caption(f"Score: {score:.4f}")
            with col_math:
                if latex:
                    st.latex(latex)
                else:
                    st.caption("*(no LaTeX)*")
            with col_meta:
                if vid:
                    st.caption(f"VID: {vid}")
                if post_id:
                    st.caption(f"Post: {post_id}")
        st.markdown("---")


def _render_image_hits(hits: List[dict]) -> None:
    if not hits:
        st.info("No image matches found.")
        return
    cols = st.columns(min(len(hits), 3))
    for i, hit in enumerate(hits):
        col = cols[i % 3]
        with col:
            image_path = hit.get("file_path") or hit.get("path") or ""
            title = hit.get("title", "")
            url = hit.get("url", "")
            score = hit.get("score", 0.0)
            rank = hit.get("rank", i + 1)

            st.markdown(f"**Rank {rank}** — score: `{score:.4f}`")
            if image_path and Path(image_path).exists():
                st.image(
                    image_path, caption=title or image_path, use_container_width=True
                )
            else:
                st.caption(f"*(image not found: {image_path})*")
            if title:
                st.caption(title)
            if url:
                st.markdown(f"[Source]({url})")
            st.markdown("---")


# ---------------------------------------------------------------------------
# Main retrieval run
# ---------------------------------------------------------------------------
if run_btn and query.strip():
    st.divider()

    tab_text, tab_formula, tab_image, tab_prompt = st.tabs(
        ["📄 Text", "∑ Formulas", "🖼 Images", "🤖 LLM Prompt & Answer"]
    )

    # ---- Text ---------------------------------------------------------------
    with tab_text:
        st.subheader("Text Retrieval (OpenSearch / sentence-transformer)")
        if top_k_text == 0:
            st.info("Text retrieval disabled (k=0).")
            text_result = ""
        else:
            with st.spinner("Querying text index..."):
                try:
                    from text_handler import TextHandler

                    text_result = TextHandler().retrieve_relevant_text(
                        query, top_k=top_k_text
                    )
                    if text_result:
                        st.markdown(text_result)
                    else:
                        st.info("No text results returned.")
                except Exception as e:
                    st.error(f"Text retrieval failed: {e}")
                    text_result = ""

    # ---- Formulas -----------------------------------------------------------
    with tab_formula:
        st.subheader(f"Formula Retrieval ({formula_backend})")
        if top_k_formulas == 0:
            st.info("Formula retrieval disabled (k=0).")
            formula_hits: List[dict] = []
        else:
            with st.spinner(f"Querying formula index via {formula_backend}..."):
                try:
                    from formula_utils import extract_formulas

                    detected = extract_formulas(query)
                    if not detected:
                        st.info("No LaTeX formulas detected in the query.")
                        formula_hits = []
                    else:
                        st.caption(f"Detected formulas: {detected}")
                        formula_hits = []

                        if formula_backend == "TangentCFT":
                            from formula_handler import FormulaHandler

                            hits_raw = FormulaHandler(
                                detected
                            ).retrieve_similar_formulas(top_k=top_k_formulas * 3)
                            formula_hits = hits_raw
                        else:
                            import gnn_handler

                            for latex in detected:
                                for h in gnn_handler.search(
                                    latex, k=top_k_formulas * 3
                                ):
                                    formula_hits.append(h)

                        _render_formula_hits(formula_hits, top_k=top_k_formulas)
                except Exception as e:
                    st.error(f"Formula retrieval failed: {e}")
                    formula_hits = []

    # ---- Images -------------------------------------------------------------
    with tab_image:
        st.subheader("Image Retrieval (CLIP)")
        if top_k_images == 0:
            st.info("Image retrieval disabled (k=0).")
            image_hits = []
        else:
            with st.spinner("Querying image index..."):
                try:
                    from image_handler import search as image_search

                    image_hits = image_search(query, k=top_k_images)
                    _render_image_hits(image_hits)
                except Exception as e:
                    st.error(f"Image retrieval failed: {e}")
                    image_hits = []

    # ---- LLM Prompt & Answer ------------------------------------------------
    with tab_prompt:
        st.subheader("Assembled Prompt & LLM Answer")

        # Reconstruct the context blocks the same way tri_search.py does
        text_block = text_result if text_result else "(no text documents retrieved)"

        formula_lines: List[str] = []
        for hit in formula_hits:
            latex = hit.get("returned_formula") or hit.get("latex") or ""
            score = hit.get("score", 0.0)
            if latex:
                formula_lines.append(f"[Formula Match (score={score:.3f})]: ${latex}$")
        formula_block = (
            "\nFormula-matched posts:\n" + "\n".join(formula_lines) + "\n"
            if formula_lines
            else ""
        )

        image_lines: List[str] = []
        for hit in image_hits:
            url = hit.get("url", "")
            title = hit.get("title") or url
            if url:
                image_lines.append(f"[Image source: {title}]")
        image_block = (
            "\nImage-sourced posts:\n" + "\n".join(image_lines) + "\n"
            if image_lines
            else ""
        )

        prompt = f"""<|system|>
You are a mathematical answer engine. Respond with the final answer only — a single expression or number, nothing else. No steps, no explanation, no preamble.
<|user|>
Documents:
{text_block}
{formula_block}{image_block}
Question: {query}
<|assistant|>
"""
        with st.expander("View assembled prompt", expanded=False):
            st.code(prompt, language="text")

        if run_llm:
            with st.spinner("Loading LLM and generating answer..."):
                try:
                    import tri_search

                    tri_search.load_model()
                    answer = tri_search.prompt_model(prompt)
                    st.success("**LLM Answer:**")
                    st.markdown(f"### {answer}")
                except Exception as e:
                    st.error(f"LLM generation failed: {e}")
        else:
            st.info("Enable 'Run LLM' in the sidebar to generate a final answer.")

elif run_btn:
    st.warning("Please enter a query first.")
else:
    st.markdown("> Enter a query in the sidebar and click **Run Retrieval** to begin.")
