"""
Tri-Search RAG Visualizer Dashboard

Streamlit app to explore retrieval results from three modalities before
they reach the LLM:
  - Text    — OpenSearch KNN via TextHandler
  - Formula — GNN dual-encoder via gnn_handler
  - Images  — LLM2CLIP via clip_handler

Run from the orchestrator directory:
    streamlit run visualize_rag.py
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

load_dotenv(str(_DIR / ".env"))

from clip_handler import fetch_clip_results
from gnn_handler import fetch_gnn_results
from tri_search import build_prompt, text_sources

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
st.markdown("Explore what each retrieval modality returns before it reaches the LLM.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Query")
query = st.sidebar.text_area(
    "Enter a math question:",
    value=r"Solve the quadratic equation: $x^2 - 5x + 6 = 0$",
    height=100,
)

st.sidebar.header("Retrieval Settings")
top_k_text     = st.sidebar.slider("Text results (k)",    0, 20, 5)
top_k_formulas = st.sidebar.slider("Formula results (k)", 0, 20, 5)
top_k_images   = st.sidebar.slider("Image results (k)",   0, 10, 3)

run_llm = st.sidebar.checkbox(
    "Run LLM and show final answer",
    value=False,
    help="Requires the Llama model to be loaded. Slow on first run.",
)

run_btn = st.sidebar.button("Run Retrieval", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render_formula_hits(hits: List[dict]) -> None:
    """Display formula hits as LaTeX with rank/score/metadata."""
    if not hits:
        st.info("No formula matches found.")
        return
    for hit in hits:
        latex    = hit.get("latex", "")
        score    = hit.get("score", 0.0)
        rank     = hit.get("rank", "?")
        post_id  = hit.get("post_id", "")
        url      = hit.get("url", "")

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
                if post_id:
                    st.caption(f"Post: {post_id}")
                if url:
                    st.markdown(f"[Source]({url})")
        st.markdown("---")


def _render_image_hits(hits: List[dict]) -> None:
    """Display image hits in a 3-column grid with score and source link."""
    if not hits:
        st.info("No image matches found.")
        return
    cols = st.columns(min(len(hits), 3))
    for i, hit in enumerate(hits):
        with cols[i % 3]:
            image_path = hit.get("file_path", "")
            title      = hit.get("title", "")
            url        = hit.get("url", "")
            score      = hit.get("score", 0.0)
            rank       = hit.get("rank", i + 1)

            st.markdown(f"**Rank {rank}** — score: `{score:.4f}`")
            if image_path and Path(image_path).exists():
                st.image(image_path, caption=title or image_path, use_container_width=True)
            else:
                st.caption(f"*(image not on this machine: {image_path})*")
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

    tab_text, tab_formula, tab_image, tab_rag = st.tabs(
        ["📄 Text", "∑ Formulas", "🖼 Images", "🤖 RAG Answer"]
    )

    # Collect results for the RAG tab.
    text_context    = ""
    formula_context = ""
    image_context   = ""

    # ---- Text ---------------------------------------------------------------
    with tab_text:
        st.subheader("Text Retrieval (OpenSearch / sentence-transformer)")
        if top_k_text == 0:
            st.info("Text retrieval disabled (k=0).")
        else:
            with st.spinner("Querying text index..."):
                try:
                    text_context = text_sources(query, top_k=top_k_text)
                    if text_context:
                        st.markdown(text_context)
                    else:
                        st.info("No text results returned.")
                except Exception as e:
                    st.error(f"Text retrieval failed: {e}")

    # ---- Formulas -----------------------------------------------------------
    with tab_formula:
        st.subheader("Formula Retrieval (GNN dual-encoder)")
        if top_k_formulas == 0:
            st.info("Formula retrieval disabled (k=0).")
        else:
            with st.spinner("Querying GNN formula index..."):
                formula_hits = fetch_gnn_results(query, k=top_k_formulas)
                if not formula_hits:
                    st.info("No formula matches found. Make sure the query contains LaTeX.")
                else:
                    _render_formula_hits(formula_hits)
                    # Build context for RAG from the scraped post text.
                    formula_context = "\n\n".join(
                        f"[Formula source: {h['url']}]\n{h['scraped_text']}"
                        for h in formula_hits
                        if h.get("scraped_text")
                    )

    # ---- Images -------------------------------------------------------------
    with tab_image:
        st.subheader("Image Retrieval (LLM2CLIP)")
        if top_k_images == 0:
            st.info("Image retrieval disabled (k=0).")
        else:
            with st.spinner("Querying image index..."):
                image_hits = fetch_clip_results(query, k=top_k_images)
                if not image_hits:
                    st.info("No image matches found.")
                else:
                    _render_image_hits(image_hits)
                    # Build context for RAG from the scraped post text.
                    image_context = "\n\n".join(
                        f"[Image source: {h.get('title') or h['url']}]\n{h['scraped_text']}"
                        for h in image_hits
                        if h.get("scraped_text")
                    )

    # ---- RAG Answer ---------------------------------------------------------
    with tab_rag:
        st.subheader("Assembled Prompt & LLM Answer")

        prompt = build_prompt(
            query,
            text_context=text_context,
            formula_context=formula_context,
            image_context=image_context,
        )

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
