"""
Results Visualizer Dashboard

A Streamlit app to interactively explore the top results from the Structural Re-ranker.
Features two modes:
1. ARQMath Evaluation: Explores the offline JSON run.
2. Live Search: Sends custom LaTeX queries to the FastAPI backend.
"""

import sys
import json
import sqlite3
import requests
import pandas as pd
import streamlit as st
from pathlib import Path

# Paths
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

_RUN_PATH = _PROJECT_ROOT / "data/processed/end_to_end_run_new.json"
_DB_PATH = _PROJECT_ROOT / "data/processed/formula_cache.db"
_API_URL = "http://0.0.0.0:8567/search"

from src.task3.dataset import load_qrels

st.set_page_config(page_title="Formula RAG Visualizer", layout="wide")

# ==========================================
# DATA LOADING (SQLite Optimized)
# ==========================================
@st.cache_data
def load_eval_data():
    """Loads the offline run data for the benchmark tab."""
    if not _RUN_PATH.exists():
        return {}, {}

    with open(_RUN_PATH, "r") as f:
        run_data = json.load(f)

    qrels = load_qrels("eval")  
    proxy_vid_to_topic = {}
    for topic_id, items in qrels.items():
        positives = [str(vid) for vid, grade in items.items() if grade >= 2.0]
        if positives:
            proxy_vid_to_topic[positives[0]] = topic_id
            
    topic_to_proxy_vid = {v: k for k, v in proxy_vid_to_topic.items()}

    topic_top_results = {}
    for topic_id, results in run_data.items():
        topic_top_results[topic_id] = sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]

    return topic_to_proxy_vid, topic_top_results

def fetch_mathml_from_sqlite(vids: list) -> dict:
    """Instantly fetches MathML strings from the SQLite cache."""
    extracted_math = {}
    if not vids or not _DB_PATH.exists():
        return extracted_math
        
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' for _ in vids)
            cursor.execute(f"SELECT visual_id, slt FROM formulas WHERE visual_id IN ({placeholders})", vids)
            for row in cursor.fetchall():
                extracted_math[str(row[0])] = row[1]
    except Exception as e:
        st.error(f"Database Error: {e}")
        
    return extracted_math

def render_result_card(rank, vid, score, math_slt, current_avg=None):
    """Helper to render a clean UI card for a retrieved formula."""
    with st.container():
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(f"<h3 style='margin-bottom: 0px; padding-bottom: 0px;'>Rank #{rank}</h3>", unsafe_allow_html=True)
            
            if current_avg is not None:
                score_delta = score - current_avg
                delta_color = "green" if score_delta >= 0 else "red"
                delta_sign = "+" if score_delta >= 0 else ""
                st.markdown(
                    f"<div style='font-size: 0.85em; font-weight: bold;'>Score: {score:.4f}</div>"
                    f"<div style='font-size: 0.75em; color: {delta_color}; margin-top: -5px;'>{delta_sign}{score_delta:.4f} vs avg</div>", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"<div style='font-size: 0.85em; font-weight: bold;'>Score: {score:.4f}</div>", unsafe_allow_html=True)
                
            st.caption(f"VID: {vid}")
        with col2:
            st.markdown(f"<div style='font-size: 1.3em; padding-top: 10px;'>{math_slt}</div>", unsafe_allow_html=True)
    st.markdown("---")


# ==========================================
# UI LAYOUT
# ==========================================
st.title("Formula RAG: Retrieval Dashboard")
st.markdown(f"Explore the Tri-RAG Dual Encoder Engine from {_RUN_PATH.name}")

tab1, tab2 = st.tabs(["Live Custom Search", "ARQMath Evaluation Benchmark"])

# ---------------------------------------------------------
# TAB 1: LIVE SEARCH (Queries the FastAPI Backend)
# ---------------------------------------------------------
with tab1:
    st.subheader("Test Live Inference")
    st.markdown("Enter a LaTeX formula to query the 8.3M corpus in real-time. *(Ensure the FastAPI server is running!)*")
    
    with st.form("search_form"):
        col_query, col_k = st.columns([4, 1])
        with col_query:
            user_query = st.text_input("LaTeX Query:", value=r"\int_{0}^{\infty} e^{-x^2} dx")
        with col_k:
            top_k_val = st.number_input("Top K:", min_value=1, max_value=50, value=10)
            
        submitted = st.form_submit_button("Search 🔍", type="primary")

    if submitted and user_query:
        with st.spinner("Querying API..."):
            try:
                response = requests.post(_API_URL, json={"query": user_query, "top_k": top_k_val}, timeout=30)
                if response.status_code == 200:
                    api_data = response.json()
                    results = api_data.get("results", [])
                    
                    if not results:
                        st.warning("No structural matches found.")
                    else:
                        st.success(f"Found {len(results)} matches!")
                        
                        # Fetch XMLs for the results
                        result_vids = [r["visual_id"] for r in results]
                        math_dict = fetch_mathml_from_sqlite(result_vids)
                        
                        # Render
                        for res in results:
                            slt = math_dict.get(res["visual_id"], "<i>MathML not found</i>")
                            render_result_card(res["rank"], res["visual_id"], res["score"], slt)
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to API. Please ensure `python src/task3/api_server.py` is running in another terminal.")

# ---------------------------------------------------------
# TAB 2: ARQMATH BENCHMARK (Reads offline JSON)
# ---------------------------------------------------------
with tab2:
    topic_to_proxy_vid, topic_top_results = load_eval_data()
    
    if not topic_top_results:
        st.warning("Evaluation run data not found. Please run the evaluation script first.")
    else:
        available_topics = sorted(list(topic_top_results.keys()))
        selected_topic = st.selectbox("Select ARQMath Topic Query:", available_topics)

        if selected_topic:
            st.divider()
            results = topic_top_results.get(selected_topic, [])
            
            # Fetch all needed MathML (Query + Top 10)
            needed_vids = [vid for vid, score in results]
            query_vid = topic_to_proxy_vid.get(selected_topic)
            if query_vid: needed_vids.append(query_vid)
            
            math_dict = fetch_mathml_from_sqlite(needed_vids)

            # Metrics
            avg_score = sum(score for _, score in results) / len(results) if results else 0.0
            max_score = results[0][1] if results else 0.0
            
            colA, colB, colC = st.columns(3)
            colA.metric(label="Average Top 10 Score", value=f"{avg_score:.4f}")
            colB.metric(label="Maximum Score (Rank 1)", value=f"{max_score:.4f}")
            colC.metric(label="Candidates Retrieved", value=len(results))
            
            st.divider()
            
            # Display Target Query
            query_math = math_dict.get(query_vid, "<i>Query MathML not found</i>")
            st.subheader(f"Target Query (Topic {selected_topic})")
            st.info("What the user searched for:")
            st.markdown(f"<div style='font-size: 1.5em;'>{query_math}</div>", unsafe_allow_html=True)
            
            st.divider()
            st.subheader("Top 10 Retrieved Formulas")
            
            # Display Results
            for rank, (vid, score) in enumerate(results, 1):
                math_slt = math_dict.get(vid, "<i>MathML missing</i>")
                render_result_card(rank, vid, score, math_slt, current_avg=avg_score)