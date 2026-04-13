"""
Results Visualizer Dashboard

A Streamlit app to interactively explore the top results from the Structural Re-ranker.
Reads the raw SLT (Presentation MathML) and renders it natively in the browser,
supplemented with analytical metrics for retrieval confidence.
"""

import sys
import json
import pandas as pd
import streamlit as st
import pyarrow.parquet as pq
from pathlib import Path

# Run using `streamlit run src/task3/visualize_results.py` from the project root
# Paths
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

_RUN_PATH = _PROJECT_ROOT / "data/processed/end_to_end_run.json"
_PARQUET_DIR = _PROJECT_ROOT / "data/processed/formula_index"

from src.task3.dataset import load_qrels

st.set_page_config(page_title="Retrieval Visualizer", layout="wide")

@st.cache_data
def load_dashboard_data():
    """Loads the run data and extracts only the needed MathML from the 8.3M corpus."""
    
    # Load the run results
    with open(_RUN_PATH, "r") as f:
        run_data = json.load(f)

    # Get Proxy IDs for queries
    qrels = load_qrels("eval")  
    proxy_vid_to_topic = {}
    for topic_id, items in qrels.items():
        positives = [str(vid) for vid, grade in items.items() if grade >= 2.0]
        if positives:
            proxy_vid_to_topic[positives[0]] = topic_id
            
    topic_to_proxy_vid = {v: k for k, v in proxy_vid_to_topic.items()}

    # Collect all visual_ids we need to extract (Queries + Top K Results)
    needed_vids = set(proxy_vid_to_topic.keys())
    top_k = 10
    topic_top_results = {}

    for topic_id, results in run_data.items():
        sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        topic_top_results[topic_id] = sorted_res
        for vid, score in sorted_res:
            needed_vids.add(str(vid))

    # Scan Parquets to extract the Presentation MathML
    extracted_math = {}
    shard_files = sorted(list(_PARQUET_DIR.glob("shard_*.parquet")))

    progress_bar = st.progress(0, text="Scanning Parquet Shards for MathML...")
    
    for i, shard in enumerate(shard_files):
        df = pq.read_table(shard, columns=["visual_id", "slt"]).to_pandas()
        df = df.dropna(subset=["slt"])
        
        mask = df["visual_id"].astype(str).isin(needed_vids)
        target_df = df[mask]

        for _, row in target_df.iterrows():
            extracted_math[str(row["visual_id"])] = row["slt"]

        progress_bar.progress(
            (i + 1) / len(shard_files), 
            text=f"Extracting MathML... ({len(extracted_math)}/{len(needed_vids)} found)"
        )

        if len(extracted_math) == len(needed_vids):
            break

    progress_bar.empty()
    return run_data, topic_to_proxy_vid, topic_top_results, extracted_math

# ==========================================
# UI RENDERING & ANALYTICS
# ==========================================
st.title("Formula RAG: Retrieval Dashboard")
st.markdown("Exploring the Top 10 results retrieved by the Structural Re-ranker.")

with st.spinner("Initializing Data (This will take ~60 seconds on the first run)..."):
    run_data, topic_to_proxy_vid, topic_top_results, extracted_math = load_dashboard_data()

# --- GLOBAL METRICS CALCULATION ---
avg_scores_dict = {}
for t_id, res in topic_top_results.items():
    if res:
        avg_scores_dict[t_id] = sum(score for _, score in res) / len(res)

# Convert to DataFrame for easier Streamlit charting
df_global_metrics = pd.DataFrame(
    list(avg_scores_dict.items()), 
    columns=["Topic ID", "Average Score"]
).set_index("Topic ID")

# Global Chart Expander
with st.expander("📊 Global Analytics: Top 10 Average Score per Query", expanded=False):
    st.markdown("Identifies which topics have the strongest overall structural alignments.")
    st.bar_chart(df_global_metrics, y="Average Score", color="#4CAF50")

# --- SIDEBAR & TOPIC SELECTION ---
available_topics = sorted(list(run_data.keys()))
selected_topic = st.sidebar.selectbox("Select ARQMath Topic Query:", available_topics, help=f"Total: {len(available_topics)} Topics")

if selected_topic:
    st.divider()
    
    # LOCAL QUERY METRICS (KPI Cards)
    results = topic_top_results.get(selected_topic, [])
    current_avg = avg_scores_dict.get(selected_topic, 0.0)
    max_score = results[0][1] if results else 0.0
    
    colA, colB, colC = st.columns(3)
    colA.metric(label="Average Top 10 Score", value=f"{current_avg:.4f}")
    colB.metric(label="Maximum Score (Rank 1)", value=f"{max_score:.4f}")
    colC.metric(label="Candidates Retrieved", value=len(results))
    
    st.divider()

    # DISPLAY TARGET QUERY
    query_vid = topic_to_proxy_vid.get(selected_topic)
    query_math = extracted_math.get(query_vid, "<i>Query MathML not found in shards</i>")
    
    st.subheader(f"Target Query (Topic {selected_topic})")
    st.info("What the user searched for:")
    st.markdown(f"<div style='font-size: 1.5em;'>{query_math}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.subheader("Top 10 Retrieved Formulas")
    
    # DISPLAY RESULTS
    for rank, (vid, score) in enumerate(results, 1):
        math_slt = extracted_math.get(vid, "<i>MathML missing</i>")
        
        # Calculate Delta from Average
        score_delta = score - current_avg
        
        with st.container():
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"<h3 style='margin-bottom: 0px; padding-bottom: 0px;'>Rank #{rank}</h3>", unsafe_allow_html=True)
                
                # Show score and delta compared to the query average
                if score_delta >= 0:
                    delta_color = "green"
                    delta_sign = "+"
                else:
                    delta_color = "red"
                    delta_sign = ""
                
                st.markdown(
                    f"<div style='font-size: 0.85em; font-weight: bold;'>Score: {score:.4f}</div>"
                    f"<div style='font-size: 0.75em; color: {delta_color}; margin-top: -5px;'>{delta_sign}{score_delta:.4f} vs avg</div>", 
                    unsafe_allow_html=True
                )
                st.caption(f"VID: {vid}")
            with col2:
                st.markdown(f"<div style='font-size: 1.3em; padding-top: 10px;'>{math_slt}</div>", unsafe_allow_html=True)
        st.markdown("---")