"""
Live Formula Retrieval Engine (Inference Pipeline)

This module provides the FormulaRetriever class, designed to be instantiated 
once by the Tri-RAG routing agent. It keeps models loaded in memory and 
exposes a .search() method to instantly process live LaTeX queries.
"""

import gc
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import subprocess
import numpy as np
import pyarrow.parquet as pq
import torch
import faiss
import bm25s
import sqlite3
import latex2mathml.converter
from torch_geometric.data import Batch

# Internal project imports
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.task3.model.formula_encoder import DualFormulaEncoder
from src.data.formula_graph import opt_to_pyg, slt_to_pyg

class FormulaRetriever:
    def __init__(self):
        """Initializes the engine and loads all neural/sparse indices into memory."""
        print("Initializing Formula Retrieval Engine...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Paths ---
        self._CHECKPOINT_PATH = _PROJECT_ROOT / "checkpoints/task3/phase3_fusion/phase3_atten_fusion_ensemble_soup.pt"
        self._FAISS_INDEX_PATH = _PROJECT_ROOT / "checkpoints/task3/faiss_index/phase3_dense.faiss"
        self._FAISS_IDS_PATH = _PROJECT_ROOT / "checkpoints/task3/faiss_index/phase3_corpus_ids.npy"
        self._BM25_DIR = _PROJECT_ROOT / "checkpoints/task3/bm25_index"
        self._DB_PATH = _PROJECT_ROOT / "data/processed/formula_cache.db"

        # --- Hyperparameters ---
        self.ALPHA = 0.80  
        self.TOP_K_DENSE = 1000
        self.MAX_PATH_DEPTH = 4  
        self.K_RRF_DENSE = 60          
        self.K_RRF_STRUCT = 15  
        self.IGNORE_TAGS: Set[str] = {
            'mrow', 'mstyle', 'mpadded', 'mphantom', 'maligngroup', 'malignmark',
            'semantics', 'annotation', 'annotation-xml', 'id', 'mspace', 'menclose', 'maction'
        }

        self._load_models()

    def _load_models(self):
        """Loads FAISS, BM25, and the GNN into VRAM."""
        print(f"Loading Dual Encoder to {self.device}...")
        self.encoder = DualFormulaEncoder.load(self._CHECKPOINT_PATH, map_location=self.device).to(self.device)
        self.encoder.eval()

        print("Loading FAISS Index...")
        self.faiss_index = faiss.read_index(str(self._FAISS_INDEX_PATH))
        self.faiss_ids = np.load(str(self._FAISS_IDS_PATH))

        print("Loading BM25 Sparse Index...")
        self.bm25_retriever = bm25s.BM25.load(str(self._BM25_DIR), load_corpus=True)
        print("Engine Ready.\n")

    # ==========================================
    # HELPER FUNCTIONS
    # ==========================================

    def _parse_latex(self, latex_str: str) -> Dict[str, str]:
        """
        Converts raw LaTeX to MathML XML using a local LaTeXML installation.
        Safely generates both Presentation MathML (SLT) and Content MathML (OPT).
        """
        result = {"opt": "", "slt": ""}
        
        try:
            # We pass the LaTeX via stdin (input=latex_str) and use the "-" flag.
            # This completely avoids dealing with shell escaping and special characters in LaTeX.

            # Generate Presentation MathML (SLT)
            slt_process = subprocess.run(
                ["latexmlmath", "--pmml=-", "-"],
                input=latex_str,
                text=True,
                capture_output=True,
                check=True
            )
            result["slt"] = slt_process.stdout.strip()

            # Generate Content MathML / Operator Tree (OPT)
            opt_process = subprocess.run(
                ["latexmlmath", "--cmml=-", "-"],
                input=latex_str,
                text=True,
                capture_output=True,
                check=True
            )
            result["opt"] = opt_process.stdout.strip()

        except subprocess.CalledProcessError as e:
            print(f"LaTeXML Parsing Error on query '{latex_str}':\nSTDERR: {e.stderr}")
        except FileNotFoundError:
            print("CRITICAL ERROR: 'latexmlmath' is not installed on this system.")
            print("Please run: sudo apt install latexml")
            
        return result

    def _min_max_normalize(self, scores_dict: dict) -> dict:
        if not scores_dict: return {}
        vals = list(scores_dict.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v: return {k: 1.0 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

    def _extract_math_tokens(self, xml_str: str) -> list:
        if not xml_str: return []
        raw_tokens = re.findall(r'>\s*([^<]+?)\s*<', xml_str)
        return [t.strip() for t in raw_tokens if t.strip()]

    def _get_depth_weight(self, path_token: str) -> float:
        return float(len(path_token.split("::")) - 1) ** 1.5

    def _extract_structural_paths(self, xml_str: str) -> List[str]:
        if not xml_str: return []
        xml_str = re.sub(r'\sxmlns="[^"]+"', '', xml_str, count=1)
        try:
            root = ET.fromstring(xml_str)
        except Exception:
            return []
            
        def get_shape_string(node: ET.Element, mask_vars: bool = True) -> str:
            tag = node.tag.split('}')[-1]
            if tag in self.IGNORE_TAGS:
                return "".join(get_shape_string(child, mask_vars) for child in node)
            if mask_vars and tag in {'mi', 'ci', 'identifier'}:
                res = f"<{tag}>V</{tag}>"
            else:
                res = f"<{tag}>{(node.text or '').strip()}</{tag}>"
            for child in node: res += get_shape_string(child, mask_vars)
            return res

        def canonicalize_tree(node: ET.Element):
            for child in list(node): canonicalize_tree(child)
            tag = node.tag.split('}')[-1]
            if tag == 'apply' and len(node) > 1:
                op_tag = node[0].tag.split('}')[-1]
                if op_tag in {'plus', 'times', 'eq', 'and', 'or', 'union', 'intersect', 'equivalent'}:
                    op_node = node[0]
                    args = list(node)[1:]
                    args.sort(key=lambda n: (get_shape_string(n, True), get_shape_string(n, False)))
                    node[:] = [op_node] + args

        canonicalize_tree(root)
            
        paths = []
        var_map, var_counter = {}, 1
        
        def dfs(node: ET.Element, path_exact: List[str], path_alpha: List[str], path_univ: List[str], edge_pos: str = "R"):
            nonlocal var_counter
            tag = node.tag.split('}')[-1]
            
            if tag in self.IGNORE_TAGS:
                for idx, child in enumerate(node): dfs(child, path_exact, path_alpha, path_univ, edge_pos)
                return

            text = (node.text or "").strip()
            base_repr = f"{edge_pos}_{tag}"
            repr_exact, repr_alpha, repr_univ = base_repr, base_repr, base_repr
            
            if text and len(text) < 15:
                clean_text = re.sub(r'\W+', '', text)
                if clean_text:
                    if tag in ['mi', 'ci', 'identifier']:
                        repr_exact = f"{base_repr}_{clean_text}"
                        if clean_text not in var_map:
                            var_map[clean_text] = f"V{var_counter}"
                            var_counter += 1
                        repr_alpha = f"{base_repr}_{var_map[clean_text]}"
                        repr_univ = f"{base_repr}_V"
                    else:
                        repr_exact = repr_alpha = repr_univ = f"{base_repr}_{clean_text}"
                    
            pe, pa, pu = path_exact + [repr_exact], path_alpha + [repr_alpha], path_univ + [repr_univ]
            
            for i in range(1, min(len(pe), self.MAX_PATH_DEPTH) + 1):
                paths.extend(["E::" + "::".join(pe[-i:]), "E::" + "::".join(pe[-i:])]) 
                paths.extend(["A::" + "::".join(pa[-i:]), "U::" + "::".join(pu[-i:])])
                
            for idx, child in enumerate(node): dfs(child, pe, pa, pu, str(idx))
                
        dfs(root, [], [], [], "R")
        return paths

    # ==========================================
    # CORE SEARCH PIPELINE
    # ==========================================
    def search(self, latex_query: str, final_top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Executes the full pipeline on a live LaTeX string.
        Returns a list of tuples: (Visual_ID, Fusion_Score)
        """
        print(f"Processing Query: {latex_query}")
        
        # Parsing
        # q_xml = self._parse_latex(latex_query)
        returned_xml = self._parse_latex(latex_query)
        q_xml = {"opt": returned_xml["opt"], "slt": returned_xml["slt"]}
        
        if not q_xml["opt"]:
            return []

        # Dense GNN Search
        o_g, s_g = opt_to_pyg(q_xml["opt"]), slt_to_pyg(q_xml["slt"])
        dense_scores = {}
        if o_g is not None and s_g is not None:
            with torch.no_grad():
                o_batch, s_batch = Batch.from_data_list([o_g]).to(self.device), Batch.from_data_list([s_g]).to(self.device)
                q_emb = self.encoder(o_batch, s_batch, normalize=True).cpu().numpy()
                
            D_dense, I_dense = self.faiss_index.search(q_emb, self.TOP_K_DENSE)
            dense_scores = {str(self.faiss_ids[idx]): float(score) for score, idx in zip(D_dense[0], I_dense[0]) if idx != -1}

        # Sparse Token Search
        query_tokens = self._extract_math_tokens(q_xml["slt"])
        sparse_scores = {}
        if query_tokens:
            docs, D_sparse = self.bm25_retriever.retrieve([query_tokens], k=self.TOP_K_DENSE)
            for doc, score in zip(docs[0], D_sparse[0]):
                vid = str(doc.get("text", doc.get("id", ""))) if isinstance(doc, dict) else str(getattr(doc, "id", doc))
                if vid: sparse_scores[vid] = float(score)

        # Phase 3 Alpha Fusion
        norm_dense, norm_sparse = self._min_max_normalize(dense_scores), self._min_max_normalize(sparse_scores)
        hybrid_scores = {vid: (self.ALPHA * norm_dense.get(vid, 0.0)) + ((1.0 - self.ALPHA) * norm_sparse.get(vid, 0.0))
                         for vid in set(norm_dense.keys()) | set(norm_sparse.keys())}
        
        top_1000_vids = dict(sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:self.TOP_K_DENSE])

        # Fetch XMLs for Phase 4 via SQLite Cache
        xml_cache = {}
        vids_to_fetch = list(top_1000_vids.keys())
        
        try:
            with sqlite3.connect(self._DB_PATH) as conn:
                cursor = conn.cursor()
                
                # SQLite has a ~999 variable limit per query. Chunking handles this safely.
                chunk_size = 900 
                for i in range(0, len(vids_to_fetch), chunk_size):
                    chunk = vids_to_fetch[i : i + chunk_size]
                    placeholders = ','.join('?' for _ in chunk)
                    
                    cursor.execute(f"""
                        SELECT visual_id, opt, slt 
                        FROM formulas 
                        WHERE visual_id IN ({placeholders})
                    """, chunk)
                    
                    for row in cursor.fetchall():
                        xml_cache[str(row[0])] = {"opt": row[1], "slt": row[2]}
        except Exception as e:
            print(f"SQLite XML Cache Error: {e}")

        # Phase 4 Structural Re-ranking
        q_tokens = self._extract_structural_paths(q_xml["opt"]) + self._extract_structural_paths(q_xml["slt"])
        if not q_tokens:
            return list(top_1000_vids.items())[:final_top_k]

        valid_doc_vids, doc_tokens_list = [], []
        for vid in top_1000_vids.keys():
            if vid in xml_cache:
                d_tokens = self._extract_structural_paths(xml_cache[vid]["opt"]) + self._extract_structural_paths(xml_cache[vid]["slt"])
                if d_tokens:
                    doc_tokens_list.append(d_tokens)
                    valid_doc_vids.append(vid)
                    
        n_docs = len(valid_doc_vids)
        df_counts = defaultdict(int)
        for tokens in doc_tokens_list:
            for t in set(tokens): df_counts[t] += 1
        idf = {t: math.log(1.0 + (n_docs - f + 0.5) / (f + 0.5)) for t, f in df_counts.items()}
        
        q_counts = Counter(q_tokens)
        q_total_tokens = sum(q_counts.values())
        max_q_score = sum(count * idf.get(t, 1.0) * self._get_depth_weight(t) for t, count in q_counts.items()) or 1.0
        
        structural_scores = {}
        for vid, d_tokens in zip(valid_doc_vids, doc_tokens_list):
            d_counts = Counter(d_tokens)
            d_total_tokens = sum(d_counts.values())
            
            overlap_score = 0.0
            for t, count in q_counts.items():
                matched = min(count, d_counts.get(t, 0))
                overlap_score += matched * idf.get(t, 1.0) * self._get_depth_weight(t)
                
            length_ratio = q_total_tokens / max(1.0, d_total_tokens)
            length_penalty = min(1.0, length_ratio) ** 0.25
            
            structural_scores[vid] = (overlap_score / max_q_score) * length_penalty
        
        struct_ranked = sorted(structural_scores.keys(), key=lambda v: structural_scores[v], reverse=True)
        dense_ranked = list(top_1000_vids.keys()) 
        
        fused_scores = defaultdict(float)
        for rank, vid in enumerate(dense_ranked): fused_scores[vid] += 1.0 / (self.K_RRF_DENSE + rank + 1)
        for rank, vid in enumerate(struct_ranked): fused_scores[vid] += 3.0 / (self.K_RRF_STRUCT + rank + 1)  
            
        final_ranking = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:final_top_k]
        return final_ranking

# Example usage when run as a script
if __name__ == "__main__":
    retriever = FormulaRetriever()
    
    query = r"\int_{0}^{\infty} e^{-x^2} dx"
    results = retriever.search(query, final_top_k=5)
    
    print(f"\nTop Results for '{query}':")
    for rank, (vid, score) in enumerate(results, 1):
        print(f"Rank {rank} | VID: {vid} | Score: {score:.4f}")