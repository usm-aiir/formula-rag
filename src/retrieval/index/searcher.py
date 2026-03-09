"""
Query the FAISS index at inference time.

Given a question (text, images, formulas), encodes each modality with the
same encoders used at index-build time, fuses them, and returns the top-k
most similar answer posts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.encoders import CLIPEncoder, FormulaEncoder

INDEX_DIR = REPO_ROOT / "data/processed/index"


class Searcher:
    """
    Encode a multimodal query and search the FAISS index.

    All three modalities are optional — the searcher fuses whatever is
    present and searches over all indexed answer posts.
    """

    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        formula_ckpt: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load FAISS index
        index_path = index_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_path}. Run build_index.py first."
            )
        self.index = faiss.read_index(str(index_path))

        # Load metadata (parallel to FAISS index rows)
        self.metadata: list[dict] = []
        meta_path = index_dir / "metadata.jsonl"
        with meta_path.open(encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        # Load encoders
        self.clip = CLIPEncoder(device=self.device)
        self.formula_enc = FormulaEncoder(
            device=self.device, clip_dim=self.clip.dim
        )
        if formula_ckpt and formula_ckpt.exists():
            self.formula_enc.load(formula_ckpt)

    def _fuse(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Mean of embeddings, L2-normalised."""
        mean = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-8 else mean

    def encode_query(
        self,
        text: str | None = None,
        images: list[Image.Image | str | Path] | None = None,
        formulas: list[str] | None = None,
    ) -> np.ndarray:
        """
        Encode a multimodal query into a single fused vector.

        At least one of text, images, or formulas must be provided.

        Args:
            text     — question text (LaTeX-stripped)
            images   — PIL Images or paths to image files
            formulas — list of Content MathML strings (OPT)

        Returns:
            L2-normalised numpy vector of shape (dim,)
        """
        parts: list[np.ndarray] = []

        if text:
            emb = self.clip.encode_text_single(text, normalize=True).numpy()
            parts.append(emb)

        if images:
            img_embs = self.clip.encode_images(images, normalize=True).numpy()
            parts.append(img_embs.mean(axis=0))

        if formulas:
            form_embs, _ = self.formula_enc.encode(formulas, normalize=True)
            if form_embs.shape[0] > 0:
                parts.append(form_embs.mean(dim=0).numpy())

        if not parts:
            raise ValueError("At least one of text, images, or formulas must be provided.")

        return self._fuse(parts)

    def search(
        self,
        text: str | None = None,
        images: list[Image.Image | str | Path] | None = None,
        formulas: list[str] | None = None,
        k: int = 10,
    ) -> list[dict]:
        """
        Search for the top-k most relevant answer posts.

        Returns a list of dicts, each containing the metadata fields plus
        a 'score' key (cosine similarity).
        """
        query_vec = self.encode_query(text=text, images=images, formulas=formulas)
        query_vec = query_vec.reshape(1, -1).astype(np.float32)

        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:    # FAISS returns -1 when fewer than k results exist
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)

        return results
