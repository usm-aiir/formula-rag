# LLM2CLIP image index and search handler
# Uses microsoft/LLM2CLIP-Openai-L-14-336 (768-D) with LLaMA-3-8B text encoder (LLM2Vec)
# This is the best retrieval model from the MATVB benchmark. Code derived from matvb
#
# python clipvb_handler.py --build
# python clipvb_handler.py --search "eigenvalue decomposition"
# python clipvb_handler.py --search /path/to/image.png --k 10

import io
import json
from pathlib import Path
from typing import Union

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
import cairosvg

from dataset_handler import iter_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INDEX_DIR  = Path(__file__).parent / "data" / "llm2clip_index"
INDEX_FILE = INDEX_DIR / "mathimages.index"
META_FILE  = INDEX_DIR / "metadata.json"
ITM_MODEL_NAME = "microsoft/LLM2CLIP-Openai-L-14-336"
LLM_MODEL_NAME = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
CLIP_PROC_NAME = "openai/clip-vit-large-patch14-336"

_itm_model    = None
_llm_model    = None
_llm_tokenizer = None
_processor    = None


def _load_itm():
    """Load the vision model and processor. Required for image encoding."""
    global _itm_model, _processor
    if _itm_model is not None:
        return
    print("[llm2clip] loading image model...")
    _processor = CLIPImageProcessor.from_pretrained(CLIP_PROC_NAME)
    _itm_model = AutoModel.from_pretrained(
        ITM_MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE).eval()
    print("[llm2clip] image model ready")


def _mean_pool(model_output, attention_mask):
    """Mean pool over token embeddings, ignoring padding."""
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _load_llm() -> None:
    """Load the LLaMA text encoder. Required for text queries (also loads ITM model)."""
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return
    _load_itm()
    print("[llm2clip] loading text model (~16 GB)...")
    _llm_model = AutoModel.from_pretrained(
        LLM_MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    print("[llm2clip] text model ready")


def _open_image(path: Union[str, Path]) -> Image.Image:
    path = str(path)
    if path.lower().endswith(".svg"):
        png_bytes = cairosvg.svg2png(url=path, output_width=336, output_height=336)
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return Image.open(path).convert("RGB")


def encode_image(image_path: Union[str, Path]) -> np.ndarray:
    """Encode a single image file to an L2-normalised vector. Returns shape (1, D)."""
    _load_itm()
    pil = _open_image(image_path)
    pixel_values = _processor(images=pil, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad(), torch.amp.autocast(device_type=DEVICE):
        vec = _itm_model.get_image_features(pixel_values)
    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().float().numpy()


def encode_text(query: str) -> np.ndarray:
    """Encode a text query to an L2-normalised vector. Returns shape (1, D)."""
    _load_llm()
    inputs = _llm_tokenizer(
        [query], return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        out = _llm_model(**inputs)
    llm_embed = _mean_pool(out, inputs["attention_mask"]).to(DEVICE, dtype=torch.bfloat16)
    with torch.no_grad(), torch.amp.autocast(device_type=DEVICE):
        vec = _itm_model.get_text_features(llm_embed)
    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().float().numpy()


def build_index(limit: Union[int, None] = None) -> None:
    """
    Encode every image in the MathImages dataset and save a FAISS index + metadata.

    Args:
        limit: only process the first N images (useful for a quick smoke test).
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _load_itm()

    index    = None  # created after first encode so dimension is detected automatically
    metadata = []

    for i, entry in enumerate(iter_dataset()):
        if limit is not None and i >= limit:
            break
        try:
            vec = encode_image(entry.image_path)
        except Exception as e:
            print(f"[skip] {entry.image_id}: {e}")
            continue

        if index is None:
            dim = vec.shape[-1]
            print(f"[llm2clip] embedding dim = {dim}")
            index = faiss.IndexFlatIP(dim)

        index.add(vec)
        metadata.append({
            "image_id":  entry.image_id,
            "source":    entry.source,
            "title":     entry.title,
            "url":       entry.url,
            "file_path": str(entry.image_path),
        })

        if (i + 1) % 500 == 0:
            print(f"  encoded {i + 1} images")

    if index is None or index.ntotal == 0:
        print("No images were encoded. Index not saved.")
        return

    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(metadata, indent=2))
    print(f"\nDone. Indexed {index.ntotal} images -> {INDEX_FILE}")


def search(query: Union[str, Path], k: int = 5) -> list:
    """
    Find the k most similar images to a text query or an image file.

    Args:
        query: text string or path to an image file.
        k:     number of results to return.

    Returns:
        List of dicts with keys: rank, score, image_id, source, title, url, file_path.
    """
    if Path(str(query)).exists():
        query_vec = encode_image(query)
    else:
        query_vec = encode_text(str(query))

    index    = faiss.read_index(str(INDEX_FILE))
    metadata = json.loads(META_FILE.read_text())

    scores, indices = index.search(query_vec, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        results.append({"rank": rank, "score": float(score), **metadata[idx]})
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM2CLIP image index builder and searcher")
    parser.add_argument("--build",  action="store_true", help="Build the FAISS index.")
    parser.add_argument("--force",  action="store_true", help="Rebuild even if index already exists.")
    parser.add_argument("--limit",  type=int, default=None, metavar="N",
                        help="Only index the first N images (smoke test).")
    parser.add_argument("--search", metavar="QUERY",
                        help="Text query or image path to search with.")
    parser.add_argument("--k",      type=int, default=5, metavar="K",
                        help="Number of results to return (default: 5).")
    args = parser.parse_args()

    if args.build:
        if INDEX_FILE.exists() and not args.force:
            print(f"Index already exists at {INDEX_FILE}. Use --force to rebuild.")
        else:
            build_index(limit=args.limit)

    if args.search:
        if not INDEX_FILE.exists():
            print("No index found. Run with --build first.")
        else:
            hits = search(args.search, k=args.k)
            print(f"\nTop {args.k} results for: {args.search!r}\n")
            for hit in hits:
                print(f"  {hit['rank']:>2}. [{hit['score']:.4f}] {hit['image_id']}  {hit['title'][:60]}")
                print(f"       {hit['url']}")
