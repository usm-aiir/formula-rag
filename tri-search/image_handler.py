import json
from pathlib import Path
from typing import Union

import open_clip
import faiss
import numpy as np
import torch
from PIL import Image

from dataset_handler import iter_dataset

INDEX_DIR  = Path(__file__).parent / "data" / "clip_index"
INDEX_FILE = INDEX_DIR / "mathimages.index"  
META_FILE  = INDEX_DIR / "metadata.json"     
EMBEDDING_DIM = 512


#  load CLIP once, reuse everywhere
_model      = None
_preprocess = None

def _get_clip():
    """Load CLIP the first time it is needed, then cache it."""
    global _model, _preprocess
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        _model.eval()
    return _model, _preprocess

def encode_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Turn a single image file into a 512-number vector.
    Returns a numpy array of shape (1, 512), dtype float32.
    """
    model, preprocess = _get_clip()
    device = next(model.parameters()).device

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        vector = model.encode_image(image)

    # L2-normalise so that dot-product == cosine similarity
    vector = vector / vector.norm(dim=-1, keepdim=True)

    return vector.cpu().numpy().astype(np.float32)


def encode_text(query: str) -> np.ndarray:
    """
    Turn a text string into a 512-number vector in the same space as
    the image vectors, so you can search images with words.
    Returns a numpy array of shape (1, 512), dtype float32.
    """
    model, _ = _get_clip()
    device = next(model.parameters()).device

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer([query]).to(device)

    with torch.no_grad():
        vector = model.encode_text(tokens)

    vector = vector / vector.norm(dim=-1, keepdim=True)

    return vector.cpu().numpy().astype(np.float32)

def build_index(limit: Union[int, None] = None) -> None:
    """
    Loop over every image in the dataset, encode it with CLIP, add the
    vector to a FAISS index, and save both the index and a metadata list.

    Args:
        limit: only process the first N images (useful for a quick test).
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # IndexFlatIP = exact nearest-neighbour search using inner product
    # (inner product on L2-normalised vectors == cosine similarity)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    # One metadata record per row — keeps the same order as the faiss index
    metadata = []

    for i, entry in enumerate(iter_dataset()):
        if limit is not None and i >= limit:
            break

        try:
            vector = encode_image(entry.image_path)     # shape (1, 512)
        except Exception as e:
            print(f"[skip] {entry.image_id}: {e}")
            continue

        index.add(vector)                               # add the row to FAISS
        metadata.append({
            "image_id": entry.image_id,
            "source":   entry.source,
            "title":    entry.title,
            "url":      entry.url,
            "file_path": str(entry.image_path),
        })

        if (i + 1) % 500 == 0:
            print(f"  encoded {i + 1} images so far ")

    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(metadata, indent=2))

    print(f"\nDone. Indexed {index.ntotal} images.")
    print(f"  index  → {INDEX_FILE}")
    print(f"  metadata → {META_FILE}")


def search(query: Union[str, Path], k: int = 5) -> list:
    """
    Find the k most similar images to a query.

    Args:
        query: a text string ("a triangle diagram") OR a path to an image file.
        k:     how many results to return.

    Returns:
        A list of dicts, each with keys: rank, score, image_id, source, title, url.
    """
    # Encode the query into a vector
    if isinstance(query, (str, Path)) and Path(query).exists():
        query_vector = encode_image(query)
    else:
        query_vector = encode_text(str(query))

    # Load the saved index and metadata
    index    = faiss.read_index(str(INDEX_FILE))
    metadata = json.loads(META_FILE.read_text())

    # Ask FAISS for the k nearest neighbours
    scores, indices = index.search(query_vector, k)
    # scores[0]  similarity scores (higher = more similar, max 1.0)
    # indices[0]  row numbers in the index

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        result = {"rank": rank, "score": float(score)}
        result.update(metadata[idx])
        results.append(result)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLIP image index builder and searcher")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the full FAISS index if it does not already exist.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild the index even if it already exists.",
    )
    parser.add_argument(
        "--search",
        metavar="QUERY",
        help="Search the index with a text string or an image file path.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        metavar="K",
        help="Number of results to return when searching (default: 5).",
    )
    args = parser.parse_args()

    if args.build or args.force:
        if INDEX_FILE.exists() and not args.force:
            print(f"Index already exists at {INDEX_FILE}. Use --force to rebuild.")
        else:
            if args.force and INDEX_FILE.exists():
                print("Forcing rebuild of existing index …\n")
            else:
                print("Building full index …\n")
            build_index()

    if args.search:
        if not INDEX_FILE.exists():
            print("No index found. Run with --build first.")
        else:
            hits = search(args.search, k=args.k)
            for hit in hits:
                print(f"  [{hit['rank']}] score={hit['score']:.3f}  {hit['image_id']}  {hit['title'][:60]}  path={hit.get('file_path', 'N/A')}")

    if not args.build and not args.search:
        parser.print_help()
