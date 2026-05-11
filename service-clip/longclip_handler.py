# LongCLIP image index and search handler
# python longclip_handler.py --build
# python longclip_handler.py --search "eigenvalue decomposition diagram"
# python longclip_handler.py --search /path/to/image.png --k 10

import copy
import io
import json
import sys
from pathlib import Path
from typing import Union

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import cairosvg

# Ensure the model/ package from Long-CLIP is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import longclip

from dataset_handler import iter_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INDEX_DIR            = Path(__file__).parent / "data" / "longclip_index"
INDEX_FILE           = INDEX_DIR / "mathimages.index"
META_FILE            = INDEX_DIR / "metadata.json"
BASE_CHECKPOINT      = Path(__file__).parent / "checkpoints" / "longclip" / "longclip-L.pt"
FINETUNED_CHECKPOINT = Path(__file__).parent / "checkpoints" / "longclip" / "longclip-L-finetuned.pt"

_model      = None
_preprocess = None


def _load_model(force_base: bool = False) -> None:
    global _model, _preprocess
    if _model is not None:
        return
    if not force_base and FINETUNED_CHECKPOINT.exists():
        ckpt = FINETUNED_CHECKPOINT
        print(f"[longclip] loading fine-tuned checkpoint: {ckpt.name}")
    elif BASE_CHECKPOINT.exists():
        ckpt = BASE_CHECKPOINT
        print(f"[longclip] loading base checkpoint: {ckpt.name}")
    else:
        raise FileNotFoundError(
            f"No checkpoint found. Need either:\n"
            f"  {FINETUNED_CHECKPOINT}  (run --finetune to create)\n"
            f"  {BASE_CHECKPOINT}\n"
            "Download base: hf download BeichenZhang/LongCLIP-L longclip-L.pt "
            "--local-dir checkpoints/longclip/"
        )
    _model, _preprocess = longclip.load(str(ckpt), device=DEVICE)
    _model.eval()
    print("[longclip] model ready")


def load_base_model() -> None:
    """Force-load the base (non-fine-tuned) checkpoint, resetting any cached model."""
    global _model, _preprocess
    _model = None
    _preprocess = None
    _load_model(force_base=True)


def _open_image(path: Union[str, Path]) -> Image.Image:
    path = str(path)
    if path.lower().endswith(".svg"):
        png_bytes = cairosvg.svg2png(url=path, output_width=336, output_height=336)
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return Image.open(path).convert("RGB")


def encode_image(image_path: Union[str, Path]) -> np.ndarray:
    """Encode a single image to an L2-normalised vector. Returns shape (1, D)."""
    _load_model()  # no-op if already loaded
    pil = _open_image(image_path)
    tensor = _preprocess(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast(device_type=DEVICE):
        vec = _model.encode_image(tensor)
    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().float().numpy()


def encode_text(query: str) -> np.ndarray:
    """Encode a text query to an L2-normalised vector. Returns shape (1, D)."""
    _load_model()  # no-op if already loaded
    tokens = longclip.tokenize([query], truncate=True).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast(device_type=DEVICE):
        vec = _model.encode_text(tokens)
    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().float().numpy()


# ─── Fine-tuning ──────────────────────────────────────────────────────────────

class _MathImagesDataset(Dataset):
    """Image-title pairs from iter_dataset(), used for contrastive fine-tuning."""

    def __init__(self, processor):
        self.entries   = list(iter_dataset())
        self.processor = processor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        try:
            img = _open_image(entry.image_path)
            return self.processor(img), entry.title
        except Exception:
            return None, None


def _collate_skip_bad(batch):
    """Collate function that silently drops entries where image loading failed."""
    batch = [(img, txt) for img, txt in batch if img is not None]
    if not batch:
        return None, None
    imgs, txts = zip(*batch)
    return torch.stack(imgs), list(txts)


def _contrastive_loss(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if torch.isnan(img_feats).any() or torch.isinf(img_feats).any():
        return torch.tensor(0.0, device=img_feats.device, requires_grad=True)
    if torch.isnan(txt_feats).any() or torch.isinf(txt_feats).any():
        return torch.tensor(0.0, device=txt_feats.device, requires_grad=True)

    img_feats   = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-8)
    txt_feats   = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-8)
    temperature = max(temperature, 0.01)

    logits = torch.matmul(img_feats, txt_feats.T) / temperature
    logits = torch.clamp(logits, min=-100, max=100)

    labels    = torch.arange(img_feats.shape[0]).to(img_feats.device)
    loss_i2t  = nn.CrossEntropyLoss()(logits,   labels)
    loss_t2i  = nn.CrossEntropyLoss()(logits.T, labels)
    total     = (loss_i2t + loss_t2i) / 2

    if torch.isnan(total):
        return torch.tensor(0.0, device=img_feats.device, requires_grad=True)
    return total


def _validate(model: nn.Module, val_dataset: Dataset, batch_size: int = 8) -> dict:
    model.eval()
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                         collate_fn=_collate_skip_bad)

    total_loss   = 0.0
    correct      = 0
    total        = 0
    valid_batches = 0

    with torch.no_grad():
        for images, texts in loader:
            if images is None:
                continue
            images = images.to(DEVICE)
            try:
                tokens    = longclip.tokenize(list(texts), truncate=True).to(DEVICE)
                img_feats = model.encode_image(images)
                txt_feats = model.encode_text(tokens)
            except Exception as e:
                print(f"[val] skipping batch: {e}")
                continue

            if torch.isnan(img_feats).any() or torch.isnan(txt_feats).any():
                continue

            loss = _contrastive_loss(img_feats, txt_feats)
            if torch.isnan(loss) or loss.item() > 100:
                continue

            sims  = img_feats @ txt_feats.T
            preds = sims.argmax(dim=1)
            correct      += (preds == torch.arange(images.shape[0]).to(DEVICE)).sum().item()
            total        += images.shape[0]
            total_loss   += loss.item()
            valid_batches += 1

    return {
        "val_loss":     total_loss / valid_batches if valid_batches > 0 else float("inf"),
        "val_accuracy": correct / total if total > 0 else 0.0,
        "valid_batches": valid_batches,
    }


def fine_tune(
    epochs: int       = 10,
    batch_size: int   = 8,
    learning_rate: float = 1e-6,
    patience: int     = 3,
) -> None:
    """
    Fine-tune visual.proj + text_projection on MathImages (post title as caption text).
    Saves the complete model state dict to FINETUNED_CHECKPOINT.
    After this runs, longclip-L.pt (base) is no longer required.
    """
    _load_model()

    full_ds = _MathImagesDataset(_preprocess)
    if len(full_ds) == 0:
        print("[longclip] no training samples found — check MathImages paths")
        return

    train_size = int(0.9 * len(full_ds))
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"[longclip] fine-tune: {train_size} train / {val_size} val")

    # Freeze all layers; unfreeze only the two projection matrices
    trainable = []
    for name, param in _model.named_parameters():
        if "visual.proj" in name or "text_projection" in name:
            param.requires_grad = True
            trainable.append(param)
        else:
            param.requires_grad = False
    print(f"[longclip] trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader    = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                          collate_fn=_collate_skip_bad)

    best_state   = None
    best_acc     = -1.0
    patience_ctr = 0

    for epoch in range(epochs):
        _model.train()
        _model.float()
        total_loss   = 0.0
        good_batches = 0

        for i, (images, texts) in enumerate(loader):
            if images is None:
                continue
            images = images.to(DEVICE)
            try:
                tokens = longclip.tokenize(list(texts), truncate=True).to(DEVICE)
            except Exception as e:
                print(f"[longclip] tokenise error batch {i}: {e}, skipping")
                continue
            try:
                img_feats = _model.encode_image(images)
                txt_feats = _model.encode_text(tokens)
            except Exception as e:
                print(f"[longclip] forward error batch {i}: {e}, skipping")
                continue

            loss = _contrastive_loss(img_feats, txt_feats)
            if torch.isnan(loss) or loss.item() > 100:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            total_loss   += loss.item()
            good_batches += 1

        avg_loss    = total_loss / good_batches if good_batches > 0 else float("inf")
        val_metrics = _validate(_model, val_ds, batch_size=batch_size)
        print(
            f"Epoch {epoch+1}/{epochs}  "
            f"train_loss={avg_loss:.4f}  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"val_acc={val_metrics['val_accuracy']:.4f}"
        )

        if val_metrics["valid_batches"] > 0:
            if val_metrics["val_accuracy"] > best_acc:
                best_acc     = val_metrics["val_accuracy"]
                best_state   = copy.deepcopy(_model.state_dict())
                patience_ctr = 0
                print(f"  *** best model updated (val_acc={best_acc:.4f}) ***")
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print("[longclip] early stopping triggered")
                    break
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("[longclip] early stopping triggered (no valid val batches)")
                break

        scheduler.step()

    if best_state is not None:
        _model.load_state_dict(best_state)

    FINETUNED_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_model.state_dict(), str(FINETUNED_CHECKPOINT))
    print(f"[longclip] fine-tuned checkpoint saved → {FINETUNED_CHECKPOINT}")
    print("[longclip] longclip-L.pt (base) is no longer required for inference")

    # Restore eval mode and freeze all params for normal inference use
    _model.eval()
    for param in _model.parameters():
        param.requires_grad = False


# ─── Index build & search ─────────────────────────────────────────────────────

def build_index(limit: Union[int, None] = None) -> None:
    """
    Encode every image in the MathImages dataset and save a FAISS index + metadata.

    Args:
        limit: only process the first N images (useful for a quick smoke test).
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _load_model()

    index    = None
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
            print(f"[longclip] embedding dim = {dim}")
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
    print(f"\nDone. Indexed {index.ntotal} images → {INDEX_FILE}")


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

    parser = argparse.ArgumentParser(description="LongCLIP image index builder and searcher")
    parser.add_argument("--build",      action="store_true", help="Build the FAISS index.")
    parser.add_argument("--force",      action="store_true", help="Rebuild even if index already exists.")
    parser.add_argument("--limit",      type=int, default=None, metavar="N",
                        help="Only index the first N images (smoke test).")
    parser.add_argument("--search",     metavar="QUERY",
                        help="Text query or image path to search with.")
    parser.add_argument("--k",          type=int, default=5, metavar="K",
                        help="Number of results to return (default: 5).")
    parser.add_argument("--finetune",   action="store_true",
                        help="Fine-tune on MathImages and save a self-contained checkpoint.")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-6)
    parser.add_argument("--patience",   type=int,   default=3)
    args = parser.parse_args()

    if args.finetune:
        fine_tune(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
        )
    elif args.build:
        if INDEX_FILE.exists() and not args.force:
            print(f"Index already exists at {INDEX_FILE}. Use --force to rebuild.")
        else:
            build_index(limit=args.limit)
    elif args.search:
        results = search(args.search, k=args.k)
        for r in results:
            print(f"  [{r['rank']}] {r['image_id']}  score={r['score']:.4f}  {r['title']}")
    else:
        parser.print_help()
