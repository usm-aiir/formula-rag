"""
Stage 2: Align the formula encoder to CLIP space using ARQMath Task 1 qrels.

Task 1: given a question post, retrieve relevant answer posts. The qrels tell
us which (topic, answer) pairs are relevant.

Training objective:
  The fused embedding of a question should be close to the fused embedding of
  its relevant answer posts in CLIP space.

  Fused embedding = mean(CLIP_text, CLIP_images, FormulaGNN_formulas)
                    [L2-normalised]

CLIP is frozen throughout. Only the FormulaGNN (+ projection head) is
updated, bringing formula embeddings into alignment with the CLIP text/image
space that is already aligned.

The formula encoder is initialised from the Stage 1 checkpoint. For the first
few epochs the formula branch can be optionally frozen to stabilise training,
then unfrozen for full fine-tuning.

Usage:
    python -m src.retrieval.train.train_retrieval \\
        --stage1-ckpt checkpoints/formula_encoder_stage1.pt \\
        --epochs 5 --batch-size 32

Outputs:
    checkpoints/formula_encoder_stage2.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.encoders import CLIPEncoder, FormulaEncoder, FormulaGNN
from src.retrieval.index.build_index import build_formula_cache
from src.data_pipeline.parsing import TopicReader

# ── Paths ─────────────────────────────────────────────────────────────────────
POSTS_JSONL = REPO_ROOT / "data/processed/posts.jsonl"
IMAGES_DIR  = REPO_ROOT / "data/raw/arqmath/images"
OPT_DIR     = REPO_ROOT / "data/raw/arqmath/formulas/opt_representation_v3"
TOPICS_DIR  = REPO_ROOT / "data/raw/arqmath/topics/task1"
QRELS_DIR   = REPO_ROOT / "data/raw/arqmath/qrels/task1"
CKPT_DIR    = REPO_ROOT / "checkpoints"

TRAIN_TOPICS = TOPICS_DIR / "arqmath1/Topics_V2.0.xml"
TRAIN_QRELS  = QRELS_DIR  / "arqmath1/qrel_task1_2020_official"
VAL_TOPICS   = TOPICS_DIR / "arqmath2/Topics_Task1_2021_V1.1.xml"
VAL_QRELS    = QRELS_DIR  / "arqmath2/qrel_task1_2021_official.tsv"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_posts(posts_jsonl: Path) -> dict[int, dict]:
    """Load all posts into a {post_id: post_dict} map."""
    posts: dict[int, dict] = {}
    with posts_jsonl.open(encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading posts"):
            post = json.loads(line)
            posts[post["post_id"]] = post
    return posts


def load_topics(xml_path: Path) -> dict[str, dict]:
    """
    Parse ARQMath Task 1 topic XML using TopicReader.
    Returns {topic_id: {post_id, text}} dicts compatible with embed_topic().
    """
    if not xml_path.exists():
        print(f"Warning: topics file not found: {xml_path}")
        return {}

    reader = TopicReader(str(xml_path))
    topics: dict[str, dict] = {}
    for topic_id, topic in reader.map_topics.items():
        title    = (topic.title    or "").strip()
        question = (topic.question or "").strip()
        topics[topic_id] = {
            "topic_id": topic_id,
            "post_id":  topic.post_id,
            "text":     f"{title} {question}".strip(),
        }
    return topics


def load_task1_qrels(qrel_path: Path) -> dict[str, list[int]]:
    """
    Load Task 1 qrels → {topic_id: [relevant_answer_post_ids]}.
    Includes only relevance > 0.
    """
    qrels: dict[str, list[int]] = defaultdict(list)
    if not qrel_path.exists():
        print(f"Warning: qrel file not found: {qrel_path}")
        return dict(qrels)

    with qrel_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            topic_id  = parts[0]
            answer_id = int(parts[2])
            relevance = int(parts[3])
            if relevance > 0:
                qrels[topic_id].append(answer_id)

    return dict(qrels)


# ── Dataset ───────────────────────────────────────────────────────────────────

class RetrievalPairDataset(Dataset):
    """
    Each item is (topic_dict, relevant_answer_dict).
    Negatives are other items in the same batch (in-batch negatives).
    """

    def __init__(
        self,
        topics: dict[str, dict],
        qrels: dict[str, list[int]],
        posts: dict[int, dict],
        formula_cache: dict[int, list[str]],
    ) -> None:
        self.posts         = posts
        self.formula_cache = formula_cache
        self.pairs: list[tuple[dict, dict]] = []

        for topic_id, topic in topics.items():
            relevant_ids = qrels.get(topic_id, [])
            for ans_id in relevant_ids:
                if ans_id in posts:
                    self.pairs.append((topic, posts[ans_id]))

        print(f"Dataset: {len(self.pairs):,} (topic, answer) pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        return self.pairs[idx]


# ── Embedding helpers ─────────────────────────────────────────────────────────

def embed_post(
    post: dict,
    clip: CLIPEncoder,
    formula_enc: FormulaEncoder,
    formula_cache: dict[int, list[str]],
    images_dir: Path,
) -> torch.Tensor | None:
    """
    Compute the fused embedding for a single post.
    Returns a 1D normalised tensor, or None if encoding fails entirely.
    """
    parts: list[torch.Tensor] = []

    # Text
    text = post.get("text", "").strip()
    if text:
        emb = clip.encode_text_single(text, normalize=True)
        parts.append(emb)

    # Images
    img_dir = images_dir / str(post["post_id"])
    img_files = sorted(img_dir.glob("*")) if img_dir.exists() else []
    if img_files:
        img_embs = clip.encode_images(img_files, normalize=True)
        parts.append(img_embs.mean(dim=0))

    # Formulas
    formulas = formula_cache.get(post["post_id"], [])
    if formulas:
        form_embs, _ = formula_enc.encode(formulas, normalize=True)
        if form_embs.shape[0] > 0:
            parts.append(form_embs.mean(dim=0))

    if not parts:
        return None

    mean_emb = torch.stack(parts).mean(dim=0)
    return F.normalize(mean_emb, dim=0)


def embed_topic(
    topic: dict,
    posts: dict[int, dict],
    clip: CLIPEncoder,
    formula_enc: FormulaEncoder,
    formula_cache: dict[int, list[str]],
    images_dir: Path,
) -> torch.Tensor | None:
    """Embed a topic using its text. If a matching post exists, also use its images/formulas."""
    post_id = topic.get("post_id")
    post = posts.get(post_id) if post_id else None

    if post:
        return embed_post(post, clip, formula_enc, formula_cache, images_dir)

    # Fall back to text-only if post not in corpus
    text = topic.get("text", "").strip()
    if not text:
        return None
    return clip.encode_text_single(text, normalize=True)


# ── Loss ──────────────────────────────────────────────────────────────────────

def infonce_loss(q: torch.Tensor, d: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = (q @ d.T) / temperature
    labels = torch.arange(len(q), device=q.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    stage1_ckpt: Path | None = None,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-5,
    temperature: float = 0.07,
    freeze_formula_epochs: int = 1,
    save_every: int = 1,
    clip_model: str = "openai/clip-vit-base-patch32",
) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load encoders
    clip = CLIPEncoder(model_name=clip_model, device=device)
    formula_enc = FormulaEncoder(device=device, clip_dim=clip.dim)
    if stage1_ckpt and stage1_ckpt.exists():
        print(f"Loading Stage 1 checkpoint: {stage1_ckpt}")
        formula_enc.load(stage1_ckpt)
    else:
        print("Warning: no Stage 1 checkpoint found — starting from random weights.")

    # CLIP is frozen; only formula encoder is trained
    for p in clip.model.parameters():
        p.requires_grad_(False)

    # Load data
    print("Loading posts...")
    posts = load_posts(POSTS_JSONL)

    print("Loading formula cache (OPT index)...")
    formula_cache = build_formula_cache(OPT_DIR)

    topics = load_topics(TRAIN_TOPICS)
    qrels  = load_task1_qrels(TRAIN_QRELS)

    dataset = RetrievalPairDataset(topics, qrels, posts, formula_cache)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: list(zip(*b)),  # returns (topics_list, answers_list)
        num_workers=0,   # keep 0: embed_post is not picklable
    )

    optimizer = torch.optim.AdamW(
        formula_enc.model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        # Optionally freeze formula encoder for the first N epochs
        if epoch <= freeze_formula_epochs:
            for p in formula_enc.model.parameters():
                p.requires_grad_(False)
            print(f"Epoch {epoch}: formula encoder frozen")
        else:
            for p in formula_enc.model.parameters():
                p.requires_grad_(True)
            formula_enc.to_train_mode()

        total_loss = 0.0
        n_batches  = 0

        for topic_batch, answer_batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            q_embs_list: list[torch.Tensor] = []
            d_embs_list: list[torch.Tensor] = []

            for topic, answer in zip(topic_batch, answer_batch):
                q_emb = embed_topic(
                    topic, posts, clip, formula_enc, formula_cache, IMAGES_DIR
                )
                d_emb = embed_post(
                    answer, clip, formula_enc, formula_cache, IMAGES_DIR
                )
                if q_emb is not None and d_emb is not None:
                    q_embs_list.append(q_emb)
                    d_embs_list.append(d_emb)

            if len(q_embs_list) < 2:
                continue

            q_embs = torch.stack(q_embs_list).to(device)
            d_embs = torch.stack(d_embs_list).to(device)

            loss = infonce_loss(q_embs, d_embs, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}: avg loss = {avg_loss:.4f}")

        if epoch % save_every == 0:
            ckpt = CKPT_DIR / f"formula_encoder_stage2_epoch{epoch}.pt"
            formula_enc.save(ckpt)
            print(f"  Saved: {ckpt}")

    final_ckpt = CKPT_DIR / "formula_encoder_stage2.pt"
    formula_enc.save(final_ckpt)
    print(f"\nStage 2 complete. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: align formula encoder to CLIP space")
    parser.add_argument("--stage1-ckpt",          type=Path,  default=CKPT_DIR / "formula_encoder_stage1.pt")
    parser.add_argument("--epochs",               type=int,   default=5)
    parser.add_argument("--batch-size",           type=int,   default=32)
    parser.add_argument("--lr",                   type=float, default=5e-5)
    parser.add_argument("--temperature",          type=float, default=0.07)
    parser.add_argument("--freeze-formula-epochs",type=int,   default=1)
    parser.add_argument("--save-every",           type=int,   default=1)
    parser.add_argument("--clip-model",           type=str,   default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    train(
        stage1_ckpt=args.stage1_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        freeze_formula_epochs=args.freeze_formula_epochs,
        save_every=args.save_every,
        clip_model=args.clip_model,
    )
