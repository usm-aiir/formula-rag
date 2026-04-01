"""
python fine_tune_clip.py --device cuda --resume /home/lucas.matheson/formula-rag/tri-search/checkpoints/clip_finetune/last.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import open_clip
import torch
from PIL import Image
from torch.amp import GradScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATA_ROOT = Path("/home/lucas.matheson/MathImages")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "clip_finetune"
SOURCE_SPECS = {
  "MSE": ("MSEImages", "MSE.tsv"),
  "MathOverflow": ("MathOverFlowImages", "math_overflow.tsv"),
  "Mathematica": ("MathematicaImages", "mathematica.tsv"),
}


@dataclass
class MathImageEntry:
  source: str
  image_id: str
  title: str
  url: str
  image_path: Path


class MathImagesClipDataset(Dataset):
  def __init__(self, entries: Sequence[MathImageEntry], preprocess: Callable):
    self.entries = list(entries)
    self.preprocess = preprocess

  def __len__(self) -> int:
    return len(self.entries)

  def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, str]]:
    entry = self.entries[index]
    try:
      with Image.open(entry.image_path) as image:
        image_tensor = self.preprocess(image.convert("RGB"))
    except Exception as exc:
      print(f"[skip] unreadable image {entry.image_path}: {exc}")
      return None
    text = entry.title.strip() or entry.image_id
    return image_tensor, text


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Fine-tune CLIP on the MathImages dataset.")
  parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
  parser.add_argument(
    "--sources",
    nargs="+",
    default=list(SOURCE_SPECS.keys()),
    choices=sorted(SOURCE_SPECS.keys()),
    help="Subset of MathImages sources to train on.",
  )
  parser.add_argument("--model-name", default="ViT-B-32")
  parser.add_argument("--pretrained", default="openai")
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--lr", type=float, default=5e-5)
  parser.add_argument("--weight-decay", type=float, default=0.2)
  parser.add_argument("--beta1", type=float, default=0.9)
  parser.add_argument("--beta2", type=float, default=0.98)
  parser.add_argument("--eps", type=float, default=1e-6)
  parser.add_argument("--num-workers", type=int, default=4)
  parser.add_argument("--val-split", type=float, default=0.1)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--limit", type=int, default=None, help="Only use the first N examples.")
  parser.add_argument("--device", default=None, help="Override the training device, e.g. cuda:0 or cpu.")
  parser.add_argument("--log-every", type=int, default=50)
  parser.add_argument("--save-every", type=int, default=1)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--resume", type=Path, default=None)
  return parser.parse_args()


def resolve_device(device_arg: Optional[str]) -> torch.device:
  if device_arg:
    return torch.device(device_arg)
  if torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")

# build the dataset and dataloaders, and run the training loop with validation and checkpointing.
def retrieve_math_image_entries(
  data_root: Path,
  sources: Sequence[str],
  limit: Optional[int] = None,
) -> List[MathImageEntry]:
  entries: List[MathImageEntry] = []

  for source in sources:
    # for each folder of images defined in mathimges, get the corresponding tsv file, read the metadata, and create MathImageEntry objects for each valid image-title pair, up to the specified limit.
    folder_name, tsv_name = SOURCE_SPECS[source]
    source_dir = data_root / folder_name
    tsv_path = source_dir / tsv_name
    if not tsv_path.exists():
      raise FileNotFoundError(f"Missing TSV for source {source}: {tsv_path}")

    # open the source in read mode, get the needed data, append the entries
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
      reader = csv.reader(handle, delimiter="\t")
      for row in reader:
        if len(row) < 4:
          continue

        image_id, title, _, url = row[0], row[1], row[2], row[3]
        image_path = source_dir / f"{image_id}.png"
        if not image_path.exists():
          continue

        entries.append(
          MathImageEntry(
            source=source,
            image_id=image_id,
            title=title,
            url=url,
            image_path=image_path,
          )
        )

        if limit is not None and len(entries) >= limit:
          return entries

  return entries

# split the dataset into train and validation sets, ensuring that each set has at least 2 examples for contrastive learning, and build a collate function for the dataloader that handles variable batch sizes due to skipped examples. Then define the training loop with mixed precision support, checkpoint saving/loading, and logging of training/validation loss. Finally, run the main function to execute the training process.
def split_entries(
  entries: Sequence[MathImageEntry],
  val_split: float,
  seed: int,
) -> Tuple[List[MathImageEntry], List[MathImageEntry]]:
  if not 0.0 <= val_split < 1.0:
    raise ValueError("--val-split must be in the range [0, 1).")

  shuffled = list(entries)
  random.Random(seed).shuffle(shuffled)

  if len(shuffled) < 4 or val_split == 0.0:
    return shuffled, []

  val_size = int(len(shuffled) * val_split)
  if val_size < 2:
    val_size = 2
  if len(shuffled) - val_size < 2:
    val_size = len(shuffled) - 2

  if val_size < 2:
    return shuffled, []

  val_entries = shuffled[:val_size]
  train_entries = shuffled[val_size:]
  return train_entries, val_entries

# filter out batches with fewer than 2 samples, since CLIP contrastive loss requires at least 2 pairs, and log the number of skipped batches. Also, implement mixed precision training with automatic device selection and proper handling of the optimizer and scaler states in the checkpointing functions.
def build_collate_fn(tokenizer: Callable) -> Callable:
  def collate_fn(batch: Sequence[Optional[Tuple[torch.Tensor, str]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
      return torch.empty(0), torch.empty(0, dtype=torch.long)

    images, texts = zip(*valid_batch)
    image_batch = torch.stack(list(images), dim=0)
    token_batch = tokenizer(list(texts))
    return image_batch, token_batch

  return collate_fn


def clip_loss(
  image_features: torch.Tensor,
  text_features: torch.Tensor,
  logit_scale: torch.Tensor,
  loss_img: nn.Module,
  loss_txt: nn.Module,
) -> torch.Tensor:
  #  note the @ is for matrix multi, .t for transpose
  logits_per_image = logit_scale * image_features @ text_features.t()
  logits_per_text = logits_per_image.t()
  ground_truth = torch.arange(image_features.size(0), dtype=torch.long, device=image_features.device)
  return (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

def save_checkpoint(
  checkpoint_path: Path,
  epoch: int,
  model: nn.Module,
  optimizer: optim.Optimizer,
  scaler: GradScaler,
  args: argparse.Namespace,
  best_val_loss: Optional[float],
  metrics: Dict[str, float],
) -> None:
  checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "best_val_loss": best_val_loss,
    "metrics": metrics,
    "args": vars(args),
  }
  torch.save(payload, checkpoint_path)


def load_checkpoint(
  checkpoint_path: Path,
  model: nn.Module,
  optimizer: optim.Optimizer,
  scaler: GradScaler,
) -> Tuple[int, Optional[float]]:
  checkpoint = torch.load(checkpoint_path, map_location="cpu")
  model.load_state_dict(checkpoint["model_state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
  scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
  start_epoch = int(checkpoint.get("epoch", -1)) + 1
  best_val_loss = checkpoint.get("best_val_loss")
  return start_epoch, best_val_loss

# load the CLIP model and preprocess function, ensuring that the model is in evaluation mode and moved to the correct device, and that the preprocess function is compatible with the model's expected input size and normalization.
def get_autocast(device: torch.device):
  if device.type == "cuda":
    return torch.autocast(device_type="cuda", dtype=torch.float16)
  return nullcontext()

# train one epoch, iterating over the dataloader, computing the loss, and updating the model parameters if in training mode. 
# Log the average loss every log_every steps, and handle cases where all batches are skipped due to insufficient samples.
def run_epoch(
  model: nn.Module,
  dataloader: DataLoader,
  optimizer: Optional[optim.Optimizer],
  scaler: GradScaler,
  loss_img: nn.Module,
  loss_txt: nn.Module,
  device: torch.device,
  log_every: int,
  epoch: int,
  phase: str,
) -> float:
  is_training = optimizer is not None
  model.train(is_training)

  total_loss = 0.0
  num_steps = 0
  skipped_batches = 0
  start_time = time.time()

  for step, (images, texts) in enumerate(dataloader, start=1):
    if images.size(0) < 2:
      skipped_batches += 1
      continue

    images = images.to(device, non_blocking=True)
    texts = texts.to(device, non_blocking=True)

    if is_training:
      optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(is_training):
      with get_autocast(device):
        image_features, text_features, logit_scale = model(images, texts)
        loss = clip_loss(image_features, text_features, logit_scale, loss_img, loss_txt)

      if is_training:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    total_loss += float(loss.detach().cpu())
    num_steps += 1

    if step % log_every == 0:
      elapsed = time.time() - start_time
      avg_loss = total_loss / max(num_steps, 1)
      print(
        f"[{phase}] epoch={epoch + 1} step={step}/{len(dataloader)} "
        f"avg_loss={avg_loss:.4f} elapsed={elapsed:.1f}s"
      )

  if num_steps == 0:
    raise RuntimeError(
      f"No usable {phase} batches were produced. Increase the dataset size or reduce --batch-size."
    )

  average_loss = total_loss / num_steps
  if skipped_batches:
    print(f"[{phase}] skipped {skipped_batches} batch(es) smaller than 2 samples.")
  return average_loss


def summarize_entries(entries: Sequence[MathImageEntry]) -> Dict[str, int]:
  counts: Dict[str, int] = {}
  for entry in entries:
    counts[entry.source] = counts.get(entry.source, 0) + 1
  return counts


def main() -> None:
  args = parse_args()

  if args.batch_size < 2:
    raise ValueError("--batch-size must be at least 2 for CLIP contrastive training.")

  torch.manual_seed(args.seed)
  random.seed(args.seed)

# use a gpu, please
  device = resolve_device(args.device)
  if device.type == "cuda":
    # allegedly uses tensorfloat-32 which is used for faster matrix multiplications 
    # on recent nvidia hardware
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

  print(f"Loading MathImages from: {args.data_root}")
  entries = retrieve_math_image_entries(args.data_root, args.sources, args.limit)
  if not entries:
    raise RuntimeError("No usable MathImages examples were found.")

  train_entries, val_entries = split_entries(entries, args.val_split, args.seed)
  print(f"Loaded {len(entries)} image-title pairs.")
  print(f"Train: {len(train_entries)} | Val: {len(val_entries)}")
  print(f"Source counts: {json.dumps(summarize_entries(entries), sort_keys=True)}")

  model, _, preprocess = open_clip.create_model_and_transforms(
    args.model_name,
    pretrained=args.pretrained,
    device=device,
  )
  tokenizer = open_clip.get_tokenizer(args.model_name)
  collate_fn = build_collate_fn(tokenizer)

  train_dataset = MathImagesClipDataset(train_entries, preprocess)
  val_dataset = MathImagesClipDataset(val_entries, preprocess)

  loader_kwargs = {
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,
    "pin_memory": device.type == "cuda",
    "collate_fn": collate_fn,
  }
  train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
  val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_entries else None

  loss_img = nn.CrossEntropyLoss()
  loss_txt = nn.CrossEntropyLoss()
  optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    eps=args.eps,
    weight_decay=args.weight_decay,
  )
  scaler = GradScaler(device.type, enabled=device.type == "cuda")

  start_epoch = 0
  best_val_loss: Optional[float] = None
  if args.resume is not None:
    start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, scaler)
    print(f"Resumed from {args.resume} at epoch {start_epoch}.")

  args.output_dir.mkdir(parents=True, exist_ok=True)

  for epoch in range(start_epoch, args.epochs):
    train_loss = run_epoch(
      model=model,
      dataloader=train_loader,
      optimizer=optimizer,
      scaler=scaler,
      loss_img=loss_img,
      loss_txt=loss_txt,
      device=device,
      log_every=args.log_every,
      epoch=epoch,
      phase="train",
    )

    metrics = {"train_loss": train_loss}
    print(f"Epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f}")

    if val_loader is not None:
      with torch.no_grad():
        val_loss = run_epoch(
          model=model,
          dataloader=val_loader,
          optimizer=None,
          scaler=scaler,
          loss_img=loss_img,
          loss_txt=loss_txt,
          device=device,
          log_every=max(args.log_every, 1),
          epoch=epoch,
          phase="val",
        )
      metrics["val_loss"] = val_loss
      print(f"Epoch {epoch + 1}/{args.epochs} val_loss={val_loss:.4f}")

      if best_val_loss is None or val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
          args.output_dir / "best.pt",
          epoch,
          model,
          optimizer,
          scaler,
          args,
          best_val_loss,
          metrics,
        )
        print(f"Saved new best checkpoint to {args.output_dir / 'best.pt'}")

    if (epoch + 1) % args.save_every == 0:
      save_checkpoint(
        args.output_dir / f"epoch_{epoch + 1:03d}.pt",
        epoch,
        model,
        optimizer,
        scaler,
        args,
        best_val_loss,
        metrics,
      )

    save_checkpoint(
      args.output_dir / "last.pt",
      epoch,
      model,
      optimizer,
      scaler,
      args,
      best_val_loss,
      metrics,
    )

  print("Training complete.")
  print(f"Final checkpoint: {args.output_dir / 'last.pt'}")
  if best_val_loss is not None:
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
  main()