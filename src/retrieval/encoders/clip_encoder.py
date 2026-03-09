"""
CLIP-based encoders for text and images.

Both branches share the same embedding space (CLIP's joint vision-language
space), so text and image embeddings are directly comparable without any
additional alignment step.

Default model: openai/clip-vit-base-patch32
  - Embedding dim: 512
  - Text token limit: 77

For longer math posts, truncation is applied. The first 77 tokens typically
capture the question title and leading formulas, which is usually sufficient.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder:
    """
    Shared wrapper around a HuggingFace CLIP model.

    Provides separate encode_text() and encode_image() methods, both
    returning L2-normalised embeddings in the same CLIP space.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # Embedding dimensionality (512 for ViT-B/32, 768 for ViT-L/14)
        self.dim: int = self.model.config.projection_dim

    @torch.no_grad()
    def encode_text(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of strings into CLIP text embeddings.

        Args:
            texts      — list of raw strings (LaTeX-stripped, plain text)
            batch_size — number of strings to process at once
            normalize  — if True, L2-normalize the output vectors

        Returns:
            Tensor of shape (len(texts), dim)
        """
        all_embeddings: list[torch.Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)

            features = self.model.get_text_features(**inputs)
            if normalize:
                features = F.normalize(features, dim=-1)
            all_embeddings.append(features.cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_images(
        self,
        images: list[Image.Image | str | Path],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of images into CLIP vision embeddings.

        Args:
            images     — PIL Images, or paths to image files
            batch_size — number of images to process at once
            normalize  — if True, L2-normalize the output vectors

        Returns:
            Tensor of shape (len(images), dim)
        """
        # Load from path if necessary
        loaded: list[Image.Image] = []
        for img in images:
            if isinstance(img, (str, Path)):
                loaded.append(Image.open(img).convert("RGB"))
            else:
                loaded.append(img.convert("RGB"))

        all_embeddings: list[torch.Tensor] = []

        for i in range(0, len(loaded), batch_size):
            batch = loaded[i : i + batch_size]
            inputs = self.processor(
                images=batch,
                return_tensors="pt",
            ).to(self.device)

            features = self.model.get_image_features(**inputs)
            if normalize:
                features = F.normalize(features, dim=-1)
            all_embeddings.append(features.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode_text_single(self, text: str, normalize: bool = True) -> torch.Tensor:
        """Convenience wrapper for a single string. Returns shape (dim,)."""
        return self.encode_text([text], normalize=normalize).squeeze(0)

    def encode_image_single(
        self,
        image: Image.Image | str | Path,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Convenience wrapper for a single image. Returns shape (dim,)."""
        return self.encode_images([image], normalize=normalize).squeeze(0)
