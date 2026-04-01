"""
MathBERTa text encoder for ARQMath Task 1 (Answer Retrieval).

Uses witiko/mathberta — RoBERTa-base fine-tuned on ArXMLiv + Math StackExchange
with extended LaTeX tokenizer. We apply mean pooling over the last hidden state
and optional L2 normalisation for dense retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "witiko/mathberta"


class TextEncoder(nn.Module):
    """MathBERTa (or any HF model) with mean pooling for sentence embedding."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 256,
        normalize: bool = True,
        gradient_checkpointing: bool = False,
        _local_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        tok_src = str(_local_path) if _local_path and (Path(_local_path) / "tokenizer.json").exists() else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        self.encoder = AutoModel.from_pretrained(model_name)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self._hidden_size = self.encoder.config.hidden_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        do_norm = normalize if normalize is not None else self.normalize
        if do_norm:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def encode(
        self,
        texts: List[str],
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 32,
    ) -> torch.Tensor:
        self.eval()
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                embs = self.forward(**enc)
                all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pretrained(self, path: Union[str, Path]):
        """Save model weights (safetensors) + config.json."""
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "normalize": self.normalize,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        save_file(self.state_dict(), str(path / "model.safetensors"))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
    ) -> "TextEncoder":
        """Load from any supported format (directory or legacy .pt)."""
        path = Path(path)
        if path.is_dir():
            return cls._load_dir(path, map_location=map_location)
        return cls._load_pt(path, map_location=map_location)

    @classmethod
    def _load_dir(
        cls,
        path: Path,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "TextEncoder":
        """Load from a directory (config.json + model.safetensors or pytorch_model.bin)."""
        config: dict = {}
        config_file = path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)

        model = cls(
            model_name=config.get("model_name", DEFAULT_MODEL),
            max_length=config.get("max_length", 256),
            normalize=config.get("normalize", True),
            gradient_checkpointing=False,
            _local_path=path,
        )

        st = path / "model.safetensors"
        pt = path / "pytorch_model.bin"
        if st.exists():
            from safetensors.torch import load_file
            state = load_file(str(st), device=str(map_location))
        elif pt.exists():
            state = torch.load(pt, map_location=map_location)
        else:
            raise FileNotFoundError(f"No model weights in {path}")
        model.load_state_dict(state)
        return model

    @classmethod
    def _load_pt(
        cls,
        path: Path,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "TextEncoder":
        """Load from legacy single .pt file."""
        payload = torch.load(str(path), map_location=map_location)
        model = cls(
            model_name=payload.get("model_name", DEFAULT_MODEL),
            max_length=payload.get("max_length", 256),
            normalize=payload.get("normalize", True),
            gradient_checkpointing=False,
        )
        model.load_state_dict(payload["state_dict"])
        return model
