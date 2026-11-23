"""Versioned, self-contained AR checkpoint metadata and strict weight loading."""
import json
from pathlib import Path

import torch

from . import ARCHITECTURE
from .body import digest_file
from .codecs import CodecConfig
from .model import build_model
from .sequence import SequenceBuilder


def read_metadata(path):
    path = Path(path)
    if not (path / "COMPLETE").is_file():
        raise ValueError("Incomplete or non-token_ar checkpoint directory")
    saved = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if saved.get("architecture") != ARCHITECTURE:
        raise ValueError("Diffusion checkpoints are not interchangeable with token_ar")
    return saved


def load_run(path, device="cpu", dtype="float32"):
    from transformers import AutoTokenizer
    path = Path(path)
    saved = read_metadata(path)
    if digest_file(path / "codecs.pt") != saved["codec_sha256"]:
        raise ValueError("Checkpoint VQ codebook changed")
    model, _ = build_model(saved["config"], CodecConfig(**saved["codec_config"]), saved["backbone_config"])
    if (path / "model.safetensors").is_file():
        from safetensors.torch import load_model
        load_model(model, str(path / "model.safetensors"), strict=True)
    elif (path / "pytorch_model_fsdp.bin").is_file():
        model.load_state_dict(torch.load(path / "pytorch_model_fsdp.bin", map_location="cpu", weights_only=True))
    else:
        raise ValueError("Expected full Accelerate/FSDP state; sharded state requires consolidation")
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True, trust_remote_code=False)
    model.to(device=device, dtype=getattr(torch, dtype)).eval()
    builder = SequenceBuilder(tokenizer, model.vocab, saved["config"].get("max_text_tokens", 128),
                              saved["config"].get("max_sequence_tokens", 1024))
    return model, builder, saved
