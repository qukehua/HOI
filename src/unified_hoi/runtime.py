"""Shared checkpoint, device and archive helpers for the command-line programs."""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch

from .diffusion import HOIDiffusion
from .model import ModelConfig, UnifiedHOIDenoiser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(value="auto"):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path, device="cpu", ema=True):
    # Checkpoints contain optimizer/RNG Python state. Load only files you trust.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = UnifiedHOIDenoiser(ModelConfig(**checkpoint["config"]["model"]))
    model.load_state_dict(checkpoint["ema"] if ema and checkpoint.get("ema") else checkpoint["model"])
    return model.to(device).eval(), HOIDiffusion(checkpoint["config"].get("diffusion_steps", 1000)), checkpoint


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
