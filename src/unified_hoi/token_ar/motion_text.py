"""Dataset-specific frozen motion/text evaluator; not the HumanML3D evaluator.

Text inputs are existing frozen CLIP text features. Only the text projection and
motion encoders train here, exclusively on native training motions/captions.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import MODALITIES
from .body import digest_file, human_dim
from .data import MotionDataset

KIND = "native_motion_text_evaluator_v1"
VIEWS = (*MODALITIES, "joint")
TEXT_ENCODER = "openai/clip-vit-base-patch32"


def caption_key(value):
    return " ".join(value.casefold().split())


class EvaluatorDataset(MotionDataset):
    def __init__(self, *args, **kwargs):
        kwargs["include_geometry"] = False
        super().__init__(*args, **kwargs)
        # OMOMO has native sequences without captions. Keep them in VQ training,
        # but report and exclude them from this explicitly paired evaluation space.
        self.excluded_unlabelled = []
        retained = []
        for index, row in enumerate(self.records):
            variants = [v for v in row.get("text_variants", []) if v.get("text", "").strip()] or [row]
            if not any(v.get("text", "").strip() for v in variants):
                self.excluded_unlabelled.append(row["sequence_id"])
            else:
                retained.append(index)
        mapping = {old: new for new, old in enumerate(retained)}
        self.excluded_windows = sum(index not in mapping for index, _, _ in self.windows)
        self.windows = [(mapping[index], start, end) for index, start, end in self.windows if index in mapping]
        self.records = [self.records[i] for i in retained]
        self.loaded = [self.loaded[i] for i in retained]
        if not self.records:
            raise ValueError("No captioned native sequences for motion-text evaluation")
        self.features, self.feature_hashes = {}, {}
        for row in self.records:
            variants = [v for v in row.get("text_variants", []) if v.get("text", "").strip()] or [row]
            for variant in variants:
                text = variant.get("text", "").strip()
                path = variant.get("text_features_path")
                encoder = variant.get("text_encoder", row.get("text_encoder"))
                if not text or not path or encoder != TEXT_ENCODER:
                    raise ValueError(f"{row['sequence_id']}: evaluator requires caption-matched CLIP ViT-B/32 caches")
                array = np.load(path, allow_pickle=False)
                if array.shape != (512,) or not np.isfinite(array).all() or np.linalg.norm(array) == 0:
                    raise ValueError(f"Invalid frozen text features: {path}")
                key = caption_key(text)
                tensor = torch.from_numpy(array.astype(np.float32).copy())
                if key in self.features and not torch.allclose(self.features[key], tensor, atol=1e-5, rtol=1e-5):
                    raise ValueError("Identical normalized captions have inconsistent cached text embeddings")
                self.features[key] = tensor
                self.feature_hashes[str(Path(path).resolve())] = digest_file(path)

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        row = self.records[self.windows[index][0]]
        key = caption_key(batch["text"])
        batch["text_embedding"] = self.features[key].clone()
        batch["caption_key"] = key
        batch["source_id"] = row.get("source_sequence_id", row["sequence_id"])
        return batch


@dataclass
class EvaluatorConfig:
    dataset: str = "behave"
    width: int = 256
    embedding_dim: int = 256
    layers: int = 2
    temperature: float = .07

    def __post_init__(self):
        if self.dataset not in ("behave", "omomo") or min(self.width, self.embedding_dim, self.layers) < 1:
            raise ValueError("Invalid motion-text evaluator configuration")
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")


class MotionEncoder(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(dim + 1, config.width), nn.GELU())
        self.temporal = nn.GRU(config.width, config.width, config.layers, batch_first=True, bidirectional=True)
        self.output = nn.Linear(2 * config.width, config.embedding_dim)

    def forward(self, values, lengths, fps):
        rate = (fps / 10).log()[:, None, None].expand(-1, values.shape[1], 1)
        hidden = self.project(torch.cat((values, rate), -1))
        packed = nn.utils.rnn.pack_padded_sequence(hidden, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.temporal(packed)
        return F.normalize(self.output(torch.cat((hidden[-2], hidden[-1]), -1)), dim=-1)


class MotionTextEvaluator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dims = dict(human=human_dim(config.dataset), object=9, contact=690)
        self.motion = nn.ModuleDict({k: MotionEncoder(dim, config)
                                    for k, dim in (dims | {"joint": sum(dims.values())}).items()})
        self.text = nn.Sequential(nn.Linear(512, config.width), nn.GELU(), nn.Linear(config.width, config.embedding_dim))
        for key, dim in dims.items():
            self.register_buffer(key + "_mean", torch.zeros(dim))
            self.register_buffer(key + "_std", torch.ones(dim))

    @torch.no_grad()
    def fit_normalization(self, dataset):
        if dataset.split != "train":
            raise ValueError("Evaluator normalization requires train split")
        for key, source in (("human", "human_params"), ("object", "object"), ("contact", "surface_contact")):
            count, total, square = 0, 0, 0
            for record in dataset.loaded:
                x = torch.as_tensor(record[source], dtype=torch.float64)
                count += len(x)
                total, square = total + x.sum(0), square + x.square().sum(0)
            mean = total / count
            getattr(self, key + "_mean").copy_(mean.float())
            getattr(self, key + "_std").copy_((square / count - mean.square()).clamp_min(0).sqrt().clamp_min(.05).float())

    def encode_motion(self, batch, view):
        keys = MODALITIES if view == "joint" else (view,)
        values = torch.cat([(batch[k] - getattr(self, k + "_mean")) / getattr(self, k + "_std") for k in keys], -1)
        return self.motion[view](values, batch["valid_frames"].long().sum(-1), batch["fps"])

    def encode_text(self, features):
        return F.normalize(self.text(F.normalize(features, dim=-1)), dim=-1)

    def loss(self, batch):
        text = self.encode_text(batch["text_embedding"])
        source, captions = batch["source_id"], batch["caption_key"]
        positives = torch.tensor([[s == t or c == d for t, d in zip(source, captions)]
                                  for s, c in zip(source, captions)], device=text.device)
        if positives.all():
            raise ValueError("Contrastive batch has no negative pairs; increase source/caption diversity")
        target = positives.float() / positives.sum(-1, keepdim=True)
        losses = []
        for view in VIEWS:
            logits = text @ self.encode_motion(batch, view).T / self.config.temperature
            # Multi-positive targets prevent repeated captions/windows becoming false negatives.
            losses.append(-.5 * ((target * logits.log_softmax(-1)).sum(-1).mean()
                                 + (target * logits.T.log_softmax(-1)).sum(-1).mean()))
        return torch.stack(losses).mean()


def evaluator_payload(model, metadata):
    return dict(kind=KIND, model_config=asdict(model.config), model=model.state_dict(),
                text_encoder=TEXT_ENCODER, **metadata)


def load_evaluator(path, dataset, support_hash, records, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("kind") != KIND or saved.get("training_split") != "train" or saved.get("step", 0) < 1:
        raise ValueError("Need an independently trained native motion-text evaluator checkpoint")
    config = EvaluatorConfig(**saved["model_config"])
    if config.dataset != dataset or saved.get("support_sha256") != support_hash or saved.get("text_encoder") != TEXT_ENCODER:
        raise ValueError("Evaluator dataset/support/text encoder mismatch")
    if not saved.get("training_sources"):
        raise ValueError("Evaluator checkpoint lacks training split provenance")
    observed = {r.get("source_sequence_id", r["sequence_id"]) for r in records}
    if observed.intersection(saved["training_sources"]):
        raise ValueError("Evaluator training/evaluation source leakage")
    model = MotionTextEvaluator(config)
    model.load_state_dict(saved["model"])
    return model.to(device).eval().requires_grad_(False), saved
