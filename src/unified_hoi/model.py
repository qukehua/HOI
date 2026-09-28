"""Temporal TriDi-style joint denoising with Kimodo-style observed-feature conditioning.

This is a new HOI model; its weights are NOT compatible with either published checkpoint.
The official TriDi Projection and timestep embedding primitives are reused under MIT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .geometry import human_in_object_frame, nearest_surface
from .normalization import StateNormalizer
from .integrations._tridi_primitives import Projection, get_timestep_embedding


@dataclass
class ModelConfig:
    joints: int = 22
    width: int = 192
    heads: int = 6
    layers: int = 4
    dropout: float = .1
    text_dim: int = 512
    relations: bool = True
    two_stage: bool = True

    def asdict(self):
        return asdict(self)


def positions(length, width, device, dtype):
    idx = torch.arange(length, device=device, dtype=dtype)[:, None]
    freq = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10000) / width))
    return torch.stack(((idx * freq).sin(), (idx * freq).cos()), -1).flatten(-2)[:, :width]


class FactorizedBlock(nn.Module):
    """Spatial attention per frame, temporal attention per token; avoids O((T*J)^2)."""
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.spatial = nn.TransformerEncoderLayer(width, heads, width * 4, dropout,
                                                 batch_first=True, norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoderLayer(width, heads, width * 4, dropout,
                                                  batch_first=True, norm_first=True, activation="gelu")

    def forward(self, x, valid):
        b, t, n, d = x.shape
        x = self.spatial(x.reshape(b * t, n, d)).reshape(b, t, n, d)
        y = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
        padding = (~valid)[:, None].expand(b, n, t).reshape(b * n, t)
        y = self.temporal(y, src_key_padding_mask=padding)
        return y.reshape(b, n, t, d).permute(0, 2, 1, 3) * valid[:, :, None, None]


class UnifiedHOIDenoiser(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        j, d = config.joints, config.width
        if d % config.heads or d % 2 or d < 8:
            raise ValueError("width must be even and divisible by heads")
        self.normalizer = StateNormalizer(j)
        self.human_in = Projection(18, d)
        self.object_in = Projection(18, d)
        self.contact_in = Projection(2, d)
        self.joint_embed = nn.Parameter(torch.randn(1, 1, j, d) * .02)
        self.modality_embed = nn.Parameter(torch.randn(3, d) * .02)
        self.time_proj = Projection(d, d)
        self.geometry_net = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.shape_net = Projection(j * 3, d)
        self.text_net = Projection(config.text_dim, d)
        self.fps_net = Projection(1, d)
        if config.two_stage:
            self.coarse_context = Projection(j * 18 + 18 + j * 2, d)
            self.coarse_block = nn.TransformerEncoderLayer(d, config.heads, d * 4, config.dropout,
                                                         activation="gelu", batch_first=True, norm_first=True)
            self.coarse_head = Projection(d, 18)
            self.coarse_to_body = Projection(18, d)
        if config.relations:
            # Object-local joint location, nearest surface offset, distance, contact and observation flags.
            self.relation_net = Projection(12, d)
        self.blocks = nn.ModuleList(FactorizedBlock(d, config.heads, config.dropout)
                                    for _ in range(config.layers))
        self.human_head = Projection(d, 9)
        self.object_head = Projection(d, 9)
        self.contact_head = Projection(d, 1)

    def forward(self, states, timesteps, controls, batch):
        """states/controls in normalized units, timesteps[B,3] independent across H/O/C."""
        x = controls.apply(states)
        m = controls.masks
        valid = controls.valid_frames
        if not valid.any(1).all():
            raise ValueError("Every clip needs at least one valid frame")
        b, t, j, _ = x["human"].shape
        d = self.config.width
        text = batch.get("text_features", x["object"].new_zeros(b, self.config.text_dim))
        geometry = self.geometry_net(batch["object_points"]).amax(1)
        shape = self.shape_net(batch["rest_offsets"].flatten(1))
        fps = batch.get("fps", x["object"].new_full((b,), 30.)).reshape(b, 1).to(text)
        condition = geometry + shape + self.text_net(text) + self.fps_net(torch.log(fps.clamp_min(1)))
        pe = positions(t, d, x["human"].device, x["human"].dtype)[None, :, None]
        times = [self.time_proj(get_timestep_embedding(d, timesteps[:, i], x["human"].device))
                 for i in range(3)]
        hi = torch.cat((x["human"], m["human"].to(x["human"])), -1)
        oi = torch.cat((x["object"], m["object"].to(x["object"])), -1)
        ci = torch.stack((x["contact"], m["contact"].to(x["contact"])), -1)
        h = self.human_in(hi) + self.joint_embed + self.modality_embed[0] + times[0][:, None, None]
        o = self.object_in(oi).unsqueeze(2) + self.modality_embed[1] + times[1][:, None, None]
        c = self.contact_in(ci) + self.joint_embed + self.modality_embed[2] + times[2][:, None, None]
        if self.config.two_stage:
            coarse_in = torch.cat((hi.flatten(2), oi, ci.flatten(2)), -1)
            context = self.coarse_context(coarse_in) + condition[:, None] + pe.squeeze(2)
            context = context + sum(times)[:, None]
            coarse = self.coarse_head(self.coarse_block(context, src_key_padding_mask=~valid))
            # Both coarse roots and object poses see all conditions, then condition detailed tokens.
            root = torch.where(m["human"][:, :, 0], controls.values["human"][:, :, 0], coarse[..., :9])
            obj = torch.where(m["object"], controls.values["object"], coarse[..., 9:])
            stage = self.coarse_to_body(torch.cat((root, obj), -1))[:, :, None]
            h, o, c = h + stage, o + stage, c + stage
        if self.config.relations:
            physical = self.normalizer.decode(x)
            local = human_in_object_frame(physical["human"][..., :3], physical["object"])
            distance, near = nearest_surface(physical["human"][..., :3], physical["object"], batch["object_points"])
            relation = torch.cat((local, near - local, distance[..., None],
                                  physical["contact"][..., None],
                                  m["human"][..., :3].any(-1, keepdim=True).to(local),
                                  m["object"][..., :3].any(-1)[:, :, None, None].expand(b, t, j, 1).to(local),
                                  m["contact"][..., None].to(local),
                                  m["object"][..., 3:].all(-1)[:, :, None, None].expand(b, t, j, 1).to(local)), -1)
            rel = self.relation_net(relation)
            h, c, o = h + rel, c + rel, o + rel.mean(2, keepdim=True)
        tokens = torch.cat((h, o, c), 2) + pe + condition[:, None, None]
        for block in self.blocks:
            tokens = block(tokens, valid)
        return {"human": self.human_head(tokens[:, :, :j]),
                "object": self.object_head(tokens[:, :, j]),
                "contact": self.contact_head(tokens[:, :, j + 1:]).squeeze(-1)}
