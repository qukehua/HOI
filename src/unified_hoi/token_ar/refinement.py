"""Continuous, geometry-aware residual refinement of frozen AR/VQ predictions.

This module never receives unknown ground truth in forward(). Geometry losses
train the refiner only; discrete AR sampling is deliberately outside autograd.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..geometry import matrix_to_rotation6d, rotation6d_to_matrix
from . import MODALITIES, MODES
from .body import beta_count, human_dim

KIND = "token_ar_refiner_v1"
LOSS_WEIGHTS = dict(body_vertices=1., object_vertices=10., relative_joints=1.,
                    surface_distance=5., contact_bce=1., geometry_contact_bce=.1,
                    contact_consistency=.1, rotation=.1, velocity=.01, residual=.01)


@dataclass
class RefinerConfig:
    dataset: str
    width: int = 192
    layers: int = 4
    contact_threshold: float = .05
    contact_softness: float = .01

    def __post_init__(self):
        if self.dataset not in ("behave", "omomo") or min(self.width, self.layers) < 1:
            raise ValueError("Invalid refiner architecture")
        if not all(math.isfinite(v) and v > 0 for v in (self.contact_threshold, self.contact_softness)):
            raise ValueError("Contact distance settings must be positive metres")


def rotation_delta(vector):
    """SO(3) exponential with finite derivatives at the zero residual."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)
    angle = vector.norm(dim=-1)[..., None, None]
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype)
    return identity + torch.sinc(angle / math.pi) * skew + .5 * torch.sinc(angle / (2 * math.pi)).square() * (skew @ skew)


def object_local(points, pose):
    return (points - pose[:, None, :3]) @ rotation6d_to_matrix(pose[:, 3:])


def nearest_surface(points, pose, vertices):
    """Exact nearest mesh VERTEX proxy, matching the evaluation convention.

    Only the discrete nearest-neighbour lookup is detached. Distances retain
    gradients to human vertices and object SE(3). Chunk both axes to bound RAM.
    """
    local = object_local(points, pose)
    flat = local.reshape(-1, 3)
    indices = []
    with torch.no_grad():
        for query in flat.split(512):
            best = query.new_full((len(query),), float("inf"))
            chosen = torch.zeros(len(query), dtype=torch.long, device=query.device)
            for start in range(0, len(vertices), 2048):
                distance = (query[:, None] - vertices[None, start:start + 2048]).square().sum(-1)
                value, index = distance.min(-1)
                better = value < best
                chosen = torch.where(better, index + start, chosen)
                best = torch.minimum(best, value)
            indices.append(chosen)
    near = vertices[torch.cat(indices)].reshape_as(local)
    return (local - near).norm(dim=-1), near, local


def object_vertex_mse(pred, target, vertices):
    """Exact all-vertex squared error using mesh moments, without T x V meshes."""
    mean = vertices.mean(0)
    centered = vertices - mean
    covariance = centered.T @ centered / len(vertices)
    delta = rotation6d_to_matrix(pred[:, 3:]) - rotation6d_to_matrix(target[:, 3:])
    shift = (delta @ mean) + pred[:, :3] - target[:, :3]
    return ((delta @ covariance * delta).sum((-1, -2)) + shift.square().sum(-1)).mean() / 3


def balanced_mean(error, contact):
    groups = [error[mask].mean() for mask in (contact >= .5, contact < .5) if mask.any()]
    return torch.stack(groups).mean()


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.out = nn.Linear(width, width)

    def forward(self, x):
        value = self.conv(self.norm(x).T[None])[0].T
        return x + self.out(F.gelu(value))


class ContinuousRefiner(nn.Module):
    """One unpadded window; same parameters serve all seven generated-bit masks."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim, width = human_dim(config.dataset), config.width
        for name, size in (("human", dim), ("object", 9)):
            self.register_buffer(name + "_mean", torch.zeros(size))
            self.register_buffer(name + "_std", torch.ones(size))
        self.geometry = nn.Sequential(nn.Linear(3072, width), nn.GELU(), nn.Linear(width, width))
        self.input = nn.Linear(dim + 9 + 690 + 22 * 7 + 4, width)
        self.blocks = nn.Sequential(*(TemporalBlock(width, 2 ** i) for i in range(config.layers)))
        # 52 local rotations, root translation, object rotation/translation, contact logits.
        self.head = nn.Linear(width, 52 * 3 + 3 + 6 + 690)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, coarse, mode, bps, gender, fps, geometry, vertices):
        if mode not in MODES:
            raise ValueError("Invalid generated-bit mask")
        n = len(coarse["human"])
        if n < 1 or any(coarse[k].shape != (n, d) for k, d in
                        (("human", human_dim(self.config.dataset)), ("object", 9), ("contact", 690))):
            raise ValueError("Expected one unpadded H/O/C window")
        if any(not torch.isfinite(v).all() for v in coarse.values()):
            raise ValueError("Nonfinite coarse predictions")
        if not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("Invalid fps")
        h, o, c = (coarse[k].detach().float() for k in MODALITIES)
        if ((c < 0) | (c > 1)).any():
            raise ValueError("Contact input must contain probabilities")
        with torch.no_grad():
            _, joints = geometry(h[None], gender.reshape(1))
            distance, near, local = nearest_surface(joints[0, :, :22], o, vertices)
            relation = torch.cat((local, near - local, distance[..., None]), -1).flatten(1)
        flags = h.new_tensor([int(bit) for bit in mode]).expand(n, -1)
        rate = h.new_full((n, 1), math.log(float(fps)))
        features = torch.cat(((h - self.human_mean) / self.human_std,
                              (o - self.object_mean) / self.object_std, 2 * c - 1, relation, flags, rate), -1)
        hidden = self.input(features) + self.geometry(bps.float().reshape(1, 3072))
        residual = self.head(self.blocks(F.gelu(hidden)))
        pose_delta, translation, obj_delta, contact_delta = residual.split((156, 3, 6, 690), -1)
        result = {k: v.detach() for k, v in coarse.items()}
        if mode[0] == "1":
            beta = beta_count(self.config.dataset)
            pose_delta = pose_delta.reshape(n, 52, 3)
            if self.config.dataset == "omomo":
                pose_delta = torch.cat((pose_delta[:, :22], torch.zeros_like(pose_delta[:, 22:])), 1)
            rotation = rotation_delta(pose_delta) @ rotation6d_to_matrix(h[:, beta:-3].reshape(n, 52, 6))
            result["human"] = torch.cat((h[:, :beta], matrix_to_rotation6d(rotation).flatten(1),
                                         h[:, -3:] + translation), -1)
        if mode[1] == "1":
            rotation = rotation_delta(obj_delta[:, :3]) @ rotation6d_to_matrix(o[:, 3:])
            result["object"] = torch.cat((o[:, :3] + obj_delta[:, 3:], matrix_to_rotation6d(rotation)), -1)
        logits = torch.logit(c.clamp(1e-5, 1 - 1e-5)) + contact_delta
        if mode[2] == "1":
            result["contact"] = logits.sigmoid()
        # Observed modalities are the original tensors, never projected or decoded.
        return result, logits


def refinement_loss(model, coarse, target, mode, bps, gender, fps, geometry, vertices,
                    weights=None, geometry_frames=16):
    coarse = {k: v.detach() for k, v in coarse.items()}
    target = {k: v.detach() for k, v in target.items()}
    weights = LOSS_WEIGHTS | (weights or {})
    if set(weights) != set(LOSS_WEIGHTS) or any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Unknown or invalid refinement loss weight")
    if geometry_frames < 1:
        raise ValueError("geometry_frames must be positive")
    for key, bit in zip(MODALITIES, mode):
        if bit == "0" and not torch.equal(coarse[key], target[key]):
            raise ValueError(f"Observed {key} changed in refinement input")
    output, logits = model(coarse, mode, bps, gender, fps, geometry, vertices)
    n = len(output["human"])
    selected = torch.linspace(0, n - 1, min(n, geometry_frames), device=vertices.device).long()
    pv, pj = geometry(output["human"][None, selected], gender.reshape(1))
    po = output["object"][selected]
    pd, _, _ = nearest_surface(pv[0], po, vertices)
    with torch.no_grad():
        gv, gj = geometry(target["human"][None, selected], gender.reshape(1))
        go = target["object"][selected]
        gd, _, _ = nearest_surface(gv[0], go, vertices)
        relative = object_local(gj[0, :, :22], go)
    contact = target["contact"][selected]
    gl = (model.config.contact_threshold - pd) / model.config.contact_softness
    zero = logits.sum() * 0
    losses = {key: zero for key in LOSS_WEIGHTS}
    if mode[0] == "1":
        losses["body_vertices"] = F.mse_loss(pv, gv)
    if mode[1] == "1":
        losses["object_vertices"] = object_vertex_mse(po, go, vertices)
    if "1" in mode[:2]:
        losses["relative_joints"] = F.mse_loss(object_local(pj[0, :, :22], po), relative)
        losses["surface_distance"] = balanced_mean((pd - gd).square(), contact)
        losses["geometry_contact_bce"] = balanced_mean(F.binary_cross_entropy_with_logits(gl, contact, reduction="none"), contact)
        losses["contact_consistency"] = balanced_mean(F.binary_cross_entropy_with_logits(
            gl, output["contact"][selected].detach(), reduction="none"), contact)
    if mode[2] == "1":
        losses["contact_bce"] = balanced_mean(F.binary_cross_entropy_with_logits(logits, target["contact"], reduction="none"), target["contact"])
        losses["contact_consistency"] = losses["contact_consistency"] + balanced_mean(
            F.binary_cross_entropy_with_logits(logits[selected], gl.detach().sigmoid(), reduction="none"), contact)
    for key, bit in zip(MODALITIES[:2], mode[:2]):
        if bit == "0":
            continue
        start, end = (beta_count(model.config.dataset), -3) if key == "human" else (3, None)
        pred_rot = rotation6d_to_matrix(output[key][:, start:end].reshape(n, -1, 6))
        true_rot = rotation6d_to_matrix(target[key][:, start:end].reshape(n, -1, 6))
        losses["rotation"] = losses["rotation"] + F.mse_loss(pred_rot, true_rot)
        scale = getattr(model, key + "_std")
        losses["residual"] = losses["residual"] + ((output[key] - coarse[key]) / scale).square().mean()
        if n > 1:
            # Match GT velocity, rather than rewarding motionless outputs.
            losses["velocity"] = losses["velocity"] + (((output[key].diff(dim=0) - target[key].diff(dim=0)) * fps / scale).square().mean())
    total = sum(weights[key] * value for key, value in losses.items())
    return total, {key: float(value.detach()) for key, value in losses.items()}, output


def load_refiner(path, ar_metadata, ar_metadata_sha256, geometry, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("kind") != KIND:
        raise ValueError("Expected a trained continuous refiner")
    source = saved["source"]
    expected = dict(dataset=ar_metadata["config"]["dataset"], codec_sha256=ar_metadata["codec_sha256"],
                    support_sha256=ar_metadata["support_sha256"], ar_metadata_sha256=ar_metadata_sha256,
                    body_assets={k: v.asset_hash for k, v in geometry.layers.items()})
    for key, value in expected.items():
        if source[key] != value:
            raise ValueError(f"Refiner/source mismatch: {key}")
    model = ContinuousRefiner(RefinerConfig(**saved["model_config"]))
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval().requires_grad_(False), saved


def check_sampler(saved, temperature, top_k, text_conditioned):
    source = saved["source"]
    if (temperature != source["temperature"] or (temperature != 0 and top_k != source["top_k"])
            or text_conditioned != source["text_conditioned"]):
        raise ValueError("Refiner cache and inference sampler/text settings differ")


def refiner_payload(model, source, **extra):
    return dict(kind=KIND, model_config=asdict(model.config), model=model.state_dict(), source=source, **extra)
