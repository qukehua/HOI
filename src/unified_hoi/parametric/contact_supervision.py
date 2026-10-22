"""GT-supervised contact decoding and class-balanced, full-mesh distances."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from ..geometry import nearest_surface
from ..mesh_geometry import load_mesh_asset


def contact_loss_options(config):
    # Missing options preserve the objective of checkpoints predating these losses.
    defaults = {"decoded_bce_weight": 0., "balance_surface_distance": False, "full_mesh_distance": False}
    supplied = config.get("contact_supervision", {})
    if not isinstance(supplied, dict) or set(supplied) - set(defaults):
        raise ValueError("Invalid contact_supervision options")
    options = defaults | supplied
    weight = options["decoded_bce_weight"]
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0):
        raise ValueError("contact_supervision.decoded_bce_weight must be finite and nonnegative")
    for key in ("balance_surface_distance", "full_mesh_distance"):
        if not isinstance(options[key], bool):
            raise ValueError(f"contact_supervision.{key} must be boolean")
    return options


def balanced_contact_mean(values, target, support=None):
    """Average each present class first, then give the classes equal weight.

    Empty support returns differentiable zero. Single-class batches use that
    class's mean, with no division by zero or artificial missing-class penalty.
    """
    if support is None:
        support = torch.ones_like(target, dtype=torch.bool)
    positive, negative = (target >= .5) & support, (target < .5) & support
    n_pos, n_neg = positive.sum(), negative.sum()
    pos = values.masked_fill(~positive, 0).sum() / n_pos.clamp_min(1)
    neg = values.masked_fill(~negative, 0).sum() / n_neg.clamp_min(1)
    classes = (n_pos > 0).to(values) + (n_neg > 0).to(values)
    return (pos + neg) / classes.clamp_min(1), {
        "positive": pos, "negative": neg, "positive_count": n_pos, "negative_count": n_neg,
    }


def decoded_contact_loss(codec, latent, target, frames, threshold):
    """Decode physical (unnormalized) x0 latents; preserve gradients through frozen AE."""
    logits = codec.decode_logits(latent[frames])
    target = target[frames].to(logits)
    value, stats = balanced_contact_mean(
        F.binary_cross_entropy_with_logits(logits, target, reduction="none"), target)
    with torch.no_grad():
        predicted, positive = logits.sigmoid() >= threshold, target >= .5
        tp = (predicted & positive).sum().float()
        fp, fn = (predicted & ~positive).sum(), (~predicted & positive).sum()
        stats.update(precision=tp / (tp + fp).clamp_min(1), recall=tp / (tp + fn).clamp_min(1),
                     f1=2 * tp / (2 * tp + fp + fn).clamp_min(1), frames=frames.sum())
    return value, stats


def full_mesh_distances(body_vertices, object_state, batch, batch_indices):
    """Distances to all face-referenced vertices, matching stored contact labels.

    Only capped geometry frames call this function. The existing nearest_surface
    chunks large meshes and retains only chosen neighbors in its backward graph.
    Ignore orphan vertices, as the native-data preparer's object_tree did.
    """
    groups = {}
    scales = batch["object_mesh_scale"].detach().cpu().tolist()
    for row, bi in enumerate(batch_indices.detach().cpu().tolist()):
        key = (batch["object_mesh_path"][bi], scales[bi])
        groups.setdefault(key, []).append(row)
    result = body_vertices.new_zeros(body_vertices.shape[:-1])
    for (path, scale), rows in groups.items():
        vertices, faces = load_mesh_asset(path)
        points = torch.tensor(vertices[np.unique(faces)] * scale,
                              device=body_vertices.device, dtype=body_vertices.dtype)
        index = torch.tensor(rows, device=body_vertices.device, dtype=torch.long)
        distance, _ = nearest_surface(body_vertices[index], object_state[index],
                                      points[None].expand(len(rows), -1, -1))
        result = result.index_copy(0, index, distance)
    return result
