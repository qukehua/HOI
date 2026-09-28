"""Geometric objectives. These do not certify dynamic feasibility."""
import torch
from torch.nn import functional as F

from .geometry import forward_kinematics, human_in_object_frame, nearest_surface, rotation6d_to_matrix


def masked_mean(value, mask):
    mask = torch.broadcast_to(mask, value.shape).to(value)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def geometric_losses(states, batch, contact_target=None, contact_mask=None, threshold=.05):
    human, obj = states["human"], states["object"]
    valid = batch["valid_frames"].bool()
    threshold = torch.as_tensor(batch.get("contact_threshold", threshold), device=human.device).reshape(-1, 1, 1)
    distance, _ = nearest_surface(human[..., :3], obj, batch["object_points"])
    target = states["contact"].detach().clamp(0, 1) if contact_target is None else contact_target
    cmask = valid[..., None] if contact_mask is None else contact_mask & valid[..., None]
    contact = masked_mean(distance.square(), cmask * target)
    noncontact = masked_mean(F.relu(threshold - distance).square(), cmask * (1 - target))
    rotations = rotation6d_to_matrix(human[..., 3:])
    fk = forward_kinematics(rotations, batch["rest_offsets"][:, None], human[:, :, 0, :3])
    kinematic = masked_mean((human[..., :3] - fk).square(), valid[..., None, None])
    # Do not push camera-coordinate data toward a fictitious zero-height ground.
    # A floor penalty is enabled only when a calibrated scene floor is supplied.
    floor_y = torch.as_tensor(batch.get("floor_y", 0.), device=human.device).reshape(-1, 1, 1)
    floor_available = torch.as_tensor(batch.get("floor_available", False), device=human.device).reshape(-1, 1, 1)
    floor = masked_mean(F.relu(floor_y - human[..., 1]).square(), valid[..., None] & floor_available)
    local = human_in_object_frame(human[..., :3], obj)
    if human.shape[1] > 1:
        pair_valid = valid[:, 1:] & valid[:, :-1]
        pair_contact = target[:, 1:] * target[:, :-1]
        if contact_mask is not None:
            pair_contact = pair_contact * cmask[:, 1:] * cmask[:, :-1]
        fps = batch.get("fps", human.new_full((human.shape[0],), 30.)).reshape(-1, 1, 1, 1)
        slip = masked_mean(((local[:, 1:] - local[:, :-1]) * fps).square(),
                           pair_valid[..., None, None] * pair_contact[..., None])
    else:
        slip = human.sum() * 0
    return {"contact": contact, "noncontact": noncontact, "fk": kinematic,
            "floor": floor, "slip": slip}


def reconstruction_loss(pred, target, masks, valid):
    parts = {}
    for key in target:
        shape = [*valid.shape] + [1] * (target[key].ndim - 2)
        support = valid.reshape(shape)
        # Unknown coordinates are the generative task; a small observed loss trains conditioning fidelity.
        weight = (~masks[key]).to(pred[key]) + .05 * masks[key]
        parts[key] = masked_mean((pred[key] - target[key]).square(), support * weight)
    return parts
