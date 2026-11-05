"""Remove one shared horizontal anchor before quantization and put it back after decoding.

The anchor is the first frame's yaw about SMPL Y and its horizontal translation.
Human and object use the same rigid transform, so their relative pose is unchanged.
Local joint rotations, height, shape and contact labels stay in their original channels.
"""
import torch

from ..geometry import matrix_to_rotation6d, rotation6d_to_matrix
from .body import beta_count, split_human


def _as_batch(human, obj):
    single = human.ndim == 2
    if single:
        human, obj = human[None], obj[None]
    if human.ndim != 3 or obj.ndim != 3 or human.shape[0] != obj.shape[0] or human.shape[1] != obj.shape[1]:
        raise ValueError("Human and object windows must share batch and time dimensions")
    return human, obj, single


def yaw_matrix(yaw):
    """Right-handed rotation about Y. yaw is [B]."""
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    zeros, ones = torch.zeros_like(cosine), torch.ones_like(cosine)
    return torch.stack((
        torch.stack((cosine, zeros, sine), -1),
        torch.stack((zeros, ones, zeros), -1),
        torch.stack((-sine, zeros, cosine), -1),
    ), -2)


def scene_anchor(human, dataset):
    """Yaw and horizontal origin from the first frame of each window."""
    human, _, _ = _as_batch(human, human[..., -3:])
    _, pose, trans = split_human(human[:, :1], dataset)
    root = rotation6d_to_matrix(pose[:, 0, 0])
    forward = root[:, :, 2]
    yaw = torch.atan2(forward[:, 0], forward[:, 2])
    flat = forward[:, [0, 2]].square().sum(-1).sqrt() < 1e-6
    yaw = torch.where(flat, torch.zeros_like(yaw), yaw)
    origin = trans[:, 0].clone()
    origin[:, 1] = 0
    return yaw_matrix(-yaw).detach(), origin.detach()


def _apply(human, obj, rotation, origin, dataset, invert):
    human, obj, single = _as_batch(human, obj)
    beta, pose, trans = split_human(human, dataset)
    root = rotation6d_to_matrix(pose[:, :, 0])
    obj_rot = rotation6d_to_matrix(obj[..., 3:])
    if invert:
        trans = torch.einsum("bij,btj->bti", rotation, trans) + origin[:, None]
        root = torch.einsum("bij,btjk->btik", rotation, root)
        obj_trans = torch.einsum("bij,btj->bti", rotation, obj[..., :3]) + origin[:, None]
        obj_rot = torch.einsum("bij,btjk->btik", rotation, obj_rot)
    else:
        trans = torch.einsum("bij,btj->bti", rotation, trans - origin[:, None])
        root = torch.einsum("bij,btjk->btik", rotation, root)
        obj_trans = torch.einsum("bij,btj->bti", rotation, obj[..., :3] - origin[:, None])
        obj_rot = torch.einsum("bij,btjk->btik", rotation, obj_rot)
    pose = pose.clone()
    pose[:, :, 0] = matrix_to_rotation6d(root)
    human = torch.cat((beta, pose.flatten(2), trans), -1)
    obj = torch.cat((obj_trans, matrix_to_rotation6d(obj_rot)), -1)
    if single:
        human, obj = human[0], obj[0]
    return human, obj


def to_canonical(human, obj, dataset):
    if beta_count(dataset) + 315 != human.shape[-1]:
        raise ValueError("Canonical scene expects native human parameters")
    rotation, origin = scene_anchor(human, dataset)
    canonical_human, canonical_obj = _apply(human, obj, rotation, origin, dataset, invert=False)
    return canonical_human, canonical_obj, {"rotation": rotation, "origin": origin}


def restore_scene(human, obj, anchor, dataset):
    if anchor is None:
        return human, obj
    return _apply(human, obj, anchor["rotation"].transpose(-1, -2), anchor["origin"], dataset, invert=True)
