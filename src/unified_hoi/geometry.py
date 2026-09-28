"""Differentiable geometry in meters, Y-up, column-vector rotations.

6D features store the FIRST TWO ROWS of R, following PyTorch3D (NOT Kimodo's columns).
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def matrix_to_rotation6d(matrix: Tensor) -> Tensor:
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rotation6d_to_matrix(d6: Tensor) -> Tensor:
    a, b = d6[..., :3], d6[..., 3:]
    # Degenerate early diffusion predictions need a valid, finite SO(3) fallback.
    default = torch.zeros_like(a)
    default[..., 0] = 1
    u = F.normalize(torch.where(a.norm(dim=-1, keepdim=True) > 1e-7, a, default), dim=-1)
    orth = b - (u * b).sum(-1, keepdim=True) * u
    axis = F.one_hot(u.abs().argmin(-1), 3).to(u.dtype)
    fallback = axis - (axis * u).sum(-1, keepdim=True) * u
    v = F.normalize(torch.where(orth.norm(dim=-1, keepdim=True) > 1e-7, orth, fallback), dim=-1)
    w = torch.cross(u, v, dim=-1)
    return torch.stack((u, v, w), dim=-2)


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*x.shape, 3, 3)
    angle = axis_angle.norm(dim=-1, keepdim=True)[..., None]
    a = torch.sinc(angle / torch.pi)
    b = 0.5 * torch.sinc(angle / (2 * torch.pi)).square()
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return eye + a * skew + b * (skew @ skew)


def forward_kinematics(global_rotations: Tensor, rest_offsets: Tensor, root_position: Tensor,
                       parents=SMPL_PARENTS) -> Tensor:
    """Global rotations [...,J,3,3], offsets [...,J,3], root [...,3] -> [...,J,3]."""
    j = global_rotations.shape[-3]
    if len(parents) < j:
        raise ValueError("Insufficient parent indices")
    positions = [root_position]
    for i in range(1, j):
        offset = rest_offsets[..., i, :]
        positions.append(positions[parents[i]] + (global_rotations[..., parents[i], :, :] @
                                                  offset.unsqueeze(-1)).squeeze(-1))
    return torch.stack(positions, -2)


def object_to_world(points: Tensor, object_state: Tensor) -> Tensor:
    """points[B,K,3], object[B,T,9] -> world points[B,T,K,3]."""
    rot = rotation6d_to_matrix(object_state[..., 3:9])
    return torch.einsum("btij,bkj->btki", rot, points) + object_state[..., None, :3]


def human_in_object_frame(human_positions: Tensor, object_state: Tensor) -> Tensor:
    rot = rotation6d_to_matrix(object_state[..., 3:9])
    return torch.einsum("btij,btnj->btni", rot.transpose(-1, -2),
                        human_positions - object_state[..., None, :3])


def nearest_surface(human_positions: Tensor, object_state: Tensor, points: Tensor,
                    chunk_size: int = 32) -> tuple[Tensor, Tensor]:
    """Unsigned sampled-surface distance and object-local nearest points.

    This is a contact proxy, not a watertight signed penetration or force test.
    """
    local = human_in_object_frame(human_positions, object_state)
    distances, nearest = [], []
    for start in range(0, local.shape[1], chunk_size):
        part = local[:, start:start + chunk_size]
        flat = part.flatten(1, 2)
        dist = torch.cdist(flat, points)
        vals, idx = dist.min(-1)
        target = points.gather(1, idx[..., None].expand(-1, -1, 3))
        distances.append(vals.reshape(*part.shape[:-1]))
        nearest.append(target.reshape_as(part))
    return torch.cat(distances, 1), torch.cat(nearest, 1)


def rotation_angle(a: Tensor, b: Tensor) -> Tensor:
    rel = a @ b.transpose(-1, -2)
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.acos(cos)
