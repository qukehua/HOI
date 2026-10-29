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

