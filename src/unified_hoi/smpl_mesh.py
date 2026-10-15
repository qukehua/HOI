"""Map the 22-joint HOI skeleton back to an SMPL-H mesh.

Training stores each joint as world position plus the first two rows of its
global rotation. Those globals are exactly the SMPL kinematic chain after the
dataset world transform, so local joint rotations are recovered by

    R_local[0] = R_global[0]
    R_local[j] = R_global[parent].T @ R_global[j]

Body shape is the SMPL-H beta vector whose T-pose bone offsets match the
stored ``rest_offsets``. Hands are the template pose: this representation has
no finger joints. The root translation is chosen so the pelvis sits on the
stored root joint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .geometry import SMPL_PARENTS, rotation6d_to_matrix

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "smplx_models"
_LAYERS: dict = {}


def _layer(model_path, gender, batch_size):
    key = (str(Path(model_path).resolve()), gender, int(batch_size))
    layer = _LAYERS.get(key)
    if layer is None:
        import smplx
        layer = smplx.build_layer(
            key[0], model_type="smplh", gender=gender, use_pca=False,
            num_betas=10, batch_size=key[2],
        )
        _LAYERS[key] = layer
    return layer


def _identity_pose(count, joints):
    eye = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
    return eye.repeat(count, joints, 1, 1)


def fit_betas(rest_offsets, gender, model_path=DEFAULT_MODEL_PATH, steps=60) -> np.ndarray:
    """10 SMPL-H betas whose T-pose bone vectors match ``rest_offsets`` [22,3]."""
    target = torch.as_tensor(np.asarray(rest_offsets, np.float32))
    if target.shape != (22, 3):
        raise ValueError("rest_offsets must be [22,3]")
    model = _layer(model_path, gender, 1)
    betas = torch.zeros(1, 10, requires_grad=True)
    optimizer = torch.optim.Adam([betas], lr=0.08)
    eye = _identity_pose(1, 1)
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(
            betas=betas, transl=torch.zeros(1, 3), global_orient=eye,
            body_pose=_identity_pose(1, 21), left_hand_pose=_identity_pose(1, 15),
            right_hand_pose=_identity_pose(1, 15), pose2rot=False,
        )
        joints = output.joints[0, :22]
        offsets = torch.zeros(22, 3)
        for joint in range(1, 22):
            offsets[joint] = joints[joint] - joints[SMPL_PARENTS[joint]]
        loss = (offsets - target).square().sum()
        loss.backward()
        optimizer.step()
    return betas.detach().cpu().numpy()[0]


def local_rotations(global_rotations: np.ndarray) -> np.ndarray:
    """[T,22,3,3] global rotations -> local SMPL rotations, same layout."""
    local = np.empty_like(global_rotations)
    local[:, 0] = global_rotations[:, 0]
    for joint in range(1, 22):
        parent = global_rotations[:, SMPL_PARENTS[joint]]
        local[:, joint] = np.einsum("tij,tjk->tik", parent.transpose(0, 2, 1), global_rotations[:, joint])
    return local


def skeleton_to_smpl(human, rest_offsets, gender, model_path=DEFAULT_MODEL_PATH, betas=None):
    """human [T,22,9] -> vertices [T,V,3], faces [F,3], dominant body joint [V].

    ``dominant`` is the SMPL-H body joint with the largest skinning weight, for
    coloring contact or per-joint error on the surface.
    """
    human = np.asarray(human, np.float32)
    if human.ndim != 3 or human.shape[1:] != (22, 9):
        raise ValueError("human must be [T,22,9]")
    frames = len(human)
    if betas is None:
        betas = fit_betas(rest_offsets, gender, model_path)
    betas = np.asarray(betas, np.float32).reshape(10)
    globals_ = rotation6d_to_matrix(torch.from_numpy(human[..., 3:])).numpy()
    local = local_rotations(globals_).astype(np.float32)
    model = _layer(model_path, gender, frames)
    output = model(
        betas=torch.from_numpy(betas).expand(frames, -1).contiguous(),
        transl=torch.zeros(frames, 3),
        global_orient=torch.from_numpy(local[:, :1]),
        body_pose=torch.from_numpy(local[:, 1:]),
        left_hand_pose=_identity_pose(frames, 15),
        right_hand_pose=_identity_pose(frames, 15),
        pose2rot=False,
    )
    joints = output.joints[:, :22].detach().cpu().numpy()
    shift = human[:, 0, :3] - joints[:, 0]
    vertices = output.vertices.detach().cpu().numpy() + shift[:, None, :]
    faces = np.asarray(model.faces, np.int64)
    weights = model.lbs_weights[:, :22].detach().cpu().numpy()
    dominant = weights.argmax(axis=1).astype(np.int64)
    return vertices.astype(np.float64), faces, dominant
