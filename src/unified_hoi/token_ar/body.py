"""Native body parameters and differentiable, exact selected-vertex LBS.

Human layout: betas, root+21 body+15 left+15 right LOCAL rotations (6D rows),
SMPL transl. SMPL-X jaw/eyes/expression are explicitly neutral, not supervised.
The 73 evaluation joints share SMPL-H semantics: 52 articulation joints and
the 21 vertex-selector landmarks. SMPL-X face-only joints are excluded.
"""
import hashlib

import numpy as np
import torch
from torch import nn

from ..body_metadata import DATASET_MODELS, MODEL_BETAS
from ..geometry import rotation6d_to_matrix
from ..smpl_mesh import DEFAULT_MODEL_PATH, _layer, model_asset_path


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def beta_count(dataset):
    return MODEL_BETAS[DATASET_MODELS[dataset]]


def human_dim(dataset):
    return beta_count(dataset) + 312 + 3


def split_human(human, dataset):
    n = beta_count(dataset)
    if human.shape[-1] != human_dim(dataset):
        raise ValueError("Human dimension does not match the isolated dataset")
    return human[..., :n], human[..., n:n + 312].reshape(*human.shape[:-1], 52, 6), human[..., -3:]


def native_rotations(pose6, dataset):
    rotations = rotation6d_to_matrix(pose6)
    if dataset == "omomo":
        identity = torch.eye(3, device=pose6.device, dtype=pose6.dtype)
        neutral = identity.expand(*pose6.shape[:-2], 3, 3, 3)
        rotations = torch.cat((rotations[..., :22, :, :], neutral, rotations[..., 22:, :, :]), -3)
    return rotations


def canonical_joint_indices(dataset):
    return list(range(52)) if dataset == "behave" else list(range(22)) + list(range(25, 55))




class SelectedBody(nn.Module):
    """Exact SMPL LBS for contact vertices plus 21 landmark vertices; no mesh approximation."""
    def __init__(self, dataset, gender, indices, models=DEFAULT_MODEL_PATH):
        super().__init__()
        self.dataset, self.gender = dataset, gender
        layer = _layer(models, gender, model_type=DATASET_MODELS[dataset])
        self.asset_hash = digest_file(model_asset_path(models, gender, model_type=DATASET_MODELS[dataset]))
        landmarks = layer.vertex_joint_selector.extra_joints_idxs.detach().cpu()
        if len(landmarks) != 21:
            raise ValueError("Expected TriDi-compatible 21 extra vertex landmarks")
        selected = torch.cat((torch.as_tensor(indices, dtype=torch.long), landmarks))
        self.contact_count = len(indices)
        self.register_buffer("template", layer.v_template.detach()[selected].clone())
        self.register_buffer("shape_dirs", layer.shapedirs.detach()[selected, :, :beta_count(dataset)].clone())
        self.register_buffer("pose_dirs", layer.posedirs.detach().reshape(-1, len(layer.v_template), 3)[:, selected].flatten(1).clone())
        self.register_buffer("weights", layer.lbs_weights.detach()[selected].clone())
        self.register_buffer("joint_template", (layer.J_regressor @ layer.v_template).detach().clone())
        self.register_buffer("joint_shape_dirs", torch.einsum("jv,vck->jck", layer.J_regressor,
                             layer.shapedirs[:, :, :beta_count(dataset)]).detach().clone())
        self.register_buffer("parents", layer.parents.detach().clone())

    def forward(self, human):
        from smplx.lbs import batch_rigid_transform
        shape = human.shape[:-1]
        flat = human.reshape(-1, human.shape[-1])
        beta, pose, translation = split_human(flat, self.dataset)
        rotation = native_rotations(pose, self.dataset)
        joints = self.joint_template + torch.einsum("bk,jck->bjc", beta, self.joint_shape_dirs)
        articulated, transforms = batch_rigid_transform(rotation, joints, self.parents, dtype=human.dtype)
        identity = torch.eye(3, device=human.device, dtype=human.dtype)
        posed = self.template + torch.einsum("bk,vck->bvc", beta, self.shape_dirs)
        posed = posed + ((rotation[:, 1:] - identity).flatten(1) @ self.pose_dirs).reshape_as(posed)
        transform = (self.weights @ transforms.flatten(2)).reshape(len(flat), -1, 4, 4)
        vertices = (transform[..., :3, :3] @ posed.unsqueeze(-1)).squeeze(-1) + transform[..., :3, 3]
        vertices = vertices + translation[:, None]
        canonical = articulated[:, canonical_joint_indices(self.dataset)] + translation[:, None]
        joints73 = torch.cat((canonical, vertices[:, self.contact_count:]), 1)
        return (vertices[:, :self.contact_count].reshape(*shape, self.contact_count, 3),
                joints73.reshape(*shape, 73, 3))

    def root_rest(self, beta):
        return self.joint_template[0] + torch.einsum("...k,ck->...c", beta, self.joint_shape_dirs[0])


class BodyGeometry(nn.Module):
    def __init__(self, dataset, support, models=DEFAULT_MODEL_PATH):
        super().__init__()
        self.dataset = dataset
        with np.load(support, allow_pickle=False) as data:
            if str(data["dataset"]) != dataset:
                raise ValueError("Cross-dataset contact support is prohibited")
            indices = data["indices"].copy()
        self.support_hash = digest_file(support)
        self.layers = nn.ModuleDict({g: SelectedBody(dataset, g, indices, models) for g in ("male", "female")})

    def forward(self, human, gender):
        if human.ndim != 3 or gender.shape != human.shape[:1]:
            raise ValueError("Body decode expects human[B,T,D] and gender[B]")
        vertices = human.new_zeros((*human.shape[:2], 690, 3))
        joints = human.new_zeros((*human.shape[:2], 73, 3))
        for value, name in ((0, "female"), (1, "male")):
            idx = (gender == value).nonzero().flatten()
            if len(idx):
                v, j = self.layers[name](human[idx])
                vertices = vertices.index_copy(0, idx, v)
                joints = joints.index_copy(0, idx, j)
        if not ((gender == 0) | (gender == 1)).all():
            raise ValueError("Only source male/female assets are configured")
        return vertices, joints


@torch.no_grad()
def decode_full(human, dataset, gender, models=DEFAULT_MODEL_PATH):
    """Decode native parameters to the full licensed body mesh and canonical 73 joints on CPU."""
    human = torch.as_tensor(human, dtype=torch.float32, device="cpu")
    if human.ndim != 2 or not len(human) or not torch.isfinite(human).all():
        raise ValueError("Expected finite native human[T,D]")
    layer = _layer(models, gender, model_type=DATASET_MODELS[dataset])
    vertices, joints = [], []
    for part in human.split(64):
        beta, pose, trans = split_human(part, dataset)
        rotation = rotation6d_to_matrix(pose)
        out = layer(betas=beta, transl=trans, global_orient=rotation[:, :1], body_pose=rotation[:, 1:22],
                    left_hand_pose=rotation[:, 22:37], right_hand_pose=rotation[:, 37:52], pose2rot=False)
        indices = list(range(73)) if dataset == "behave" else canonical_joint_indices(dataset) + list(range(55, 76))
        vertices.append(out.vertices.numpy())
        joints.append(out.joints[:, indices].numpy())
    return np.concatenate(vertices), np.asarray(layer.faces, np.int64), np.concatenate(joints)


@torch.no_grad()
def standard_parameters(human, dataset):
    """Portable SMPL-H/SMPL-X arguments in axis-angle, including explicit hand poses."""
    from scipy.spatial.transform import Rotation
    human = torch.as_tensor(human, dtype=torch.float32, device="cpu")
    beta, pose, trans = split_human(human, dataset)
    matrices = rotation6d_to_matrix(pose).detach().numpy()
    aa = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().astype(np.float32).reshape(-1, 52, 3)
    result = {"betas": beta.numpy(), "global_orient": aa[:, 0], "body_pose": aa[:, 1:22].reshape(-1, 63),
              "left_hand_pose": aa[:, 22:37].reshape(-1, 45), "right_hand_pose": aa[:, 37:52].reshape(-1, 45),
              "transl": trans.numpy()}
    if dataset == "omomo":
        result.update(expression=np.zeros((len(human), 10), np.float32),
                      jaw_pose=np.zeros((len(human), 3), np.float32),
                      leye_pose=np.zeros((len(human), 3), np.float32), reye_pose=np.zeros((len(human), 3), np.float32))
    return result
