"""Object-frame contact targets for the 690 evaluation vertices.

Each vertex stores the nearest object-mesh point in the object frame (3) and the
distance (1). Points farther than 5 cm keep a capped distance and zero coordinates,
so the codebook is not spent on the far field. Binary contact is distance < 5 cm.
"""
import numpy as np
import torch

CONTACT_VERTICES = 690
CONTACT_CHANNELS = 4
CONTACT_DIM = CONTACT_VERTICES * CONTACT_CHANNELS
CONTACT_CAP = 0.05
FAR_WEIGHT = 0.05


def _distance(offset):
    if isinstance(offset, torch.Tensor):
        return offset.reshape(*offset.shape[:-1], CONTACT_VERTICES, CONTACT_CHANNELS)[..., 3]
    array = np.asarray(offset, np.float32)
    return array.reshape(array.shape[:-1] + (CONTACT_VERTICES, CONTACT_CHANNELS))[..., 3]


def contact_active(offset):
    """True where the stored distance is inside the 5 cm cap."""
    return _distance(offset) < CONTACT_CAP


def contact_weight(target, far=FAR_WEIGHT):
    active = contact_active(target)
    weight = torch.where(active, torch.ones_like(active, dtype=target.dtype), target.new_full(active.shape, far))
    return weight.unsqueeze(-1).expand(*weight.shape, CONTACT_CHANNELS).reshape(target.shape)


def project_contact_offset(value, cap=CONTACT_CAP):
    """Clamp distance to [0, cap] and drop coordinates of vertices at the cap."""
    feat = value.reshape(*value.shape[:-1], CONTACT_VERTICES, CONTACT_CHANNELS)
    dist = feat[..., 3].clamp(0, cap)
    point = torch.where((dist >= cap).unsqueeze(-1), torch.zeros_like(feat[..., :3]), feat[..., :3])
    return torch.cat((point, dist.unsqueeze(-1)), -1).reshape(value.shape)


def pack_contact_offset(local_point, distance, cap=CONTACT_CAP):
    """Pack nearest object-frame points and distances into (..., 2760)."""
    if local_point.shape[:-1] != distance.shape or local_point.shape[-1] != 3:
        raise ValueError("Expected local points (..., 690, 3) and distances (..., 690)")
    if local_point.shape[-2] != CONTACT_VERTICES:
        raise ValueError("Contact offset uses the 690 support vertices")
    far = distance > cap
    point = torch.where(far.unsqueeze(-1), torch.zeros_like(local_point), local_point)
    stored = torch.cat((point, distance.clamp(max=cap).unsqueeze(-1)), -1)
    return stored.reshape(*distance.shape[:-1], CONTACT_DIM)


def object_world(local_points, pose):
    """Map object-frame points to the world with the same convention as object_local."""
    from ..geometry import rotation6d_to_matrix
    rotation = rotation6d_to_matrix(pose[:, 3:])
    return local_points @ rotation.transpose(-1, -2) + pose[:, None, :3]


def contact_offset_from_surfaces(body_vertices, object_pose, mesh_vertices, cap=CONTACT_CAP):
    """Nearest object vertex for each body contact vertex, packed as a contact feature."""
    from .refinement import nearest_surface
    distance, near, _ = nearest_surface(body_vertices, object_pose, mesh_vertices)
    return pack_contact_offset(near, distance, cap)
