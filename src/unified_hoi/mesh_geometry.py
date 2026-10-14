"""Mesh-derived BPS and vertex contact geometry (InfBaGel-inspired, independently implemented).

Coordinates stay in the annotation's object-local frame, in metres. BPS uses a
fixed ordered unit-ball basis and nearest mesh vertices, not a learned mesh GNN.
Vertex distances are unsigned proxies; faces are retained for provenance/rendering.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

BPS_COUNT = 1024
BPS_SEED = 20261014
CONTACT_VERTEX_COUNT = 10000
GEOMETRY_VERSION = 1
CONTACT_DEFINITION = "joint_to_mesh_vertex_distance_proxy"


def validate_mesh(vertices, faces):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError("mesh vertices must be finite [V,3], V >= 3")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in "iu":
        raise ValueError("mesh faces must be integer [F,3], F > 0")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("mesh face index outside vertex array")
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=-1)
    if not np.isfinite(area).all() or not np.any(area > 0):
        raise ValueError("mesh has no finite positive-area triangles")


@lru_cache(maxsize=1)
def basis_points():
    # Local generator: data loading must not consume the model/training RNG.
    rng = np.random.default_rng(BPS_SEED)
    directions = rng.normal(size=(BPS_COUNT, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    basis = (directions * rng.random((BPS_COUNT, 1)) ** (1 / 3)).astype(np.float32)
    basis.setflags(write=False)
    return basis


def encode_mesh(vertices, faces, scale=1.):
    """BPS from ALL referenced mesh vertices; bounded, deterministic contact support.

    Large meshes use up to 10k actual vertices for contact. Small meshes repeat
    vertices to a fixed batchable size; repetition cannot alter nearest distance.
    No centering/normalization is done here: loaders must supply the correct origin.
    """
    validate_mesh(vertices, faces)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("object_mesh_scale must be finite and positive")
    vertices = np.asarray(vertices, np.float64)[np.unique(faces)] * scale
    # Canonical order makes vertex-array permutations irrelevant, including ties.
    vertices = np.unique(vertices, axis=0)
    if not np.isfinite(vertices).all():
        raise ValueError("scaled mesh contains nonfinite vertices")
    basis = basis_points()
    index = cKDTree(vertices).query(basis)[1]
    bps = (vertices[index] - basis).astype(np.float32)
    if len(vertices) > CONTACT_VERTEX_COUNT:
        rng = np.random.default_rng(BPS_SEED)
        selected = np.sort(rng.choice(len(vertices), CONTACT_VERTEX_COUNT, replace=False))
        contact_vertices = vertices[selected]
    else:
        contact_vertices = vertices[np.arange(CONTACT_VERTEX_COUNT) % len(vertices)]
    return bps, contact_vertices.astype(np.float32)


@lru_cache(maxsize=64)
def load_mesh_asset(path):
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["geometry_version"]) != GEOMETRY_VERSION:
            raise ValueError("unsupported object mesh asset version")
        vertices = np.asarray(archive["vertices"], np.float32).copy()
        faces = np.asarray(archive["faces"]).copy()
    validate_mesh(vertices, faces)
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return vertices, faces


@lru_cache(maxsize=64)
def mesh_features(path, scale=1.):
    vertices, faces = load_mesh_asset(str(Path(path).resolve()))
    bps, points = encode_mesh(vertices, faces, scale)
    bps.setflags(write=False)
    points.setflags(write=False)
    return bps, points


def contact_labels(human, obj, points, threshold=.05):
    """Recompute 22-joint targets from the SAME mesh vertices used by the model.

    All inputs are physical, unnormalized arrays. The stored rotation6d contains
    the first two rows; recover row three without applying a second world transform.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("contact threshold must be finite and positive")
    rows = np.asarray(obj)[..., 3:].reshape(-1, 2, 3)
    rotation = np.concatenate((rows, np.cross(rows[:, :1], rows[:, 1:2])), axis=1)
    local = np.einsum("tji,tkj->tki", rotation, np.asarray(human)[..., :3] - np.asarray(obj)[:, None, :3])
    distance = cKDTree(points).query(local.reshape(-1, 3))[0].reshape(local.shape[:-1])
    return (distance <= threshold).astype(np.float32)


def attach_mesh_geometry(record, path=None, scale=1.):
    """Return a copy with BPS, contact vertices and matching labels for a clip/scene."""
    result = dict(record)
    if path is not None:
        if "source_object_scale" in record and not np.isclose(float(record["source_object_scale"]), scale):
            raise ValueError("object mesh scale conflicts with the recorded OMOMO scale")
        bps, points = mesh_features(str(Path(path).resolve()), float(scale))
        result.update(object_bps=bps.copy(), object_points=points.copy(),
                      object_geometry_version=np.int32(GEOMETRY_VERSION),
                      object_geometry_is_mesh=np.bool_(True))
    if "object_bps" not in result:
        raise ValueError("mesh_bps requires an object mesh asset; run scripts/prepare_object_meshes.py")
    if (np.asarray(result["object_bps"]).shape != (BPS_COUNT, 3)
            or not np.isfinite(result["object_bps"]).all()
            or np.asarray(result["object_points"]).shape != (CONTACT_VERTEX_COUNT, 3)
            or not np.isfinite(result["object_points"]).all()
            or not bool(result.get("object_geometry_is_mesh", False))
            or int(result.get("object_geometry_version", -1)) != GEOMETRY_VERSION):
        raise ValueError("invalid mesh BPS/contact geometry or provenance version")
    if "human" in result:
        result["contact"] = contact_labels(result["human"], result["object"], result["object_points"],
                                           float(result.get("contact_threshold", .05)))
    result["contact_definition"] = CONTACT_DEFINITION
    return result
