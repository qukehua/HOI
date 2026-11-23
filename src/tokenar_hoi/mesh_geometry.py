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

