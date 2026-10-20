"""Versioned body geometry shared by data, FK objectives and mesh decoding."""
from functools import lru_cache
import hashlib

import numpy as np
import torch

from .body_metadata import body_metadata
from .geometry import human_fk
from .mesh_geometry import contact_labels
from .smpl_mesh import DEFAULT_MODEL_PATH, _layer, _offset_basis, model_asset_path

BODY_GEOMETRY_VERSION = 1
BODY_GEOMETRY_KEYS = ("body_geometry_version", "body_model_sha256")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _asset_basis(path, gender, model_type, size, modified):
    # Stat arguments invalidate the cache when an installed model is replaced.
    digest = file_sha256(path)
    model = _layer(path, gender, model_type=model_type)
    base, basis = _offset_basis(model)
    return base, basis, digest


def model_offsets(record, model_path=DEFAULT_MODEL_PATH):
    metadata = body_metadata(record)
    if "gender" not in metadata or "betas" not in metadata:
        raise ValueError("Canonical body geometry requires original gender and betas; "
                         "run scripts/prepare_body_metadata.py first")
    gender, kind = str(metadata["gender"]), str(metadata["body_model_type"])
    asset = model_asset_path(model_path, gender, model_type=kind)
    stat = asset.stat()
    base, basis, digest = _asset_basis(str(asset), gender, kind, stat.st_size, stat.st_mtime_ns)
    offsets = np.zeros((22, 3), np.float32)
    offsets[1:] = base + np.einsum("jck,k->jc", basis, metadata["betas"].reshape(-1))
    return offsets, digest


def model_offsets24(record, model_path=DEFAULT_MODEL_PATH):
    """OMOMO's 22 body joints plus SMPL-X middle-finger bases (28, 43)."""
    metadata = body_metadata(record)
    offsets, _ = model_offsets(record, model_path)
    if str(metadata["body_model_type"]) != "smplx":
        raise ValueError("OMOMO 24-joint adapter requires SMPL-X")
    model = _layer(model_path, str(metadata["gender"]), model_type="smplx")
    # Zero hand pose: these finger bases are direct children of the wrists.
    if model.parents[[28, 43]].tolist() != [20, 21]:
        raise ValueError("Unexpected SMPL-X finger hierarchy")
    regressor = model.J_regressor.detach().numpy()[[20, 21, 28, 43]].astype(np.float64)
    shaped = model.v_template.detach().numpy().astype(np.float64) + np.einsum(
        "vck,k->vc", model.shapedirs.detach().numpy(), metadata["betas"])
    joints = regressor @ shaped
    return np.concatenate((offsets, joints[2:] - joints[:2]), axis=0)


def validate_body_geometry(record, *, model_path=None):
    """Validate persisted FK semantics without needing licensed assets in a loader."""
    if int(record.get("body_geometry_version", -1)) != BODY_GEOMETRY_VERSION:
        raise ValueError("Expected body geometry v1; run scripts/prepare_body_geometry.py "
                         "and use its new manifest")
    digest = str(record.get("body_model_sha256", ""))
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Missing/invalid body model SHA256")
    metadata = body_metadata(record)
    if not {"gender", "betas", "body_model_type"} <= metadata.keys():
        raise ValueError("Body geometry v1 requires complete original body metadata")
    offsets = np.asarray(record["rest_offsets"], np.float32)
    if model_path is not None:
        actual, actual_hash = model_offsets(record, model_path)
        if digest != actual_hash or not np.allclose(offsets, actual, atol=1e-7, rtol=0):
            raise ValueError("Installed body asset differs from canonical data; rebuild data explicitly")
    if "human" in record:
        human = torch.as_tensor(np.asarray(record["human"], np.float32))
        with torch.no_grad():
            xyz = human_fk(human, torch.from_numpy(offsets))
        if not torch.allclose(human[..., :3], xyz, atol=2e-6, rtol=0):
            raise ValueError("Body geometry v1 positions disagree with FK")


def canonicalize_body(record, model_path=DEFAULT_MODEL_PATH):
    """Copy source data; use actual asset + original betas, never refit the shape.

    Original positions, offsets and labels are retained for audits. Attach the
    intended object geometry first so the rebuilt labels share that support.
    Scene-only archives receive offsets/provenance without invented motion.
    """
    if "body_geometry_version" in record:
        validate_body_geometry(record, model_path=model_path)
        return dict(record)
    offsets, digest = model_offsets(record, model_path)
    result = dict(record)
    result["source_rest_offsets"] = np.asarray(record["rest_offsets"]).copy()
    result.update(rest_offsets=offsets, body_geometry_version=np.int32(BODY_GEOMETRY_VERSION),
                  body_model_sha256=np.array(digest))
    if "human" in record:
        human = np.asarray(record["human"], np.float32).copy()
        result["source_body_xyz"] = human[..., :3].copy()
        result["source_contact"] = np.asarray(record["contact"]).copy()
        with torch.no_grad():
            human[..., :3] = human_fk(torch.from_numpy(human), torch.from_numpy(offsets)).numpy()
        result["human"] = human
        result["contact"] = contact_labels(human, record["object"], record["object_points"],
                                           float(record.get("contact_threshold", .05)))
    return result
