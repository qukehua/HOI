"""Decode HOI root/global rotations with the dataset's official body model.

OMOMO uses SMPL-X with 16 betas; BEHAVE uses SMPL-H with 10 betas.
Body rotations occupy joints 0..21 in both models. Fingers (and SMPL-X face)
remain at zero pose, as this representation does not predict them. Nonroot xyz
channels are NOT IK targets; the caller must handle position/FK consistency.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import torch

from .body_metadata import body_metadata, body_model_settings
from .geometry import SMPL_PARENTS, rotation6d_to_matrix

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "smplx_models"
_LAYERS: dict = {}


def model_asset_path(model_path, gender, *, model_type):
    model_type, count = body_model_settings(model_type=model_type)
    gender = str(gender).lower()
    if gender not in {"male", "female", "neutral"}:
        raise ValueError(f"Unsupported body gender: {gender}")
    root = Path(model_path).resolve()
    name = f"{model_type.upper()}_{gender.upper()}"
    extensions = ("npz",) if model_type == "smplx" else ("pkl", "npz")
    candidates = ([root] if root.is_file() else
                  [folder / f"{name}.{ext}" for folder in (root / model_type, root) for ext in extensions])
    asset = next((p for p in candidates if p.is_file()), None)
    if asset is None:
        raise FileNotFoundError(f"Missing licensed {model_type} asset: {candidates[0]}. "
                                "Use the dataset's official model; no fallback to another body model.")
    return asset


def _layer(model_path, gender, batch_size=1, *, model_type):
    model_type, count = body_model_settings(model_type=model_type)
    gender = str(gender).lower()
    asset = model_asset_path(model_path, gender, model_type=model_type)
    stat = asset.stat()
    key = (str(asset), model_type, gender, count, stat.st_size, stat.st_mtime_ns)
    if key not in _LAYERS:
        import smplx
        layer = smplx.build_layer(str(asset), model_type=model_type, gender=gender,
                                 ext=asset.suffix.lstrip("."), use_pca=False,
                                 flat_hand_mean=True, num_betas=count)
        if layer.num_betas != count:
            raise ValueError(f"{asset} supplies {layer.num_betas} betas; {model_type} requires {count}")
        if tuple(layer.parents[:22].tolist()) != SMPL_PARENTS:
            raise ValueError("Body-model parents do not match the HOI 22-joint skeleton")
        _LAYERS[key] = layer
    # Layer models infer batch size at forward time; do not duplicate large assets per clip length.
    return _LAYERS[key]


def _identity_pose(count, joints):
    return torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(count, joints, 1, 1)


def _offset_basis(model):
    regressor = model.J_regressor.detach().cpu().numpy()[:22].astype(np.float64)
    joints = regressor @ model.v_template.detach().cpu().numpy().astype(np.float64)
    basis = np.einsum("jv,vck->jck", regressor, model.shapedirs.detach().cpu().numpy().astype(np.float64))
    parents = np.asarray(SMPL_PARENTS[1:])
    return joints[1:] - joints[parents], basis[1:] - basis[parents]


def _validate_offsets(rest_offsets):
    offsets = np.asarray(rest_offsets, dtype=np.float64)
    if offsets.shape != (22, 3) or not np.isfinite(offsets).all():
        raise ValueError("rest_offsets must be finite [22,3]")
    return offsets


def _check_shape(model, rest_offsets, betas, tolerance, *, strict=True):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("shape_tolerance_m must be positive")
    base, basis = _offset_basis(model)
    error = np.linalg.norm(base + np.einsum("jck,k->jc", basis, betas) - rest_offsets[1:], axis=-1)
    if error.max() > tolerance:
        message = (f"Body model/shape does not match rest_offsets: max bone offset error "
                   f"{error.max() * 1000:.3f} mm > {tolerance * 1000:.3f} mm. "
                   "Check model type, gender, original betas and asset version.")
        if strict:
            raise ValueError(message)
        warnings.warn(message + " Preserving supplied betas without refitting; "
                      "use strict_shape=True to reject this residual.", UserWarning, stacklevel=3)


def fit_betas(rest_offsets, gender, model_path=DEFAULT_MODEL_PATH, *, dataset=None,
              model_type=None, shape_tolerance_m=.005) -> np.ndarray:
    """Fallback for legacy data: solve the linear rest-joint shape system exactly.

    Prefer original dataset betas. This fit cannot recreate body surface details
    absent from the skeleton, and rejects skeletons outside the model's shape space.
    """
    model_type, count = body_model_settings(dataset, model_type)
    offsets = _validate_offsets(rest_offsets)
    model = _layer(model_path, gender, model_type=model_type)
    base, basis = _offset_basis(model)
    betas = np.linalg.lstsq(basis.reshape(-1, count), (offsets[1:] - base).reshape(-1), rcond=None)[0]
    _check_shape(model, offsets, betas, shape_tolerance_m)
    return betas.astype(np.float32)


def local_rotations(global_rotations: np.ndarray) -> np.ndarray:
    """[T,22,3,3] globals -> locals, preserving the world-space root rotation."""
    local = np.empty_like(global_rotations)
    local[:, 0] = global_rotations[:, 0]
    for joint in range(1, 22):
        parent = global_rotations[:, SMPL_PARENTS[joint]]
        local[:, joint] = parent.transpose(0, 2, 1) @ global_rotations[:, joint]
    return local


def body_skinning_weights(weights, parents):
    """Fold omitted joints into their nearest retained body ancestor.

    This sums all finger weights into the corresponding wrist, and SMPL-X
    jaw/eye weights into the head. It works with both 52- and 55-joint trees.
    """
    weights, parents = np.asarray(weights), np.asarray(parents)
    if weights.ndim != 2 or weights.shape[1] != len(parents) or len(parents) < 22:
        raise ValueError("Skinning weights and parent tree disagree")
    if tuple(parents[:22]) != SMPL_PARENTS:
        raise ValueError("Expected the HOI 22-joint body hierarchy")
    ancestors = np.arange(len(parents))
    for joint in range(22, len(parents)):
        parent = int(parents[joint])
        if not 0 <= parent < joint:
            raise ValueError("Expected topologically ordered skinning parents")
        ancestors[joint] = ancestors[parent]
    result = np.zeros((len(weights), 22), dtype=weights.dtype)
    for joint, ancestor in enumerate(ancestors):
        result[:, ancestor] += weights[:, joint]
    return result


def skeleton_to_smpl(human, rest_offsets, gender, model_path=DEFAULT_MODEL_PATH, betas=None,
                     *, dataset=None, model_type=None, shape_tolerance_m=.005, strict_shape=False):
    """Return vertices[T,V,3], faces[F,3], dominant body joint[V].

    Specify dataset (preferred) or model_type; never infer it from joint count.
    The pelvis is aligned in world space; no extra world-axis transform is needed.
    Supplied source betas are preserved; residuals above tolerance warn (or raise
    with strict_shape). Fallback fitted betas must always meet the tolerance.
    """
    model_type, count = body_model_settings(dataset, model_type)
    human = np.asarray(human, np.float32)
    if human.ndim != 3 or human.shape[1:] != (22, 9) or not len(human) or not np.isfinite(human).all():
        raise ValueError("human must be finite [T,22,9], T >= 1")
    offsets = _validate_offsets(rest_offsets)
    if betas is None:
        betas = fit_betas(offsets, gender, model_path, model_type=model_type,
                          shape_tolerance_m=shape_tolerance_m)
    betas = np.asarray(betas, np.float32)
    if betas.shape not in {(count,), (1, count)} or not np.isfinite(betas).all():
        raise ValueError(f"{model_type} requires finite betas[{count}]")
    betas = betas.reshape(count)
    model = _layer(model_path, gender, model_type=model_type)
    _check_shape(model, offsets, betas, shape_tolerance_m, strict=strict_shape)
    globals_ = rotation6d_to_matrix(torch.from_numpy(human[..., 3:])).numpy()
    local = local_rotations(globals_).astype(np.float32)
    vertices = []
    with torch.no_grad():
        for start in range(0, len(human), 128):
            part = local[start:start + 128]
            frames = len(part)
            output = model(betas=torch.from_numpy(betas).expand(frames, -1).contiguous(),
                           transl=torch.zeros(frames, 3), global_orient=torch.from_numpy(part[:, :1]),
                           body_pose=torch.from_numpy(part[:, 1:]),
                           left_hand_pose=_identity_pose(frames, 15), right_hand_pose=_identity_pose(frames, 15),
                           pose2rot=False)
            # SMPL-X Layer defaults jaw/eyes to identity and expression to zero.
            shift = human[start:start + frames, 0, :3] - output.joints[:, 0].cpu().numpy()
            vertices.append(output.vertices.cpu().numpy() + shift[:, None, :])
    weights = body_skinning_weights(model.lbs_weights.cpu().numpy(), model.parents.cpu().numpy())
    return np.concatenate(vertices).astype(np.float64), np.asarray(model.faces, np.int64), weights.argmax(1)


def record_to_smpl(record, model_path=DEFAULT_MODEL_PATH):
    """Decode a processed/prediction archive using its saved body provenance."""
    metadata = body_metadata(record)
    if "body_geometry_version" in record:
        from .body_geometry import validate_body_geometry
        validate_body_geometry(record, model_path=model_path)
    if "gender" not in metadata:
        raise ValueError("Missing gender; backfill original body metadata with scripts/prepare_body_metadata.py")
    return skeleton_to_smpl(record["human"], record["rest_offsets"], str(metadata["gender"]), model_path,
                            betas=metadata.get("betas"), dataset=metadata.get("body_dataset"),
                            model_type=str(metadata["body_model_type"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as archive:
        record = dict(archive)
    vertices, faces, dominant = record_to_smpl(record, args.models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, vertices=vertices, faces=faces, dominant=dominant,
                        **body_metadata(record), **{k: record[k] for k in ("fps", "timestamps") if k in record})
    print(args.output.resolve())


if __name__ == "__main__":
    main()
