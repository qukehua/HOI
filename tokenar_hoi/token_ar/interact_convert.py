"""Convert an InterAct sequence without rewriting human.npz or object.npz.

The converted record can stay in memory or be written as its own npz under
data/parametric_assets. Training then loads that file.
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..mesh_geometry import encode_mesh, load_mesh_asset
from ..smpl_mesh import DEFAULT_MODEL_PATH
from . import REPRESENTATION
from .body import BodyGeometry, digest_file, human_dim
from .contact_offset import pack_contact_offset
from ..body_metadata import DATASET_MODELS, MODEL_BETAS
from ..geometry import rotation6d_to_matrix
from ..smpl_mesh import _layer


def _text(value):
    return str(np.asarray(value).item()).strip()


def axis_angle_to_rot6d(axis_angle):
    """First two rows of the column-vector rotation matrix. Matches geometry.matrix_to_rotation6d."""
    matrix = Rotation.from_rotvec(np.asarray(axis_angle, np.float64).reshape(-1, 3)).as_matrix()
    return matrix[:, :2, :].reshape(axis_angle.shape[:-1] + (6,)).astype(np.float32)


@lru_cache(maxsize=4)
def _smplx_hand_pca(gender, models, n_comps=24):
    """SMPL-X PCA bases matching InterAct GRAB (flat_hand_mean=True, 24 components)."""
    import smplx
    from ..smpl_mesh import model_asset_path
    asset = model_asset_path(models, gender, model_type="smplx")
    model = smplx.create(str(asset), model_type="smplx", gender=gender, use_pca=True,
                         num_pca_comps=n_comps, flat_hand_mean=True, ext=asset.suffix.lstrip("."))
    left = model.left_hand_components.detach().cpu().numpy().astype(np.float64)
    right = model.right_hand_components.detach().cpu().numpy().astype(np.float64)
    return left, right


def expand_grab_poses(poses, gender, body_models="data/smplx_models"):
    """Expand GRAB (T,114) PCA hands into full (T,156) axis-angle poses."""
    poses = np.asarray(poses, np.float64)
    if poses.ndim != 2 or poses.shape[1] != 114:
        raise ValueError(f"GRAB poses must be (T,114), got {poses.shape}")
    left_basis, right_basis = _smplx_hand_pca(gender, str(Path(body_models).resolve()))
    left = poses[:, 66:90] @ left_basis
    right = poses[:, 90:114] @ right_basis
    return np.concatenate((poses[:, :66], left, right), 1).astype(np.float32)


def fit_betas_from_vtemp(vtemp, gender, body_models="data/smplx_models", count=16):
    """Least-squares shape coefficients that approximate a GRAB subject template."""
    layer = _layer(str(Path(body_models).resolve()), gender, model_type="smplx")
    template = layer.v_template.detach().cpu().numpy().astype(np.float64)
    shapedirs = layer.shapedirs.detach().cpu().numpy()[:, :, :count].astype(np.float64)
    delta = np.asarray(vtemp, np.float64) - template
    if delta.shape != template.shape:
        raise ValueError(f"vtemp shape {delta.shape} does not match SMPL-X template {template.shape}")
    betas, *_ = np.linalg.lstsq(shapedirs.reshape(-1, count), delta.reshape(-1), rcond=None)
    return betas.astype(np.float32)


def interact_files(path):
    """Return (human.npz, object.npz) when `path` is an untouched InterAct sequence."""
    path = Path(path)
    if path.is_dir():
        human, obj = path / "human.npz", path / "object.npz"
        return (human, obj) if human.is_file() and obj.is_file() else None
    if path.suffix != ".npz" or not path.is_file():
        return None
    if path.name == "human.npz" and (path.parent / "object.npz").is_file():
        with np.load(path, allow_pickle=True) as archive:
            interact = "poses" in archive.files and "human_params" not in archive.files
        return (path, path.parent / "object.npz") if interact else None
    return None


def _read_obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            corners = []
            for token in line.split()[1:]:
                index = int(token.split("/")[0])
                corners.append(index - 1 if index > 0 else index)
            for corner in range(1, len(corners) - 1):
                faces.append((corners[0], corners[corner], corners[corner + 1]))
    if len(vertices) < 3 or not faces:
        raise ValueError(f"Object mesh has no triangles: {path}")
    return np.asarray(vertices, np.float32), np.asarray(faces, np.int64)


def load_object_vertices(path):
    path = Path(path)
    if path.suffix == ".npy":
        points = np.load(path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected object points [V,3]: {path}")
        return np.asarray(points, np.float32), None
    if path.suffix == ".obj":
        return _read_obj(path)
    vertices, _ = load_mesh_asset(str(path))
    return np.asarray(vertices, np.float32), None


def resolve_object_mesh(sequence_dir, name, explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Object mesh does not exist: {path}")
        return path
    root = sequence_dir.parent.parent
    folders = [root / "objects" / name, root.parent / "behave_raw" / "objects" / name,
               root.parent / "omomo_raw" / "objects" / name,
               root.parent / "grab_raw" / "objects" / name,
               root.parent / "grab_correct" / "objects" / name]
    for folder in folders:
        for candidate in (folder / f"{name}.obj", folder / "sample_points.npy"):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No object mesh for {name} beside {sequence_dir}")


def surface_features(path, scale=1.):
    vertices, faces = load_object_vertices(path)
    if faces is None:
        raise ValueError(f"BPS needs a triangle mesh, not a point cloud: {path}")
    bps, points = encode_mesh(vertices, faces, scale)
    return bps, points


@lru_cache(maxsize=8)
def _geometry(dataset, support, models):
    return BodyGeometry(dataset, support, models).eval()


@lru_cache(maxsize=64)
def _object_tree(path, scale):
    vertices, _ = load_object_vertices(path)
    vertices = np.asarray(vertices, np.float64) * float(scale)
    return cKDTree(vertices), vertices.astype(np.float32)


def _contact_offsets(vertices, pose, mesh_path, scale):
    tree, points = _object_tree(str(Path(mesh_path).resolve()), float(scale))
    local = (vertices - pose[:, None, :3]) @ rotation6d_to_matrix(torch.as_tensor(pose[:, 3:])).numpy()
    flat = local.reshape(-1, 3)
    distance, index = tree.query(flat, workers=1)
    nearest = points[index].reshape(local.shape)
    return pack_contact_offset(torch.as_tensor(nearest, dtype=torch.float32),
                               torch.as_tensor(distance, dtype=torch.float32).reshape(local.shape[:-1])).numpy()


def _body_vertices(geometry, human, gender):
    pose = torch.as_tensor(human, dtype=torch.float32)
    flag = torch.tensor([int(gender == "male")])
    chunks = []
    for start in range(0, len(pose), 32):
        part = pose[start:start + 32][None]
        vertices, _ = geometry(part, flag)
        chunks.append(vertices[0].detach().cpu().numpy())
    return np.concatenate(chunks, 0)


def convert_interact(path, dataset, contact_support, *, fps=None, object_mesh=None, object_mesh_scale=1.,
                     body_models="data/smplx_models", sequence_id=None):
    """Build the in-memory parametric record for one InterAct sequence directory or human.npz."""
    files = interact_files(path)
    if files is None:
        raise ValueError(f"Not an InterAct sequence: {path}")
    if dataset not in DATASET_MODELS:
        raise ValueError("InterAct conversion needs dataset='behave', 'omomo', or 'grab'")
    if not contact_support:
        raise ValueError("InterAct conversion needs the 690-vertex contact_support file")
    human_path, object_path = files
    sequence_dir = human_path.parent
    body_models = str(Path(body_models).expanduser().resolve())
    if not (Path(body_models) / "smplx").is_dir() and not Path(body_models).is_file():
        raise FileNotFoundError(f"Body model root not found: {body_models}")
    with np.load(human_path, allow_pickle=True) as archive:
        poses = archive["poses"]
        trans = archive["trans"]
        gender = _text(archive["gender"])
        betas = archive["betas"] if "betas" in archive.files else None
        vtemp = archive["vtemp"] if "vtemp" in archive.files else None
    with np.load(object_path, allow_pickle=True) as archive:
        angles, obj_trans, name = archive["angles"], archive["trans"], _text(archive["name"])
    poses = np.asarray(poses, np.float32)
    trans = np.asarray(trans, np.float32)
    angles, obj_trans = np.asarray(angles, np.float32), np.asarray(obj_trans, np.float32)
    frames = len(poses)
    if gender not in ("male", "female"):
        raise ValueError(f"Unsupported gender in {sequence_dir}: {gender}")
    if dataset == "grab":
        if poses.shape != (frames, 114):
            raise ValueError(f"GRAB poses must be (T,114), got {poses.shape} in {sequence_dir}")
        if vtemp is None:
            raise ValueError(f"GRAB sequence needs vtemp: {sequence_dir}")
        poses = expand_grab_poses(poses, gender, body_models)
        betas = fit_betas_from_vtemp(vtemp, gender, body_models, MODEL_BETAS["smplx"])
    else:
        poses = np.asarray(poses, np.float32)
        if betas is None:
            raise ValueError(f"{dataset} sequence needs betas: {sequence_dir}")
        betas = np.asarray(betas, np.float32)
    if poses.shape != (frames, 156) or trans.shape != (frames, 3) or angles.shape != (frames, 3) or obj_trans.shape != (frames, 3):
        raise ValueError(f"Unexpected InterAct shapes in {sequence_dir}")
    expected_betas = (10 if DATASET_MODELS[dataset] == "smplh" else 16,)
    if betas.shape != expected_betas:
        raise ValueError(f"{dataset} betas must match the body model, got {betas.shape}")
    rot6 = axis_angle_to_rot6d(poses.reshape(frames, 52, 3)).reshape(frames, -1)
    repeated = np.repeat(betas.reshape(1, -1), frames, 0)
    human = np.concatenate((repeated, rot6, trans), -1).astype(np.float32)
    if human.shape[1] != human_dim(dataset):
        raise ValueError("Converted human dimension does not match the dataset")
    obj = np.concatenate((obj_trans, axis_angle_to_rot6d(angles)), -1).astype(np.float32)
    mesh = resolve_object_mesh(sequence_dir, name, object_mesh)
    rate = 30. if fps is None else float(fps)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("fps must be positive")
    geometry = _geometry(dataset, str(Path(contact_support).resolve()), body_models)
    offsets = _contact_offsets(_body_vertices(geometry, human, gender), obj, mesh, object_mesh_scale)
    text_path = sequence_dir / "text.txt"
    text = text_path.read_text(encoding="utf-8").strip() if text_path.is_file() else ""
    record = {"dataset": np.array(dataset), "representation": np.array(REPRESENTATION),
              "body_model_type": np.array(DATASET_MODELS[dataset]), "gender": np.array(gender),
              "human_params": human, "object": obj, "contact_offset": offsets.astype(np.float32),
              "timestamps": (np.arange(frames, dtype=np.float64) / rate).astype(np.float64), "fps": np.float64(rate),
              "support_sha256": np.array(digest_file(contact_support)),
              "body_model_sha256": np.array(geometry.layers[gender].asset_hash),
              "sequence_id": np.array(sequence_id or sequence_dir.name), "text": text,
              "object_mesh_path": str(mesh.resolve()), "object_mesh_scale": float(object_mesh_scale),
              "object_name": name}
    if dataset == "omomo":
        record["hand_source"] = np.array("interact_corrected")
    if dataset == "grab":
        record["hand_source"] = np.array("grab_pca24_expanded")
    return record


def _npz_value(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, str):
        return np.array(value)
    if isinstance(value, (float, np.floating)):
        return np.float64(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return np.int64(value)
    raise TypeError(f"Cannot store {type(value).__name__} in a parametric npz")


def save_parametric_record(path, record):
    """Write one converted record. The file is replaced only after the save finishes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing.npz")
    np.savez(temporary, **{key: _npz_value(value) for key, value in record.items()})
    temporary.replace(path)


def _export_job(job):
    torch.set_num_threads(2)
    sequence_dir, dataset, contact_support, dest, fps, sequence_id, body_models = job
    dest = Path(dest)
    if dest.is_file():
        return sequence_id, "exists"
    record = convert_interact(sequence_dir, dataset, contact_support, fps=fps,
                              body_models=body_models, sequence_id=sequence_id)
    save_parametric_record(dest, record)
    return sequence_id, "wrote"


def export_manifest(manifest, output_dir, *, workers=16, body_models="data/smplx_models"):
    """Convert every manifest sequence into output_dir, then point the manifest at those files."""
    manifest, output_dir = Path(manifest), Path(output_dir)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {manifest}")
    dataset = rows[0]["dataset"]
    if any(row["dataset"] != dataset for row in rows):
        raise ValueError("A parametric export manifest must contain one dataset")
    body_models = str(Path(body_models).resolve())
    contact_support = str(Path(f"data/parametric_assets/{dataset}/contact_support.npz").resolve())
    if not Path(contact_support).is_file():
        raise FileNotFoundError(contact_support)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in rows:
        sequence_id = row["sequence_id"]
        jobs.append((row["path"], dataset, contact_support, str(output_dir / f"{sequence_id}.npz"),
                     row.get("fps", 30), sequence_id, body_models))
    done = 0
    import os
    from multiprocessing import get_context
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    with get_context("spawn").Pool(workers) as pool:
        for sequence_id, status in pool.imap_unordered(_export_job, jobs):
            done += 1
            if done == 1 or done % 25 == 0 or done == len(jobs):
                print(f"export {done}/{len(jobs)} {status} {sequence_id}", flush=True)
    missing = [row["sequence_id"] for row in rows if not (output_dir / f"{row['sequence_id']}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} sequences were not written, first: {missing[0]}")
    for row in rows:
        row["path"] = str((output_dir / f"{row['sequence_id']}.npz").resolve())
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {"manifest": str(manifest), "sequences": len(rows), "output": str(output_dir.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/behave_correct.jsonl")
    parser.add_argument("--output", default="data/parametric_assets/behave/sequences")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--body-models", default="data/smplx_models")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    print(export_manifest(args.manifest, args.output, workers=args.workers, body_models=args.body_models))


if __name__ == "__main__":
    main()
