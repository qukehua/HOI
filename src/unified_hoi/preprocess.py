"""Convert official raw BEHAVE / OMOMO sequences, without inventing dynamics.

Pickle/joblib inputs must be trusted official/local dataset files. They are not a
safe interchange format; the output NPZ is loaded with allow_pickle=False.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .data import validate_record


PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                    9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
Z_UP_TO_Y_UP = Rotation.from_euler("x", -90, degrees=True).as_matrix()
# BEHAVE fits are in the Kinect reference frame (+Y down), not SMPL's rest frame.
BEHAVE_TO_Y_UP = Rotation.from_euler("x", 180, degrees=True).as_matrix()


class UnsupportedSequence(ValueError):
    """A sequence is explicitly outside the single rigid-object data contract."""


def _rot6(matrix):
    return np.asarray(matrix)[..., :2, :].reshape(*np.asarray(matrix).shape[:-2], 6)


def _fk(local_rotations, root_positions, offsets):
    n = len(local_rotations)
    positions = np.empty((n, 22, 3), dtype=np.float64)
    rotations = np.empty((n, 22, 3, 3), dtype=np.float64)
    positions[:, 0], rotations[:, 0] = root_positions, local_rotations[:, 0]
    for joint in range(1, 22):
        parent = PARENTS[joint]
        rotations[:, joint] = rotations[:, parent] @ local_rotations[:, joint]
        positions[:, joint] = positions[:, parent] + np.einsum("tij,j->ti", rotations[:, parent], offsets[joint])
    return positions, rotations


def surface_points(mesh_path, count=1024, seed=0, center=False, return_radius=False):
    """Deterministic, area-weighted sampling; preserves the annotation mesh origin."""
    import trimesh
    mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"empty object mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    if center:
        vertices -= vertices.mean(axis=0)
    triangles = vertices[np.asarray(mesh.faces)]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=-1)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("object mesh has no finite positive-area triangles")
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(triangles), size=count, p=areas / areas.sum())]
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    points = chosen[:, 0] + uv[:, :1] * (chosen[:, 1] - chosen[:, 0]) + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])
    return (points, float(np.linalg.norm(vertices, axis=-1).max())) if return_radius else points


def make_record(*, positions, rotations, object_trans, object_rot, object_points,
                rest_offsets, timestamps, fps, sequence_id, dataset, subject_id,
                object_id, text="", source_to_world=None, contact_threshold=0.05):
    """Convert explicit geometric arrays with one fixed world-frame transform.

    Local object points and skeleton offsets are NOT rotated with world axes.
    Global rotations are left-multiplied, so R' @ v_local remains correct.
    """
    transform = np.eye(3) if source_to_world is None else np.asarray(source_to_world)
    if transform.shape != (3, 3) or not np.allclose(transform.T @ transform, np.eye(3)) or not np.isclose(np.linalg.det(transform), 1):
        raise ValueError("source_to_world must be a proper fixed rotation")
    positions = np.asarray(positions) @ transform.T
    rotations = transform @ np.asarray(rotations)
    object_trans = np.asarray(object_trans).reshape(-1, 3) @ transform.T
    object_rot = transform @ np.asarray(object_rot).reshape(-1, 3, 3)
    object_points = np.asarray(object_points, dtype=np.float64)
    if not np.isfinite(contact_threshold) or contact_threshold <= 0:
        raise ValueError("contact_threshold must be positive metres")
    # Distance to sampled surface; this is deliberately NOT a force/contact truth label.
    joint_local = np.einsum("tji,tkj->tki", object_rot, positions - object_trans[:, None, :])
    distances = cKDTree(object_points).query(joint_local.reshape(-1, 3))[0].reshape(-1, 22)
    result = {
        "human": np.concatenate((positions, _rot6(rotations)), axis=-1).astype(np.float32),
        "object": np.concatenate((object_trans, _rot6(object_rot)), axis=-1).astype(np.float32),
        "contact": (distances <= contact_threshold).astype(np.float32),
        "object_points": object_points.astype(np.float32),
        "rest_offsets": np.asarray(rest_offsets, dtype=np.float32),
        "timestamps": np.asarray(timestamps, dtype=np.float64), "fps": np.float64(fps),
        "sequence_id": str(sequence_id), "dataset": str(dataset),
        "subject_id": str(subject_id), "object_id": str(object_id), "text": str(text),
        "world_up": "Y", "units": "metres", "source_to_world": transform.astype(np.float32),
        "contact_definition": "joint_to_sampled_surface_distance_proxy",
        "contact_threshold": np.float32(contact_threshold),
    }
    # Time gaps are handled before serialization in continuous_records().
    return result


def convert_omomo_record(record, object_points, *, source_fps, text="", contact_threshold=0.05,
                         scale_relative_tolerance=0.01, object_radius=None,
                         fps_provenance="explicit_source_fps_not_verified_by_metadata"):
    """Decode OMOMO *sequence* dictionaries, not canonicalized window dictionaries.

    Follows official hand_foot_dataset.py: root joint = trans - trans2joint,
    and v_world = obj_scale * obj_rot @ v_rest + obj_trans.
    """
    name = str(record["seq_name"])
    fields = name.split("_")
    if len(fields) < 2:
        raise ValueError("OMOMO seq_name must include subject and object")
    subject, obj = fields[:2]
    if obj in {"mop", "vacuum"} or any(k.startswith("obj_bottom_") for k in record):
        raise UnsupportedSequence(f"{name}: articulated top/bottom object is unsupported")
    n = len(record["trans"])
    scale = np.asarray(record["obj_scale"], dtype=np.float64).reshape(-1)
    if scale.size not in {1, n} or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("OMOMO obj_scale must be positive scalar or [T]")
    if not np.isfinite(scale_relative_tolerance) or scale_relative_tolerance < 0:
        raise ValueError("scale_relative_tolerance must be finite and nonnegative")
    fixed_scale = float(np.median(scale))
    scale_spread = float(np.ptp(scale) / fixed_scale)
    if scale_spread > scale_relative_tolerance + 1e-12:
        raise UnsupportedSequence(f"{name}: time-varying object scale relative range {scale_spread:.6g} "
                                  f"exceeds tolerance {scale_relative_tolerance:.6g}")
    sampled_radius = float(np.linalg.norm(object_points, axis=-1).max())
    radius = sampled_radius if object_radius is None else float(object_radius)
    if not np.isfinite(radius) or radius < sampled_radius - 1e-6:
        raise ValueError("object_radius must bound all supplied object points")
    displacement = float(np.abs(scale - fixed_scale).max() * radius)
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("explicit source_fps is required; OMOMO releases differ")
    if "fps" in record and not np.isclose(float(np.asarray(record["fps"]).item()), source_fps):
        raise ValueError("source_fps conflicts with OMOMO record fps")
    offsets = np.asarray(record["rest_offsets"], dtype=np.float64)[:22].copy()
    if offsets.shape != (22, 3):
        raise ValueError("OMOMO rest_offsets must include the SMPL22 skeleton")
    offsets[0] = 0
    angles = np.concatenate((np.asarray(record["root_orient"]).reshape(n, 1, 3),
                             np.asarray(record["pose_body"]).reshape(n, 21, 3)), axis=1)
    local_rot = Rotation.from_rotvec(angles.reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3)
    root = np.asarray(record["trans"]).reshape(n, 3) - np.asarray(record["trans2joint"]).reshape(1, 3)
    positions, rotations = _fk(local_rot, root, offsets)
    timestamps = record.get("timestamps", np.arange(n, dtype=np.float64) / source_fps)
    result = make_record(positions=positions, rotations=rotations,
                         object_trans=record["obj_trans"], object_rot=record["obj_rot"],
                         object_points=np.asarray(object_points) * fixed_scale, rest_offsets=offsets,
                         timestamps=timestamps, fps=source_fps, sequence_id=name, dataset="omomo",
                         subject_id=subject, object_id=obj, text=text,
                         source_to_world=Z_UP_TO_Y_UP, contact_threshold=contact_threshold)
    result["source_object_scale"] = np.float64(fixed_scale)
    result["source_object_scale_range"] = np.array([scale.min(), scale.max()], dtype=np.float64)
    result["scale_relative_range"] = np.float64(scale_spread)
    result["scale_relative_tolerance"] = np.float64(scale_relative_tolerance)
    result["scale_max_surface_displacement_m"] = np.float64(displacement)
    result["scale_displacement_domain"] = "mesh_surface_bound" if object_radius is not None else "sampled_surface"
    result["scale_reduction"] = "sequence_median"
    result["timestamps_origin"] = "source" if "timestamps" in record else "explicit_source_fps"
    result["fps_provenance"] = "record_fps" if "fps" in record else str(fps_provenance)
    result["fps_verified_by_metadata"] = np.bool_("fps" in record or "timestamps" in record)
    return result


def _seconds(values):
    values = np.asarray(values)
    if values.dtype.kind in "SUO":
        return np.array([float(str(v.decode() if isinstance(v, bytes) else v).removeprefix("t"))
                         for v in values], dtype=np.float64)
    return values.astype(np.float64)


def convert_behave_record(smpl, obj, object_points, *, sequence_id, subject_id,
                          object_id, gender, body_model, source_fps=30, text="",
                          contact_threshold=0.05, source_to_world=BEHAVE_TO_Y_UP):
    """Decode full-rate BEHAVE parameter archives. body_model is a SMPL-H layer.

    object_points must come from the vertex-mean-centered official object template.
    The template's origin convention is crucial for the annotated object translation.
    """
    import torch
    hs, os = _seconds(smpl["frame_times"]), _seconds(obj["frame_times"])
    if len(np.unique(hs)) != len(hs) or len(np.unique(os)) != len(os):
        raise ValueError("duplicate BEHAVE frame timestamps")
    timestamps, hi, oi = np.intersect1d(hs, os, return_indices=True)
    if not len(timestamps):
        raise ValueError("no synchronized human/object frames")
    n = len(timestamps)
    pose = np.asarray(smpl["poses"])[hi].reshape(n, 52, 3)
    betas = np.asarray(smpl["betas"])
    if betas.ndim == 1:
        betas = np.repeat(betas[None], n, axis=0)
    elif len(betas) == 1:
        betas = np.repeat(betas, n, axis=0)
    else:
        betas = betas[hi]
    if not np.allclose(betas, betas[:1], atol=1e-4, rtol=1e-4):
        raise UnsupportedSequence(f"{sequence_id}: time-varying body shape requires an explicit reduction policy")
    rot = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(n, 52, 3, 3)
    params = {"betas": torch.from_numpy(betas.astype(np.float32)),
              "transl": torch.from_numpy(np.asarray(smpl["trans"])[hi].astype(np.float32)),
              "global_orient": torch.from_numpy(rot[:, :1].astype(np.float32)),
              "body_pose": torch.from_numpy(rot[:, 1:22].astype(np.float32)),
              "left_hand_pose": torch.from_numpy(rot[:, 22:37].astype(np.float32)),
              "right_hand_pose": torch.from_numpy(rot[:, 37:].astype(np.float32))}
    # Bounded batches avoid allocating all full-resolution human meshes at once.
    joints = []
    with torch.no_grad():
        for start in range(0, n, 256):
            prediction = body_model(**{k: v[start:start + 256] for k, v in params.items()}, pose2rot=False)
            joints.append(prediction.joints[:, :22].detach().cpu().numpy())
        identity = torch.eye(3)[None, None]
        rest = body_model(betas=params["betas"][:1], transl=torch.zeros(1, 3),
                          global_orient=identity, body_pose=identity.repeat(1, 21, 1, 1),
                          left_hand_pose=identity.repeat(1, 15, 1, 1),
                          right_hand_pose=identity.repeat(1, 15, 1, 1), pose2rot=False)
    rest_joints = rest.joints[0, :22].detach().cpu().numpy()
    offsets = np.zeros((22, 3))
    offsets[1:] = rest_joints[1:] - rest_joints[PARENTS[1:]]
    _, global_rot = _fk(rot[:, :22], np.zeros((n, 3)), offsets)
    object_rot = Rotation.from_rotvec(np.asarray(obj["angles"])[oi].reshape(n, 3)).as_matrix()
    result = make_record(positions=np.concatenate(joints), rotations=global_rot,
                         object_trans=np.asarray(obj["trans"])[oi], object_rot=object_rot,
                         object_points=object_points, rest_offsets=offsets, timestamps=timestamps,
                         fps=source_fps, sequence_id=sequence_id, dataset="behave", subject_id=subject_id,
                         object_id=object_id, text=text, source_to_world=source_to_world,
                         contact_threshold=contact_threshold)
    result["gender"] = str(gender)
    result["timestamps_origin"] = "frame_times"
    result["fps_verified_by_metadata"] = True
    result["fps_provenance"] = "BEHAVE frame_times, checked against explicit source fps"
    return result


def continuous_records(record, target_fps=None):
    """Split holes BEFORE stride downsampling. Never bridge a missing interval."""
    stamps = np.asarray(record["timestamps"], dtype=np.float64)
    fps = float(record["fps"])
    if len(stamps) != len(record["human"]) or not np.isfinite(stamps).all():
        raise ValueError("invalid timestamps")
    if len(stamps) > 1 and not np.all(np.diff(stamps) > 0):
        raise ValueError("timestamps must strictly increase")
    target_fps = fps if target_fps is None else float(target_fps)
    ratio = fps / target_fps if target_fps > 0 else 0
    if ratio < 1 or not np.isclose(ratio, round(ratio)):
        raise ValueError("target_fps must divide source_fps; upsampling is not allowed")
    step = int(round(ratio))
    tolerance = min(0.0021, 0.1 / fps)
    if len(stamps) > 1 and not np.any(np.abs(np.diff(stamps) - 1 / fps) <= tolerance):
        raise ValueError("no intervals match source_fps; low-rate annotations cannot be relabeled as high-rate motion")
    boundaries = np.r_[0, np.flatnonzero(np.abs(np.diff(stamps) - 1 / fps) > tolerance) + 1, len(stamps)]
    for part, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if end <= start:
            continue
        result = dict(record)
        for key in ("human", "object", "contact", "timestamps"):
            result[key] = np.asarray(record[key])[start:end:step]
        result["fps"] = np.float64(target_fps)
        result["source_sequence_id"] = str(record["sequence_id"])
        if len(boundaries) > 2:
            result["sequence_id"] = f"{record['sequence_id']}__part{part:04d}"
        validate_record(result)
        yield result


def _split_mapping(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if all(key in {"train", "val", "test"} for key in data):
        result = {}
        for split, names in data.items():
            for name in names:
                if name in result:
                    raise ValueError(f"duplicate sequence in split file: {name}")
                result[name] = split
        return result
    if not all(value in {"train", "val", "test"} for value in data.values()):
        raise ValueError("split file must map sequence IDs to train/val/test")
    return data


def _official_omomo_split(name):
    match = re.fullmatch(r"sub(\d+)", name.split("_")[0])
    if not match or not 1 <= int(match[1]) <= 17:
        raise ValueError(f"unknown OMOMO subject: {name}")
    return "train" if int(match[1]) <= 15 else "test"


def _training_holdout(name, split, fraction, seed):
    """Stable sequence-level holdout: independent of iteration order or --limit."""
    if split != "train" or fraction <= 0:
        return split
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if value < fraction else "train"


def _mesh_for_behave(root, obj):
    suffix = {"chairblack": "_f2500", "chairwood": "_f2500", "tablesquare": "_f2000",
              "monitor": "_closed_f1000"}.get(obj, "_f1000")
    return Path(root) / obj / f"{obj}{suffix}.ply"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="dataset", required=True)
    for dataset in ("behave", "omomo"):
        command = sub.add_parser(dataset)
        command.add_argument("--input", type=Path, required=True,
                             help="BEHAVE sequences directory / OMOMO official raw sequence .p file")
        command.add_argument("--objects", type=Path, required=True, help="official object mesh directory")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--source-fps", type=float, default=30 if dataset == "behave" else None,
                             required=dataset == "omomo", help="actual annotation fps; never inferred from video")
        command.add_argument("--target-fps", type=float, help="optional integer-stride downsample, never upsample")
        command.add_argument("--points", type=int, default=1024)
        command.add_argument("--contact-threshold", type=float, default=0.05, help="joint/surface proxy distance in metres")
        command.add_argument("--split", choices=["train", "val", "test"], help="explicit assignment of all input sequences")
        command.add_argument("--split-file", type=Path, help="official split.json or explicit sequence mapping")
        command.add_argument("--val-from-train", type=float, default=0,
                             help="stable sequence-level validation fraction from training only; test stays test")
        command.add_argument("--split-seed", type=int, default=0)
        command.add_argument("--text-file", type=Path, help="verified sequence-ID to annotation-text JSON; absent means empty text")
        command.add_argument("--overwrite", action="store_true")
        command.add_argument("--limit", type=int, help="inspect only the first N source sequences, for a smoke conversion")
        if dataset == "behave":
            command.add_argument("--smpl-models", type=Path, required=True, help="licensed SMPL-H model directory")
        else:
            command.add_argument("--scale-relative-tolerance", type=float, default=0.01,
                                 help="max (scale.max-scale.min)/median; use 0 to require exact rigidity")
            command.add_argument("--fps-provenance", default="explicit_source_fps_not_verified_by_metadata",
                                 help="evidence or assumption supporting --source-fps, saved verbatim")
    args = parser.parse_args(argv)
    if args.points <= 0:
        parser.error("--points must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0 <= args.val_from_train < 1:
        parser.error("--val-from-train must be in [0, 1)")
    if args.dataset == "omomo" and (not np.isfinite(args.scale_relative_tolerance) or args.scale_relative_tolerance < 0):
        parser.error("--scale-relative-tolerance must be finite and nonnegative")
    if args.split and args.split_file:
        parser.error("choose --split OR --split-file")
    if args.dataset == "behave" and not (args.split or args.split_file):
        parser.error("BEHAVE requires --split-file (official split) or explicit --split")
    manifest_path = args.output / "manifest.jsonl"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; use --overwrite deliberately")
    splits = _split_mapping(args.split_file)
    texts = json.loads(args.text_file.read_text(encoding="utf-8")) if args.text_file else {}
    args.output.mkdir(parents=True, exist_ok=True)
    archive_dir = args.output / "sequences"
    archive_dir.mkdir(exist_ok=True)
    entries, rejected = [], []
    mesh_cache, body_cache = {}, {}
    if args.dataset == "omomo":
        import joblib
        source = joblib.load(args.input)
        if not isinstance(source, dict):
            raise ValueError("expected official OMOMO sequence dictionary")
        items = [(str(rec["seq_name"]), rec) for rec in source.values()]
    else:
        items = [(p.name, p) for p in sorted(args.input.glob("*")) if p.is_dir() and (p / "info.json").exists()]
    total_sequences = len(items)
    if args.limit is not None:
        items = items[:args.limit]
    seen = set()
    for item_index, (name, source) in enumerate(items):
        if name in seen:
            raise ValueError(f"duplicate source sequence ID: {name}")
        seen.add(name)
        try:
            if args.split_file and name not in splits:
                raise UnsupportedSequence("not listed in requested split mapping")
            split = args.split or splits.get(name) or _official_omomo_split(name)
            split = _training_holdout(f"{args.dataset}:{name}", split, args.val_from_train, args.split_seed)
            text = texts.get(name, "")
            if not isinstance(text, str):
                raise ValueError("verified text must be a string")
            if args.dataset == "omomo":
                obj = name.split("_")[1]
                if obj in {"mop", "vacuum"}:
                    raise UnsupportedSequence("articulated object: mop/vacuum")
                if obj not in mesh_cache:
                    mesh_cache[obj] = surface_points(args.objects / f"{obj}_cleaned_simplified.obj", args.points,
                                                    return_radius=True)
                points, radius = mesh_cache[obj]
                record = convert_omomo_record(source, points, source_fps=args.source_fps,
                                             text=text, contact_threshold=args.contact_threshold,
                                             scale_relative_tolerance=args.scale_relative_tolerance,
                                             object_radius=radius, fps_provenance=args.fps_provenance)
            else:
                import smplx
                info = json.loads((source / "info.json").read_text(encoding="utf-8"))
                obj, gender = info["cat"], info["gender"]
                if obj not in mesh_cache:
                    mesh_cache[obj] = surface_points(_mesh_for_behave(args.objects, obj), args.points, center=True)
                if gender not in body_cache:
                    if not args.smpl_models.exists():
                        raise FileNotFoundError(f"Licensed SMPL-H assets are required; --smpl-models does not exist: {args.smpl_models}")
                    body_cache[gender] = smplx.build_layer(str(args.smpl_models), model_type="smplh",
                                                         gender=gender, num_betas=10, use_pca=False)
                with np.load(source / "smpl_fit_all.npz", allow_pickle=False) as human_archive:
                    smpl = {k: human_archive[k] for k in ("poses", "betas", "trans", "frame_times")}
                with np.load(source / "object_fit_all.npz", allow_pickle=False) as object_archive:
                    obj_params = {k: object_archive[k] for k in ("angles", "trans", "frame_times")}
                record = convert_behave_record(smpl, obj_params, mesh_cache[obj], sequence_id=name,
                                              subject_id=name.split("_")[1], object_id=obj, gender=gender,
                                              body_model=body_cache[gender], source_fps=args.source_fps,
                                              text=text, contact_threshold=args.contact_threshold)
            for part in continuous_records(record, args.target_fps):
                safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", part["sequence_id"])
                dest = archive_dir / f"{args.dataset}_{safe_name}.npz"
                if dest.exists() and not args.overwrite:
                    raise FileExistsError(f"{dest} already exists")
                np.savez_compressed(dest, **part)
                entry = {"path": dest.relative_to(args.output).as_posix(), "dataset": args.dataset,
                                "sequence_id": part["sequence_id"], "source_sequence_id": name,
                                "subject_id": part["subject_id"], "object_id": part["object_id"],
                                "split": split, "fps": float(part["fps"]), "frames": len(part["human"]),
                                "text": part["text"], "source_path": str(args.input.resolve())}
                if args.dataset == "omomo":
                    entry.update({key: float(part[key]) for key in
                                  ("source_object_scale", "scale_relative_range", "scale_max_surface_displacement_m")})
                    entry["fps_provenance"] = part["fps_provenance"]
                    entry["fps_verified_by_metadata"] = bool(part["fps_verified_by_metadata"])
                entries.append(entry)
        except (ValueError, OSError, KeyError, AssertionError) as exc:
            rejected.append({"sequence_id": name, "reason": str(exc), "error_type": type(exc).__name__})
        if (item_index + 1) % 250 == 0:
            print(f"Processed {item_index + 1}/{len(items)} sequences; wrote {len(entries)}, rejected {len(rejected)}", flush=True)
    manifest_path.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries), encoding="utf-8")
    report = {"dataset": args.dataset, "source_sequences_total": total_sequences,
              "source_sequences": len(items), "written_segments": len(entries),
              "rejected": rejected, "source_fps": args.source_fps, "target_fps": args.target_fps,
              "val_from_train": args.val_from_train, "split_seed": args.split_seed,
              "split_counts": {s: sum(e["split"] == s for e in entries) for s in ("train", "val", "test")},
              "contact_definition": "joint-to-sampled-surface proxy, not measured physical contacts"}
    if args.dataset == "omomo":
        report.update({"scale_relative_tolerance": args.scale_relative_tolerance,
                       "scale_reduction": "sequence_median_changes_annotation_geometry",
                       "max_scale_surface_displacement_m": max((e["scale_max_surface_displacement_m"] for e in entries), default=None),
                       "fps_provenance": args.fps_provenance,
                       "all_fps_verified_by_metadata": bool(entries) and all(e["fps_verified_by_metadata"] for e in entries)})
    (args.output / "conversion_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "segments": len(entries), "rejected": len(rejected)}))
    if not entries:
        raise SystemExit("No usable sequences converted; see conversion_report.json")


if __name__ == "__main__":
    main()
