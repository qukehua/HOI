"""Generate a sequence from scene geometry and explicit observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .controls import (ControlBatch, check_anchor_feasibility, compile_controls, describe_mode,
                       parse_mode, sample_controls)
from .data import load_record
from .body_metadata import BODY_METADATA_KEYS, body_metadata
from .body_geometry import BODY_GEOMETRY_KEYS, canonicalize_body
from .mesh_geometry import attach_mesh_geometry
from .runtime import device_for, load_model, move_batch, write_json


def prepare_batch(path, frames=None, start=0, text_dim=512, scene_only=False, *,
                  geometry_encoder="pointnet", object_mesh=None, object_mesh_scale=1.):
    if frames is not None and frames < 1:
        raise ValueError("frames must be positive")
    if scene_only:
        with np.load(path, allow_pickle=False) as archive:
            record = {key: archive[key] for key in archive.files}
        if not frames or frames < 1 or start:
            raise ValueError("Scene-only generation requires --frames > 0 and start=0")
        n = frames
        fps = float(record["fps"])
        timestamps = np.arange(n, dtype=np.float64) / fps
        state = {"human": np.zeros((n, 22, 9), np.float32),
                 "object": np.zeros((n, 9), np.float32), "contact": np.zeros((n, 22), np.float32)}
    else:
        record = load_record(path)
        if geometry_encoder == "mesh_bps":
            record = attach_mesh_geometry(record, object_mesh, object_mesh_scale)
        if "gender" in record and "betas" in record:
            record = canonicalize_body(record)
        n = min(frames or len(record["human"]), len(record["human"]) - start)
        if start < 0 or n < 1:
            raise ValueError("Requested clip is outside the reference")
        fps = float(record["fps"])
        timestamps = record["timestamps"][start:start + n]
        state = {key: record[key][start:start + n] for key in ("human", "object", "contact")}
    if scene_only and geometry_encoder == "mesh_bps":
        record = attach_mesh_geometry(record, object_mesh, object_mesh_scale)
    if scene_only and "gender" in record and "betas" in record:
        record = canonicalize_body(record)
    if geometry_encoder not in {"pointnet", "mesh_bps"}:
        raise ValueError("geometry_encoder must be mesh_bps or pointnet")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    batch = {key: torch.as_tensor(np.asarray(value, np.float32).copy())[None] for key, value in state.items()}
    for key in ("object_points", "rest_offsets"):
        batch[key] = torch.as_tensor(np.asarray(record[key], np.float32).copy())[None]
        if not torch.isfinite(batch[key]).all():
            raise ValueError(f"Non-finite scene {key}")
    if batch["object_points"].ndim != 3 or batch["object_points"].shape[-1] != 3:
        raise ValueError("object_points must be [K,3]")
    if batch["rest_offsets"].shape != (1, 22, 3):
        raise ValueError("rest_offsets must be [22,3]")
    batch.update(valid_frames=torch.ones((1, n), dtype=torch.bool),
                 fps=torch.tensor([fps], dtype=torch.float32),
                 timestamps=torch.as_tensor(timestamps, dtype=torch.float64)[None],
                 fps_verified_by_metadata=torch.tensor([bool(record.get("fps_verified_by_metadata", False))]),
                 floor_y=torch.tensor([float(record.get("floor_y", 0.))]),
                 floor_y_m=float(record.get("floor_y", 0.)),
                 floor_available=torch.tensor(["floor_y" in record]),
                 contact_threshold=torch.tensor([float(record.get("contact_threshold", .05))]),
                 contact_threshold_m=float(record.get("contact_threshold", .05)),
                 text_features=torch.zeros((1, text_dim)), text_available=torch.tensor([False]))
    if geometry_encoder == "mesh_bps":
        batch["object_bps"] = torch.from_numpy(record["object_bps"].copy())[None]
        batch["object_geometry_is_mesh"] = torch.tensor([True])
        batch["object_geometry_version"] = torch.tensor([int(record["object_geometry_version"])])
    return batch, timestamps, record


def add_human_prior(controls, path):
    with np.load(path, allow_pickle=False) as archive:
        human = torch.from_numpy(np.asarray(archive["human"], np.float32).copy())[None]
        if archive["human_mask"].dtype != np.bool_:
            raise ValueError("Human prior mask must be boolean (True=observed)")
        mask = torch.from_numpy(np.asarray(archive["human_mask"], bool).copy())[None]
        fps = float(archive["fps"])
        if "rest_offsets" not in archive:
            raise ValueError("Human prior needs its own rest_offsets; convert it with the current Kimodo bridge")
        rest_offsets = torch.from_numpy(np.asarray(archive["rest_offsets"], np.float32).copy())[None]
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Human prior FPS must be finite and positive")
    if rest_offsets.shape != (1, 22, 3) or not torch.isfinite(rest_offsets).all():
        raise ValueError("Human prior rest_offsets must be finite [22,3]")
    if human.shape != controls.values["human"].shape or mask.shape != human.shape:
        raise ValueError("Human prior length/shape must match the requested clip; resample explicitly first")
    overlap = mask & controls.masks["human"]
    if (human[overlap] != controls.values["human"][overlap]).any():
        raise ValueError("Human prior conflicts with another explicit human observation")
    controls.values["human"] = torch.where(mask, human, controls.values["human"])
    controls.masks["human"] |= mask
    controls.signatures = None
    return controls.validate(), fps, rest_offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted project checkpoint")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--reference", help="Canonical record used as scene and optional explicit observed values")
    inputs.add_argument("--scene", help="NPZ with only object_points, rest_offsets, fps; no target motion required")
    conditions = parser.add_mutually_exclusive_group()
    conditions.add_argument("--controls", help="Explicit JSON observations; from_reference must be requested per entry")
    conditions.add_argument("--mode", type=parse_mode,
                            help="TriDi-style H/O/C mode from reference (1=generate, 0=condition). "
                                 "Accepts 011, sample_011, human-to-object, or sparse patterns")
    parser.add_argument("--object-mesh", help="Canonical mesh asset NPZ from prepare_object_meshes.py")
    parser.add_argument("--object-mesh-scale", type=float, default=1.,
                        help="Recorded object_mesh_scale from the mesh manifest; OMOMO requires its sequence scale")
    parser.add_argument("--human-prior", help="Canonical human reference produced by Kimodo bridge")
    parser.add_argument("--text-features", help="Optional actual cached text embedding .npy")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--projection-steps", type=int, default=0)
    parser.add_argument("--ik-steps", type=int, default=90,
                        help="Pose-space iterations for position anchors; FK export is always enabled")
    parser.add_argument("--text-guidance", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output)
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    device = device_for(args.device)
    model, diffusion, checkpoint = load_model(args.checkpoint, device)
    batch, timestamps, metadata = prepare_batch(args.reference or args.scene, args.frames, args.start,
                                               model.config.text_dim, bool(args.scene),
                                               geometry_encoder=model.config.geometry_encoder,
                                               object_mesh=args.object_mesh, object_mesh_scale=args.object_mesh_scale)
    if args.scene and args.mode:
        raise ValueError("Benchmark --mode requires --reference. Use explicit --controls with a scene.")
    if args.text_features:
        embedding = np.load(args.text_features, allow_pickle=False)
        if embedding.shape != (model.config.text_dim,) or not np.isfinite(embedding).all():
            raise ValueError("Text embedding has wrong shape or non-finite values")
        batch["text_features"] = torch.as_tensor(embedding, dtype=torch.float32)[None]
        batch["text_available"][:] = True
    if args.controls:
        spec = json.loads(Path(args.controls).read_text(encoding="utf-8"))
        if args.scene and any(entry.get("from_reference") for entry in spec.get("entries", [])):
            raise ValueError("Scene-only files contain no reference motion to observe")
        controls = compile_controls(spec, batch, batch["valid_frames"])
    elif args.mode:
        controls = sample_controls(batch, batch["valid_frames"], args.mode,
                                   generator=torch.Generator().manual_seed(args.seed))
        print(describe_mode(args.mode), flush=True)
    else:
        controls = ControlBatch.empty_like(batch, batch["valid_frames"])
    if args.human_prior:
        controls, fps, rest_offsets = add_human_prior(controls, args.human_prior)
        if abs(fps - float(batch["fps"][0])) > 1e-4:
            raise ValueError("Human-prior FPS does not match scene; resample explicitly first")
        batch["rest_offsets"] = rest_offsets
        # The prior supplies a different person's skeleton: do not retain the scene/reference betas.
        with np.load(args.human_prior, allow_pickle=False) as prior:
            prior_metadata = body_metadata(dict(prior))
        if not prior_metadata:
            raise ValueError("Human prior needs body_model_type provenance; reconvert with the current Kimodo bridge")
        prior_metadata.setdefault("body_dataset", np.array("external"))
        metadata = {k: v for k, v in metadata.items() if k not in (*BODY_METADATA_KEYS, *BODY_GEOMETRY_KEYS)}
        metadata.update(prior_metadata)
        if "gender" in prior_metadata and "betas" in prior_metadata:
            prior_geometry = canonicalize_body({**prior_metadata, "rest_offsets": rest_offsets[0].numpy()})
            batch["rest_offsets"] = torch.from_numpy(prior_geometry["rest_offsets"].copy())[None]
            metadata.update({k: prior_geometry[k] for k in BODY_GEOMETRY_KEYS})
    feasibility = check_anchor_feasibility(controls, batch["object_points"])
    batch, controls = move_batch(batch, device), controls.to(device)
    projection_diagnostics = {}
    prediction = diffusion.sample(model, batch, controls, args.steps, args.seed,
                                  args.text_guidance, args.projection_steps, args.ik_steps,
                                  diagnostics=projection_diagnostics)
    from .evaluate import evaluate_batch
    report = evaluate_batch(prediction, controls, batch)
    arrays = {key: value[0].detach().cpu().numpy() for key, value in prediction.items()}
    arrays.update(body_metadata(metadata))
    arrays.update({k: metadata[k] for k in BODY_GEOMETRY_KEYS if k in metadata})
    arrays["human_position_decoding"] = np.array("root_rotation_FK")
    arrays["human_before_fk"] = projection_diagnostics["human_before_fk"][0].cpu().numpy()
    arrays.update(object_points=batch["object_points"][0].cpu().numpy(),
                  rest_offsets=batch["rest_offsets"][0].cpu().numpy(),
                  fps=np.float32(batch["fps"][0].cpu()), timestamps=timestamps,
                  schema_version=np.int32(1), checkpoint_step=np.int64(checkpoint["step"]))
    arrays["fps_verified_by_metadata"] = np.bool_(batch["fps_verified_by_metadata"][0].cpu())
    arrays["text_conditioned"] = np.bool_(batch["text_available"][0].cpu())
    arrays["contact_threshold"] = np.float32(batch["contact_threshold_m"])
    if model.config.geometry_encoder == "mesh_bps":
        arrays.update(object_bps=batch["object_bps"][0].cpu().numpy(),
                      object_geometry_is_mesh=np.bool_(True), object_geometry_version=np.int32(1))
    if bool(batch["floor_available"][0]):
        arrays["floor_y"] = np.float32(batch["floor_y_m"])
    for key in ("human", "object", "contact"):
        arrays[key + "_observed"] = controls.values[key][0].cpu().numpy()
        arrays[key + "_mask"] = controls.masks[key][0].cpu().numpy()
    for key in ("dataset", "sequence_id", "subject_id", "object_id", "text"):
        arrays[key] = metadata.get(key, np.array(""))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    write_json(output.with_suffix(".metrics.json"), {"metrics": report, "anchor_feasibility": feasibility,
               "before_pose_projection": projection_diagnostics["before_pose_projection"],
               "checkpoint_step": checkpoint["step"], "seed": args.seed, "controls": list(controls.signatures),
               "mode": args.mode, "mode_description": describe_mode(args.mode) if args.mode else None,
               "body_geometry_version": int(metadata.get("body_geometry_version", 0)),
               "checkpoint_geometry_objective_version": checkpoint["config"].get("geometry_objective_version", 1),
               "physical_execution_validated": False})
    print(str(output.resolve()))


if __name__ == "__main__":
    main()
