"""Generate a sequence from scene geometry and explicit observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .controls import (ControlBatch, GENERATED_MODES, PARTIAL_PATTERNS, check_anchor_feasibility,
                       compile_controls, sample_controls)
from .data import load_record
from .runtime import device_for, load_model, move_batch, write_json


def prepare_batch(path, frames=None, start=0, text_dim=512, scene_only=False):
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
        n = min(frames or len(record["human"]), len(record["human"]) - start)
        if start < 0 or n < 1:
            raise ValueError("Requested clip is outside the reference")
        fps = float(record["fps"])
        timestamps = record["timestamps"][start:start + n]
        state = {key: record[key][start:start + n] for key in ("human", "object", "contact")}
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
    conditions.add_argument("--mode", choices=(*GENERATED_MODES, *PARTIAL_PATTERNS),
                            help="Benchmark masks taken from reference: 1=generate, 0=condition, ordered H/O/C")
    parser.add_argument("--human-prior", help="Canonical human reference produced by Kimodo bridge")
    parser.add_argument("--text-features", help="Optional actual cached text embedding .npy")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--projection-steps", type=int, default=0)
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
                                               model.config.text_dim, bool(args.scene))
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
    else:
        controls = ControlBatch.empty_like(batch, batch["valid_frames"])
    if args.human_prior:
        controls, fps, rest_offsets = add_human_prior(controls, args.human_prior)
        if abs(fps - float(batch["fps"][0])) > 1e-4:
            raise ValueError("Human-prior FPS does not match scene; resample explicitly first")
        batch["rest_offsets"] = rest_offsets
    feasibility = check_anchor_feasibility(controls, batch["object_points"])
    batch, controls = move_batch(batch, device), controls.to(device)
    prediction = diffusion.sample(model, batch, controls, args.steps, args.seed,
                                  args.text_guidance, args.projection_steps)
    from .evaluate import evaluate_batch
    report = evaluate_batch(prediction, controls, batch)
    arrays = {key: value[0].detach().cpu().numpy() for key, value in prediction.items()}
    arrays.update(object_points=batch["object_points"][0].cpu().numpy(),
                  rest_offsets=batch["rest_offsets"][0].cpu().numpy(),
                  fps=np.float32(batch["fps"][0].cpu()), timestamps=timestamps,
                  schema_version=np.int32(1), checkpoint_step=np.int64(checkpoint["step"]))
    arrays["fps_verified_by_metadata"] = np.bool_(batch["fps_verified_by_metadata"][0].cpu())
    arrays["contact_threshold"] = np.float32(batch["contact_threshold_m"])
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
               "checkpoint_step": checkpoint["step"], "seed": args.seed, "controls": list(controls.signatures),
               "physical_execution_validated": False})
    print(str(output.resolve()))


if __name__ == "__main__":
    main()
