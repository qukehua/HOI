"""Uni-HOI task metrics plus kinematic control diagnostics; no physics certification."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .controls import MODALITIES, ControlBatch, compile_controls
from .data import load_record
from .geometry import (
    SMPL_PARENTS, forward_kinematics, human_in_object_frame, nearest_surface,
    object_to_world, rotation6d_to_matrix, rotation_angle,
)


def _mean(values: Tensor, mask: Tensor) -> float | None:
    selected = values[torch.broadcast_to(mask, values.shape)]
    return float(selected.double().mean().item()) if selected.numel() else None


def _count(mask: Tensor) -> int:
    return int(mask.sum().item())


def _rotation_valid(d6: Tensor) -> Tensor:
    a, b = d6[..., :3], d6[..., 3:]
    norm = torch.linalg.vector_norm(a, dim=-1)
    unit = a / norm.clamp_min(1e-8).unsqueeze(-1)
    orth = b - (unit * b).sum(-1, keepdim=True) * unit
    return (norm > 1e-7) & (torch.linalg.vector_norm(orth, dim=-1) > 1e-7)


def _timestamps(batch: Mapping[str, Any], valid: Tensor) -> tuple[Tensor | None, str]:
    if "timestamps" in batch:
        stamps = torch.as_tensor(batch["timestamps"], dtype=torch.float64, device=valid.device)
        if stamps.ndim == 1 and valid.shape[0] == 1:
            stamps = stamps.unsqueeze(0)
        if stamps.shape != valid.shape or not torch.isfinite(stamps[valid]).all():
            raise ValueError("timestamps must have shape [B,T] and be finite on valid frames")
        for b in range(valid.shape[0]):
            observed = stamps[b, valid[b]]
            if len(observed) > 1 and not (torch.diff(observed) > 0).all():
                raise ValueError("valid timestamps must strictly increase")
        return torch.where(valid, stamps, torch.zeros_like(stamps)), "timestamps"
    if "fps" in batch:
        fps = torch.as_tensor(batch["fps"], dtype=torch.float64, device=valid.device)
        if fps.ndim == 0:
            fps = fps.repeat(valid.shape[0])
        if fps.shape != (valid.shape[0],) or not torch.isfinite(fps).all() or not (fps > 0).all():
            raise ValueError("fps must be a finite positive scalar or [B]")
        return torch.arange(valid.shape[1], device=valid.device)[None] / fps[:, None], "fps"
    return None, "unavailable"


def _differentiate(values: Tensor, times: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    pair_valid = valid[:, 1:] & valid[:, :-1]
    dt = times[:, 1:] - times[:, :-1]
    dt = torch.where(pair_valid, dt, torch.ones_like(dt))
    divisor = dt.reshape(*dt.shape, *([1] * (values.ndim - 2)))
    derivative = (values[:, 1:] - values[:, :-1]) / divisor
    return derivative, (times[:, 1:] + times[:, :-1]) / 2, pair_valid


def _motion_statistics(positions: Tensor, times: Tensor | None, valid: Tensor,
                       prefix: str) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = {}
    current, stamps, support = positions, times, valid
    for label, unit in (("speed", "m_s"), ("acceleration", "m_s2"), ("jerk", "m_s3")):
        key = f"{prefix}_{label}_mean_{unit}"
        if stamps is None or current.shape[1] < 2:
            result[key] = None
            result[f"{prefix}_{label}_sample_count"] = 0
            stamps = None
            continue
        current, stamps, support = _differentiate(current, stamps, support)
        magnitude = torch.linalg.vector_norm(current, dim=-1)
        selected = support.reshape(*support.shape, *([1] * (magnitude.ndim - 2))).expand_as(magnitude)
        result[key] = _mean(magnitude, selected)
        result[f"{prefix}_{label}_sample_count"] = _count(selected)
    return result


@torch.no_grad()
def evaluate_batch(states: Mapping[str, Tensor], controls: ControlBatch,
                   batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return a flat dictionary of JSON scalars without changing any input.

    Contact support comes only from explicit control masks, never predictions
    or unobserved reference contacts. Missing supports produce None, not zero.
    Geometry uses sampled object surfaces. Floor/FK/slip are kinematic proxies;
    acceleration and jerk describe motion but do not measure realism.
    """
    # validate() caches signatures; validate a shallow wrapper to keep evaluation
    # pure even when the caller has not populated that optional cache yet.
    controls = ControlBatch(controls.values, controls.masks, controls.valid_frames, controls.signatures).validate()
    valid = controls.valid_frames
    if "valid_frames" in batch and not torch.equal(torch.as_tensor(batch["valid_frames"], device=valid.device), valid):
        raise ValueError("batch and controls disagree on valid_frames")
    if not valid.any():
        raise ValueError("evaluation requires at least one valid frame")
    clean: dict[str, Tensor] = {}
    for name in MODALITIES:
        value = states[name]
        if value.shape != controls.values[name].shape or value.device != valid.device:
            raise ValueError(f"{name} prediction shape/device differs from controls")
        active = valid.reshape(*valid.shape, *([1] * (value.ndim - 2))).expand_as(value)
        if not torch.isfinite(value[active]).all():
            raise ValueError(f"{name} predictions contain nonfinite values on valid frames")
        clean[name] = torch.where(active, value, torch.zeros_like(value))
    human, obj = clean["human"], clean["object"]
    bsize, _, njoints, _ = human.shape
    points = torch.as_tensor(batch["object_points"], device=human.device, dtype=human.dtype)
    offsets = torch.as_tensor(batch["rest_offsets"], device=human.device, dtype=human.dtype)
    if points.ndim != 3 or points.shape[0] != bsize or points.shape[-1] != 3 or points.shape[1] < 1:
        raise ValueError("object_points must have shape [B,K,3], K > 0")
    if offsets.shape != (bsize, njoints, 3):
        raise ValueError("rest_offsets must have shape [B,J,3]")
    if not torch.isfinite(points).all() or not torch.isfinite(offsets).all():
        raise ValueError("static geometry must be finite")
    tolerance = torch.as_tensor(batch.get("contact_threshold_m", batch.get("contact_threshold", .05)),
                                device=human.device, dtype=human.dtype).reshape(-1)
    floor = torch.as_tensor(batch.get("floor_y_m", batch.get("floor_y", 0.)),
                            device=human.device, dtype=human.dtype).reshape(-1)
    if tolerance.numel() not in (1, bsize) or floor.numel() not in (1, bsize):
        raise ValueError("Contact thresholds and floor heights must be scalar or [B]")
    if not torch.isfinite(tolerance).all() or (tolerance <= 0).any() or not torch.isfinite(floor).all():
        raise ValueError("contact_threshold_m must be positive; floor_y_m must be finite")
    tolerance, floor = tolerance.reshape(-1, 1, 1), floor.reshape(-1, 1, 1)
    times, time_source = _timestamps(batch, valid)
    report: dict[str, Any] = {
        "confidence": "kinematic_proxies_only_not_physical_validation",
        "physical_execution_evaluated": False,
        "smoothness_is_realism_metric": False,
        "valid_frame_count": _count(valid),
        "timing_source": time_source,
        "contact_threshold_m": float(tolerance.flatten()[0]) if (tolerance == tolerance.flatten()[0]).all() else None,
        "floor_y_m": float(floor.flatten()[0]) if (floor == floor.flatten()[0]).all() else None,
    }
    error_sum, observed_count = 0.0, 0
    for name in MODALITIES:
        mask = controls.masks[name]
        error = (clean[name].float() - controls.values[name].float()).abs()
        count = _count(mask)
        report[f"{name}_anchor_feature_count"] = count
        report[f"{name}_anchor_feature_mae"] = _mean(error, mask)
        error_sum += float(error[mask].double().sum().item())
        observed_count += count
    report["anchor_feature_count"] = observed_count
    report["anchor_feature_mae"] = error_sum / observed_count if observed_count else None

    for name in ("human", "object"):
        pred, target, mask = clean[name], controls.values[name], controls.masks[name]
        full_xyz = mask[..., :3].all(-1)
        component_mask = mask[..., :3]
        pos_error = torch.linalg.vector_norm(pred[..., :3] - target[..., :3], dim=-1)
        report[f"{name}_anchor_position_count"] = _count(full_xyz)
        report[f"{name}_anchor_position_error_m"] = _mean(pos_error, full_xyz)
        report[f"{name}_anchor_position_component_mae_m"] = _mean((pred[..., :3] - target[..., :3]).abs(), component_mask)
        full_rot = mask[..., 3:9].all(-1)
        pred_ok, target_ok = _rotation_valid(pred[..., 3:9]), _rotation_valid(target[..., 3:9])
        angle = torch.rad2deg(rotation_angle(rotation6d_to_matrix(pred[..., 3:9]), rotation6d_to_matrix(target[..., 3:9])))
        # A degenerate encoding must not inherit the geometry helper's identity
        # fallback and appear to perfectly satisfy an identity rotation anchor.
        angle = torch.where(pred_ok & target_ok, angle, torch.full_like(angle, 180.0))
        report[f"{name}_anchor_rotation_count"] = _count(full_rot)
        report[f"{name}_anchor_rotation_error_deg"] = _mean(angle, full_rot)
        active = valid[..., None].expand_as(pred_ok) if name == "human" else valid
        report[f"{name}_degenerate_rotation_rate"] = _mean((~pred_ok).float(), active)

    contact_mask = controls.masks["contact"] & valid[..., None]
    positive = contact_mask & (controls.values["contact"] == 1)
    negative = contact_mask & (controls.values["contact"] == 0)
    distance, _ = nearest_surface(human[..., :3], obj, points)
    geometrically_close = distance < tolerance
    satisfied = torch.where(positive, geometrically_close, ~geometrically_close)
    report["contact_anchor_count"] = _count(contact_mask)
    report["contact_positive_anchor_count"] = _count(positive)
    report["contact_negative_anchor_count"] = _count(negative)
    report["contact_positive_distance_m"] = _mean(distance, positive)
    report["contact_positive_violation_rate"] = _mean((~geometrically_close).float(), positive)
    report["contact_negative_violation_rate"] = _mean(geometrically_close.float(), negative)
    report["contact_anchor_satisfaction_rate"] = _mean(satisfied.float(), contact_mask)
    value_correct = (clean["contact"] >= 0.5) == (controls.values["contact"] == 1)
    report["contact_anchor_value_accuracy"] = _mean(value_correct.float(), contact_mask)
    has_anchors = contact_mask.flatten(1).any(1)
    all_satisfied = ((~contact_mask) | satisfied).flatten(1).all(1)
    report["contact_evaluated_sequence_count"] = _count(has_anchors)
    report["contact_all_anchors_satisfied_rate"] = _mean(all_satisfied.float(), has_anchors)
    has_both = positive.flatten(1).any(1) & negative.flatten(1).any(1)
    report["contact_both_signs_sequence_count"] = _count(has_both)
    report["contact_positive_and_negative_all_satisfied_rate"] = _mean(all_satisfied.float(), has_both)

    report["contact_slip_mean_m_s"] = None
    report["contact_slip_pair_count"] = 0
    if times is not None and valid.shape[1] > 1:
        local = human_in_object_frame(human[..., :3], obj)
        velocity, _, pairs = _differentiate(local, times, valid)
        slip_support = positive[:, 1:] & positive[:, :-1] & pairs[..., None]
        report["contact_slip_pair_count"] = _count(slip_support)
        report["contact_slip_mean_m_s"] = _mean(torch.linalg.vector_norm(velocity, dim=-1), slip_support)

    parents = tuple(batch.get("parents", SMPL_PARENTS))
    if len(parents) < njoints or any(not 0 <= int(parents[i]) < i for i in range(1, njoints)):
        raise ValueError("parents must provide a topologically ordered skeleton")
    fk = forward_kinematics(rotation6d_to_matrix(human[..., 3:9]), offsets[:, None], human[..., 0, :3], parents)
    fk_error = torch.linalg.vector_norm(human[..., 1:, :3] - fk[..., 1:, :], dim=-1)
    fk_support = valid[..., None].expand_as(fk_error)
    report["human_fk_nonroot_count"] = _count(fk_support)
    report["human_fk_error_m"] = _mean(fk_error, fk_support)

    world_points = object_to_world(points, obj)
    for name, positions in (("human", human[..., :3]), ("object", world_points)):
        depth = (floor - positions[..., 1]).clamp_min(0)
        support = valid[..., None].expand_as(depth)
        report[f"{name}_floor_below_rate"] = _mean((depth > 0).float(), support)
        report[f"{name}_floor_depth_mean_m"] = _mean(depth, support)
        report[f"{name}_floor_depth_max_m"] = float(depth[support].max().item())
    timing = batch.get("fps_verified_by_metadata")
    report["timing_verified_by_metadata"] = bool(torch.as_tensor(timing).all()) if timing is not None else None
    report["floor_calibrated"] = bool(torch.as_tensor(batch.get("floor_available", False)).all())
    report.update(_motion_statistics(human[..., :3], times, valid, "human"))
    report.update(_motion_statistics(obj[..., :3], times, valid, "object"))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--controls", type=Path,
                        help="Optional override; default is the exact anchors/masks stored in the prediction")
    parser.add_argument("--reference-start", type=int, default=0)
    parser.add_argument("--profile", choices=("uni-hoi", "diagnostics"), default="uni-hoi")
    parser.add_argument("--task", choices=("auto", "object-to-human", "human-to-object", "text-to-hoi"), default="auto")
    parser.add_argument("--metric-seed", type=int, default=42)
    from .paper_metrics import add_asset_arguments, assets_from_args, evaluate_paper_sequence
    add_asset_arguments(parser)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    reference = load_record(args.reference)
    with np.load(args.prediction, allow_pickle=False) as archive:
        prediction = {key: archive[key] for key in archive.files}
    for name in (*MODALITIES, "fps", "timestamps"):
        if name not in prediction:
            raise ValueError(f"prediction archive is missing {name}")
    if prediction["human"].ndim != 3:
        raise ValueError("prediction archive must contain one unbatched sequence [T,J,9]")
    length = len(prediction["human"])
    start, end = args.reference_start, args.reference_start + length
    if length < 1 or start < 0 or end > len(reference["human"]):
        raise ValueError("prediction length/reference-start is outside the reference sequence")
    fps = float(np.asarray(prediction["fps"]).item())
    if not math.isfinite(fps) or fps <= 0 or not np.isclose(fps, float(reference["fps"])):
        raise ValueError("prediction and reference fps must match and be positive")
    stamps = np.asarray(prediction["timestamps"], dtype=np.float64)
    ref_stamps = np.asarray(reference["timestamps"][start:end], dtype=np.float64)
    if stamps.shape != (length,) or not np.isfinite(stamps).all():
        raise ValueError("prediction timestamps must be finite [T] seconds")
    if length > 1 and (not np.all(np.diff(stamps) > 0) or not np.allclose(
            stamps - stamps[0], ref_stamps - ref_stamps[0], rtol=0, atol=1e-6)):
        raise ValueError("prediction/reference timestamps disagree; align frames before evaluation")
    states = {name: torch.as_tensor(np.asarray(prediction[name], dtype=np.float32).copy())[None] for name in MODALITIES}
    reference_states = {
        name: torch.as_tensor(np.asarray(reference[name][start:end], dtype=np.float32).copy())[None]
        for name in MODALITIES
    }
    valid = torch.ones(1, length, dtype=torch.bool)
    if args.controls:
        controls = compile_controls(json.loads(args.controls.read_text(encoding="utf-8")), reference_states, valid)
        control_source = "explicit_json_override"
    else:
        control_keys = {f"{name}_{suffix}" for name in MODALITIES for suffix in ("observed", "mask")}
        present = control_keys.intersection(prediction)
        if present and present != control_keys:
            raise ValueError(f"prediction has incomplete stored controls; missing {sorted(control_keys - present)}")
        if present:
            for name in MODALITIES:
                if np.asarray(prediction[f"{name}_mask"]).dtype != np.bool_:
                    raise ValueError(f"stored {name}_mask must be boolean (True=observed)")
            controls = ControlBatch(
                {name: torch.as_tensor(np.asarray(prediction[f"{name}_observed"], dtype=np.float32).copy())[None]
                 for name in MODALITIES},
                {name: torch.from_numpy(np.asarray(prediction[f"{name}_mask"]).copy())[None]
                 for name in MODALITIES},
                valid,
            ).validate()
            control_source = "saved_prediction_anchors"
        else:
            controls = ControlBatch.empty_like(reference_states, valid)
            control_source = "empty_no_saved_anchors"
    batch = {
        "object_points": torch.as_tensor(np.asarray(reference["object_points"], dtype=np.float32).copy())[None],
        "rest_offsets": torch.as_tensor(np.asarray(reference["rest_offsets"], dtype=np.float32).copy())[None],
        "timestamps": torch.from_numpy(stamps.copy())[None],
        "fps": fps,
        "fps_verified_by_metadata": bool(reference.get("fps_verified_by_metadata", False)),
        "floor_available": "floor_y" in reference,
        "floor_y_m": float(reference.get("floor_y", 0.)),
        "contact_threshold_m": float(reference.get("contact_threshold", .05)),
    }
    report = evaluate_batch(states, controls, batch)
    if args.profile == "uni-hoi":
        report["paper_evaluation"] = evaluate_paper_sequence(
            states, controls, reference, assets=assets_from_args(args), start=start, task=args.task,
            text_conditioned=bool(np.asarray(prediction.get("text_conditioned", False)).item()),
            chamfer_samples=args.chamfer_samples, seed=args.metric_seed)
    report["reference_start_frame"] = start
    report["control_source"] = control_source
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
