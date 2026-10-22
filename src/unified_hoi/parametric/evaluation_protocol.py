"""Explicit frame selection and aggregation for comparable native HOI evaluations."""
import hashlib
import json

import numpy as np

VERSION = "tridi_metrics_v2"
MODE_DESCRIPTIONS = {
    "010": "O|H,C", "100": "H|O,C", "001": "C|H,O",
    "011": "O,C|H", "101": "H,C|O", "110": "H,O|C", "111": "H,O,C",
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def protocol_options(profile="tridi", eval_fps=None, train_reference_fps=None,
                     generation_frame="pelvis"):
    # auto is retained as an alias, no longer an ignored CLI argument.
    if profile not in ("auto", "tridi", "native"):
        raise ValueError("Unknown evaluation profile")
    profile = "tridi" if profile == "auto" else profile
    fps = (1. if profile == "tridi" else 0.) if eval_fps is None else eval_fps
    train_fps = (10. if profile == "tridi" else 0.) if train_reference_fps is None else train_reference_fps
    if not np.isfinite([fps, train_fps]).all() or min(fps, train_fps) < 0:
        raise ValueError("Evaluation fps must be nonnegative; 0 means all native frames")
    if generation_frame not in ("pelvis", "native"):
        raise ValueError("generation_frame must be pelvis or native")
    return dict(version=VERSION, profile=profile, eval_fps=float(fps),
                train_reference_fps=float(train_fps), generation_frame=generation_frame,
                primary_aggregation="frame_weighted_mean", contact_aggregation="micro_counts",
                nn_distance="squared_l2", nn_self_exclusion="by_index",
                nn_backend="scipy_cKDTree_exact", nn_ties="backend_order_among_equal_distances",
                contact_threshold_metres=.05,
                evaluation_joints=73, contact_vertices=690,
                frame_selection="stride_from_first_available_source_frame; no window reset",
                object_feature="flattened_rotation_matrix_and_mesh_centroid",
                human_feature="73_joints_pelvis_relative_excluding_root")


def frame_indices(timestamps, source_fps, target_fps, origin):
    """Select a fixed source-sequence lattice, independent of clip/window boundaries."""
    timestamps = np.asarray(timestamps, np.float64)
    if (timestamps.ndim != 1 or not np.isfinite(timestamps).all()
            or not np.isfinite([source_fps, target_fps, origin]).all()
            or source_fps <= 0 or target_fps < 0):
        raise ValueError("Invalid timestamps/fps")
    if target_fps == 0:
        return np.arange(len(timestamps))
    ratio = source_fps / target_fps
    if ratio < 1 or not np.isclose(ratio, round(ratio), atol=1e-5, rtol=0):
        raise ValueError("Evaluation fps must divide native fps; no upsampling or interpolation")
    ticks = np.rint((timestamps - origin) * source_fps).astype(np.int64)
    error = np.abs(timestamps - (origin + ticks / source_fps))
    if (error > min(.0042, .2 / source_fps)).any():
        raise ValueError("Source clips do not share a consistent timestamp grid")
    return np.flatnonzero(ticks % int(round(ratio)) == 0)


def frame_plan(records, target_fps):
    """Read timestamps only; reject repeated source frames rather than overweight them."""
    metadata, origins, rates = [], {}, {}
    for row in records:
        source = row.get("source_sequence_id", row["sequence_id"])
        with np.load(row["path"], allow_pickle=False) as archive:
            ts, fps = archive["timestamps"].copy(), float(archive["fps"])
        if source in rates and not np.isclose(rates[source], fps):
            raise ValueError("Inconsistent fps within source sequence")
        rates[source] = fps
        origins[source] = min(origins.get(source, float("inf")), float(ts[0]))
        metadata.append((source, ts, fps))
    result, seen = [], set()
    for source, ts, fps in metadata:
        indices = frame_indices(ts, fps, target_fps, origins[source])
        ids = [f"{source}@{int(round(float(ts[i]) * 1e6))}" for i in indices]
        if seen.intersection(ids) or len(set(ids)) != len(ids):
            raise ValueError("Duplicate source frames in evaluation split")
        seen.update(ids)
        result.append(dict(indices=indices, timestamps=ts, source_sequence_id=source,
                           source_fps=fps, frame_ids=dict(zip(indices.tolist(), ids))))
    return result


def window_selection(plan, start, end):
    indices = plan["indices"]
    selected = indices[(indices >= start) & (indices < end)]
    return selected - start, [plan["frame_ids"][int(i)] for i in selected]


def aggregate_reports(rows):
    """Keep window diagnostics; headline PR/F1 use pooled TP/FP/FN, never mean F1."""
    if not rows:
        raise ValueError("No selected evaluation frames")
    metrics = {key: {
        "macro_mean_per_window": float(np.mean([r["metrics"][key] for r in rows])),
        "frame_weighted_mean": float(np.average([r["metrics"][key] for r in rows],
                                               weights=[r["frames"] for r in rows])),
    } for key in rows[0]["metrics"]}
    primary = {k: v["frame_weighted_mean"] for k, v in metrics.items()}
    micro = {}
    for name in rows[0]["details"]:
        counts = {k: sum(r["details"][name][k] for r in rows) for k in ("TP", "FP", "FN", "TN")}
        tp, fp, fn, tn = (counts[k] for k in ("TP", "FP", "FN", "TN"))
        scores = dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                      f1=2 * tp / max(2 * tp + fp + fn, 1),
                      accuracy_percent=100 * (tp + tn) / max(tp + fp + fn + tn, 1))
        micro[name] = counts | scores
        for k in ("precision", "recall", "f1"):
            primary[name + "_" + k.upper()] = scores[k]
    return metrics, primary, micro


def assert_comparable(left, right, mode):
    """Reject accidental cross-mode/dataset/frame/protocol comparisons."""
    for name, report in (("left", left), ("right", right)):
        if report.get("protocol", {}).get("version") != VERSION:
            raise ValueError(f"{name}: old or missing evaluation protocol; re-evaluate predictions")
        if mode not in report.get("reconstruction", {}):
            raise ValueError(f"{name}: mode {mode} is missing; 100 and 101 are different tasks")
    for key in ("dataset", "protocol", "evaluation_frames_sha256", "reference_train_frames_sha256",
                "reference_values_sha256",
                "text_conditioned", "support_sha256", "body_assets_sha256", "split", "steps", "seed", "window", "sampler"):
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"Evaluation mismatch: {key}")
    a, b = left.get("generation", {}).get(mode), right.get("generation", {}).get(mode)
    if (a is None) != (b is None):
        raise ValueError("Generation results are missing on one side")
    if a is not None:
        for key in ("feature_kind", "reference_train_values_sha256", "distance",
                    "feature_frames_generated", "feature_frames_test", "feature_frames_train"):
            if key not in a or key not in b or a[key] != b[key]:
                raise ValueError(f"Generation evaluation mismatch: {key}")
    return {"mode": mode, "description": MODE_DESCRIPTIONS[mode],
            "left": left["reconstruction"][mode], "right": right["reconstruction"][mode],
            "generation_left": left.get("generation", {}).get(mode),
            "generation_right": right.get("generation", {}).get(mode)}


def add_protocol_arguments(parser):
    parser.add_argument("--profile", choices=("tridi", "native", "auto"), default="tridi",
                        help="tridi: evaluate at 1fps, train reference at 10fps; native: all frames")
    parser.add_argument("--eval-fps", type=float, help="Override evaluation rate; 0 keeps all native frames")
    parser.add_argument("--train-reference-fps", type=float, help="SD train-reference rate; 0 keeps all frames")
    parser.add_argument("--generation-frame", choices=("pelvis", "native"), default="pelvis",
                        help="Coordinates for NN object centers; human features are always pelvis-relative")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--predictions-dir", help="Re-score saved mode/*.npz without diffusion; requires summary.json")
