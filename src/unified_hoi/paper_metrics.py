"""Uni-HOI (arXiv:2604.27491v2) task metrics, separate from control diagnostics.

Definitions follow its section 4.1 / tables 1-3 and the referenced OMOMO and
Object Pop-up implementations. See docs/EVALUATION.md for unresolved protocol
details. Matching metric definitions does not establish table comparability.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from .geometry import SMPL_PARENTS, forward_kinematics, rotation6d_to_matrix


TEXT_METRICS = ("FID", "R_precision_top1", "R_precision_top2", "R_precision_top3", "Diversity")
HUMAN_METRICS = ("HandJPE_cm", "MPJPE_cm", "C_prec", "C_rec", "C_acc", "c_percent")
OBJECT_METRICS = ("E_ch_m", "E_v2v_m")
MODE_TASKS = {"101": "object-to-human", "011": "human-to-object", "111": "text-to-hoi"}


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _xyz(value, name, ndim):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != ndim or value.shape[-1] != 3 or not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty finite xyz with {ndim} dimensions")
    return value


def rotation_matrices(state):
    """Reject degenerate encodings instead of silently using the model's fallback."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape[-1] != 9 or not np.isfinite(state).all():
        raise ValueError("pose states must be finite [...,9]")
    a, b = state[..., 3:6], state[..., 6:9]
    if (np.linalg.norm(a, axis=-1) < 1e-7).any() or (np.linalg.norm(np.cross(a, b), axis=-1) < 1e-7).any():
        raise ValueError("paper geometry metrics require nondegenerate rotations")
    return rotation6d_to_matrix(torch.from_numpy(state[..., 3:])).numpy()


def omomo_joints24(human, rest_offsets24):
    """Decode root/global rotations with subject-specific SMPL-H rest offsets.

    OMOMO evaluates the SMPL-decoded skeleton, not its independently predicted
    xyz channels. Joints 22/23 are middle-finger bases (SMPL-H 28/43), whose
    positions depend only on wrist transforms and the two supplied offsets.
    """
    human = np.asarray(human, dtype=np.float64)
    offsets = _xyz(rest_offsets24, "rest_offsets24", 2)
    if human.ndim != 3 or human.shape[1:] != (22, 9) or offsets.shape != (24, 3):
        raise ValueError("OMOMO decoding needs human[T,22,9] and real rest_offsets24[24,3]")
    rotations = rotation_matrices(human)
    body = forward_kinematics(torch.from_numpy(rotations), torch.from_numpy(offsets[:22]),
                              torch.from_numpy(human[:, 0, :3]), SMPL_PARENTS).numpy()
    hands = body[:, [20, 21]] + np.einsum("thij,hj->thi", rotations[:, [20, 21]], offsets[22:])
    return np.concatenate((body, hands), axis=1)


def human_metrics(predicted, reference, predicted_hand_distances, reference_hand_distances):
    """OMOMO rules: global hands, pelvis-relative 24-joint MPJPE, 5cm OR-contact.

    Input units are metres; position errors are returned in cm as Uni-HOI asks
    (OMOMO's original function returns mm). c_percent is a fraction in [0,1],
    matching table 2's presentation. Empty precision/recall denominators give 0
    exactly as in OMOMO, and are accompanied by confusion counts.
    """
    pred, ref = _xyz(predicted, "predicted", 3), _xyz(reference, "reference", 3)
    if pred.shape != ref.shape or pred.shape[1] != 24:
        raise ValueError("human metrics require matching [T,24,3] skeletons")
    pd, rd = np.asarray(predicted_hand_distances), np.asarray(reference_hand_distances)
    if pd.shape != (len(pred), 2) or rd.shape != pd.shape or not np.isfinite([pd, rd]).all():
        raise ValueError("hand distances must be finite [T,2]")
    if (pd < 0).any() or (rd < 0).any():
        raise ValueError("hand distances must be nonnegative")
    pc, rc = (pd < .05).any(-1), (rd < .05).any(-1)
    tp, fp = int((pc & rc).sum()), int((pc & ~rc).sum())
    tn, fn = int((~pc & ~rc).sum()), int((~pc & rc).sum())
    result = {
        "HandJPE_cm": float(np.linalg.norm(pred[:, 22:] - ref[:, 22:], axis=-1).mean() * 100),
        "MPJPE_cm": float(np.linalg.norm((pred - pred[:, :1]) - (ref - ref[:, :1]), axis=-1).mean() * 100),
        "C_prec": tp / (tp + fp) if tp + fp else 0.,
        "C_rec": tp / (tp + fn) if tp + fn else 0.,
        "C_acc": (tp + tn) / len(pred),
        "c_percent": float(pc.mean()),
    }
    return result, {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "frames": len(pred),
                    "gt_contact_fraction": float(rc.mean())}


def sample_surface(vertices, faces, count, rng):
    """Uniform area sampling with an explicit RNG, independent of global state."""
    if count < 1:
        raise ValueError("surface sample count must be positive")
    vertices = _xyz(vertices, "vertices", 2)
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in "iu":
        raise ValueError("faces must be nonempty integer triangles")
    if (faces < 0).any() or (faces >= len(vertices)).any():
        raise ValueError("mesh face index out of bounds")
    triangles = vertices[faces]
    weights = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                      triangles[:, 2] - triangles[:, 0]), axis=-1)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("mesh must have positive finite surface area")
    selected = triangles[rng.choice(len(faces), count, p=weights / weights.sum())]
    uv = rng.random((count, 2))
    uv[uv.sum(-1) > 1] = 1 - uv[uv.sum(-1) > 1]
    return selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) + uv[:, 1:] * (selected[:, 2] - selected[:, 0])


def object_metrics(predicted_pose, reference_pose, vertices, faces, *, chamfer_samples=10000, seed=0,
                   predicted_scale=None, reference_scale=None):
    """Corresponding full-mesh V2V and sum of bidirectional *unsquared* distances.

    Object Pop-up samples 10,000 points independently from each posed surface.
    Consequently even identical meshes have a small nonzero sampled Chamfer.
    No ICP, Procrustes, recentering, or symmetry alignment is applied.
    """
    pred, ref = np.asarray(predicted_pose), np.asarray(reference_pose)
    if pred.shape != ref.shape or pred.ndim != 2 or pred.shape[1] != 9 or not len(pred):
        raise ValueError("object poses must match [T,9]")
    vertices = _xyz(vertices, "vertices", 2)
    rp, rr = rotation_matrices(pred), rotation_matrices(ref)
    ps = np.ones(len(pred)) if predicted_scale is None else np.asarray(predicted_scale)
    rs = np.ones(len(ref)) if reference_scale is None else np.asarray(reference_scale)
    if ps.shape != (len(pred),) or rs.shape != ps.shape or not np.isfinite([ps, rs]).all() or (ps <= 0).any() or (rs <= 0).any():
        raise ValueError("object scales must be positive finite [T]")
    rng = np.random.default_rng(seed)
    v2v, chamfer, centers = [], [], []
    for t in range(len(pred)):
        pv, rv = (vertices * ps[t]) @ rp[t].T + pred[t, :3], (vertices * rs[t]) @ rr[t].T + ref[t, :3]
        v2v.append(np.linalg.norm(pv - rv, axis=-1).mean())
        centers.append(np.linalg.norm(pv.mean(0) - rv.mean(0)))
        pp = sample_surface(vertices, faces, chamfer_samples, rng) * ps[t] @ rp[t].T + pred[t, :3]
        gp = sample_surface(vertices, faces, chamfer_samples, rng) * rs[t] @ rr[t].T + ref[t, :3]
        chamfer.append(cKDTree(gp).query(pp)[0].mean() + cKDTree(pp).query(gp)[0].mean())
    return {"E_ch_m": float(np.mean(chamfer)), "E_v2v_m": float(np.mean(v2v))}, {
        "object_centroid_error_m": float(np.mean(centers)), "frames": len(pred),
        "vertices": len(vertices), "chamfer_samples_per_surface": chamfer_samples,
    }


class PaperAssets:
    """Read raw geometry/evaluation offsets without changing training archives.

    OMOMO joblib is loaded only from the trusted user-provided raw release.
    Full object vertices are used for hand contact; training's 1024 surface
    samples and 22 contact bits are never substituted for these assets.
    """
    def __init__(self, behave_objects="data/raw/behave/objects", omomo_objects="OMOMO/data/captured_objects",
                 omomo_raw="OMOMO/data/test_diffusion_manip_seq_joints24.p", omomo_source_fps=30.):
        self.behave_objects, self.omomo_objects = Path(behave_objects), Path(omomo_objects)
        self.omomo_raw, self.omomo_source_fps = Path(omomo_raw), float(omomo_source_fps)
        if not np.isfinite(self.omomo_source_fps) or self.omomo_source_fps <= 0:
            raise ValueError("omomo_source_fps must be positive")
        self.meshes, self.raw = {}, None

    def mesh(self, record):
        import trimesh
        from .preprocess import _mesh_for_behave
        dataset, obj = str(record["dataset"]), str(record["object_id"])
        key = dataset, obj
        if key not in self.meshes:
            if dataset == "behave":
                path = _mesh_for_behave(self.behave_objects, obj)
            elif dataset == "omomo":
                path = self.omomo_objects / f"{obj}_cleaned_simplified.obj"
            else:
                raise ValueError(f"No Uni-HOI geometry adapter for {dataset}")
            if not path.is_file():
                raise FileNotFoundError(f"Paper metrics require the original object mesh: {path}")
            mesh = trimesh.load(str(path), process=False, force="mesh")
            vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
            if dataset == "behave":
                vertices -= vertices.mean(0)
            self.meshes[key] = vertices, np.array(mesh.faces, dtype=np.int64), str(path)
        return self.meshes[key]

    def omomo_metadata(self, record, frame_indices):
        if self.raw is None:
            import joblib
            if not self.omomo_raw.is_file():
                raise FileNotFoundError(f"OMOMO evaluation needs raw 24-joint offsets/scales: {self.omomo_raw}")
            records = joblib.load(self.omomo_raw)
            self.raw = {str(r["seq_name"]): r for r in records.values()}
            if len(self.raw) != len(records):
                raise ValueError("duplicate raw OMOMO sequence IDs")
        name = str(record.get("source_sequence_id", record["sequence_id"]))
        if name not in self.raw:
            raise ValueError(f"OMOMO raw release does not contain {name}")
        raw = self.raw[name]
        offsets = np.asarray(raw["rest_offsets"], dtype=np.float64)
        if offsets.shape != (24, 3) or not np.allclose(offsets[:22], record["rest_offsets"], atol=1e-6):
            raise ValueError("raw/canonical skeleton mismatch; cannot guess hand offsets")
        stamps = np.asarray(record["timestamps"])[frame_indices]
        source_times = np.asarray(raw.get("timestamps", np.arange(len(raw["trans"])) / self.omomo_source_fps))
        idx = cKDTree(source_times[:, None]).query(stamps[:, None])[1]
        if not np.allclose(source_times[idx], stamps, atol=1e-6, rtol=0):
            raise ValueError("cannot align OMOMO raw and canonical timestamps")
        transform = np.asarray(record["source_to_world"], dtype=np.float64)
        expected = np.asarray(raw["obj_trans"])[idx].reshape(-1, 3) @ transform.T
        expected_r = transform @ np.asarray(raw["obj_rot"])[idx]
        current = np.asarray(record["object"])[frame_indices]
        if not np.allclose(expected, current[:, :3], atol=2e-5) or not np.allclose(expected_r, rotation_matrices(current), atol=2e-5):
            raise ValueError("raw/canonical OMOMO object transforms disagree")
        scales = np.asarray(raw["obj_scale"], dtype=np.float64).reshape(-1)
        scales = np.repeat(scales, len(source_times)) if len(scales) == 1 else scales
        return offsets, scales[idx]


def infer_task(controls):
    valid = array(controls.valid_frames)[0]
    full, any_known = {}, {}
    for name in ("human", "object", "contact"):
        mask = array(controls.masks[name])[0, valid]
        full[name], any_known[name] = bool(mask.all()), bool(mask.any())
    if full["human"] and not any_known["object"] and not any_known["contact"]:
        return "human-to-object"
    if full["object"] and not any_known["human"] and not any_known["contact"]:
        return "object-to-human"
    if not any(any_known.values()):
        return "unconditioned"
    return "mixed-controls"


def evaluate_paper_sequence(states, controls, record, *, assets=None, start=0, task="auto",
                            text_conditioned=False, chamfer_samples=10000, seed=0):
    """Evaluate one aligned window. Ground truth is read only after generation."""
    controls.validate()
    if controls.valid_frames.shape[0] != 1:
        raise ValueError("paper evaluation takes one sequence/window at a time")
    length = controls.valid_frames.shape[1]
    valid = array(controls.valid_frames)[0]
    indices = start + np.flatnonzero(valid)
    if start < 0 or not len(indices) or indices[-1] >= len(record["human"]):
        raise ValueError("paper reference window is out of bounds")
    detected = infer_task(controls)
    if task == "auto":
        task = "text-to-hoi" if detected == "unconditioned" and text_conditioned else detected
    expected = "unconditioned" if task == "text-to-hoi" else task
    if expected != detected:
        raise ValueError(f"task {task} conflicts with actual control masks ({detected})")
    report = {"paper": "Uni-HOI arXiv:2604.27491v2", "task": task, "dataset": str(record["dataset"]),
              "metrics": {}, "details": {}, "directly_comparable_to_paper_table": False,
              "protocol_notes": ["Uni-HOI test IDs/window/fps/repetition settings have not been verified against this run."]}
    if task in ("mixed-controls", "unconditioned"):
        report["status"] = "no_matching_paper_task_use_control_diagnostics"
        return report
    if task == "text-to-hoi":
        if not text_conditioned:
            raise ValueError("text-to-HOI requires actual text conditioning")
        report.update(status="requires_dataset_level_evaluator_features",
                      metrics=dict.fromkeys(TEXT_METRICS),
                      reason="Use evaluate_features with aligned embeddings from a trained HOI/text evaluator; raw poses or CLIP text alone cannot replace it.")
        return report
    condition = "object" if task == "object-to-human" else "human"
    if not np.allclose(array(controls.values[condition])[0, valid], np.asarray(record[condition])[indices], atol=1e-5, rtol=1e-5):
        raise ValueError("paired paper metrics require conditioning that matches the reference")
    for name in ("human", "object", "contact"):
        if array(states[name]).shape != array(controls.values[name]).shape:
            raise ValueError(f"{name} prediction/control shape mismatch")
    assets = assets or PaperAssets()
    vertices, faces, path = assets.mesh(record)
    pred_human, pred_obj = array(states["human"])[0, valid], array(states["object"])[0, valid]
    ref_human, ref_obj = np.asarray(record["human"])[indices], np.asarray(record["object"])[indices]
    report["details"].update(valid_frames=int(valid.sum()), window_frames=length, mesh_path=path,
                              timing_verified=bool(record.get("fps_verified_by_metadata", False)))
    if task == "object-to-human":
        if str(record["dataset"]) != "omomo":
            raise ValueError("Uni-HOI table 2 is OMOMO; no verified BEHAVE 24-joint evaluation adapter")
        offsets, scale = assets.omomo_metadata(record, indices)
        pred, ref = omomo_joints24(pred_human, offsets), omomo_joints24(ref_human, offsets)
        # The object trajectory is a fixed condition. Using a different predicted
        # object here would evaluate a different task and could conceal violations.
        if not np.allclose(pred_obj, ref_obj, atol=1e-5, rtol=1e-5):
            raise ValueError("object-to-human predictions must preserve the conditioned object")
        rot = rotation_matrices(ref_obj)
        tree = cKDTree(vertices)
        distances = []
        for joints in (pred, ref):
            local = np.einsum("tji,thj->thi", rot, joints[:, 22:] - ref_obj[:, None, :3]) / scale[:, None, None]
            distances.append(tree.query(local.reshape(-1, 3))[0].reshape(-1, 2) * scale[:, None])
        report["metrics"], counts = human_metrics(pred, ref, *distances)
        report["details"].update(counts, contact_threshold_m=.05, contact_support="either_middle_finger_base_per_frame",
                                  joint_decoding="SMPL_H_rest_offsets24_FK", MPJPE_alignment="pelvis_translation_only",
                                  HandJPE_alignment="world", c_percent_unit="fraction_0_to_1",
                                  object_scale="original_per_frame_scale")
    else:
        ps, rs = None, None
        if str(record["dataset"]) == "omomo":
            _, rs = assets.omomo_metadata(record, indices)
            ps = np.full(len(indices), float(record["source_object_scale"]))
            report["protocol_notes"].append("OMOMO human-to-object is an extension; table 3 reports BEHAVE/GRAB.")
        report["metrics"], details = object_metrics(pred_obj, ref_obj, vertices, faces,
                                                    chamfer_samples=chamfer_samples, seed=seed,
                                                    predicted_scale=ps, reference_scale=rs)
        report["details"].update(details, chamfer_definition="mean_pred_to_gt_L2_plus_mean_gt_to_pred_L2",
                                  surface_sampling="independent_area_uniform_each_surface_each_frame")
        report["protocol_notes"].append("Section 4.1 defines E_ch; table 3 says E_c. We report E_ch and separately the centroid error; no ambiguous E_c alias.")
    report["status"] = "computed_metric_definitions_aligned_protocol_not_fully_verified"
    return report


def add_asset_arguments(parser):
    parser.add_argument("--behave-objects", default="data/raw/behave/objects")
    parser.add_argument("--omomo-objects", default="OMOMO/data/captured_objects")
    parser.add_argument("--omomo-raw", default="OMOMO/data/test_diffusion_manip_seq_joints24.p")
    parser.add_argument("--omomo-source-fps", type=float, default=30., help="Explicit raw-timing assumption; not metadata verification")
    parser.add_argument("--chamfer-samples", type=int, default=10000)


def assets_from_args(args):
    return PaperAssets(args.behave_objects, args.omomo_objects, args.omomo_raw, args.omomo_source_fps)
