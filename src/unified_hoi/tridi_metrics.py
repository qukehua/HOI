"""TriDi-aligned reconstruction / generation metrics for HOI predictions.

Metric definitions follow TriDi ``tridi/utils/metrics/{reconstruction,generation}.py``
(ICCV 2025). Contact is evaluated on representation joints (default 22) rather than
SMPL decimated vertices, because this package does not store body meshes.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

from .geometry import rotation6d_to_matrix
from .paper_metrics import array, rotation_matrices


CONTACT_THRESHOLD_M = 0.05


def _pose_mesh(pose: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """pose [T,9] (xyz + rot6d rows) → posed vertices [T,V,3]."""
    rot = rotation_matrices(pose)
    return vertices[None] @ np.transpose(rot, (0, 2, 1)) + pose[:, None, :3]


def _object_pose12(pose: np.ndarray) -> np.ndarray:
    """TriDi object_pose feature: flattened R (9) + center (3)."""
    rot = rotation_matrices(pose).reshape(len(pose), 9)
    return np.concatenate([rot, pose[:, :3]], axis=-1).astype(np.float64)


def _human_joints_feature(joints: np.ndarray) -> np.ndarray:
    """TriDi human_joints feature: all joints except root, flattened."""
    return joints[:, 1:].reshape(len(joints), -1).astype(np.float64)


def compute_similarity_transform(source, target):
    """TriDi similarity (sR, t) aligning source→target; points are (N,3)."""
    s1, s2 = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    mu1, mu2 = s1.mean(0, keepdims=True), s2.mean(0, keepdims=True)
    x1, x2 = s1 - mu1, s2 - mu2
    var1 = np.sum(x1 ** 2)
    k = x1.T @ x2
    u, _, vh = np.linalg.svd(k)
    v = vh.T
    z = np.eye(3)
    z[-1, -1] = np.sign(np.linalg.det(u @ v.T))
    r = v @ z @ u.T
    scale = np.trace(r @ k) / var1 if var1 > 0 else 1.0
    t = mu2 - scale * (mu1 @ r.T)
    return scale * (s1 @ r.T) + t


def get_mpjpe(pred_joints, gt_joints):
    pred = pred_joints - pred_joints[..., :1, :]
    gt = gt_joints - gt_joints[..., :1, :]
    return np.linalg.norm(gt - pred, axis=-1).mean(-1)


def get_mpjpe_pa(pred_joints, gt_joints):
    pred = pred_joints - pred_joints[..., :1, :]
    gt = gt_joints - gt_joints[..., :1, :]
    aligned = np.stack([compute_similarity_transform(pred[t], gt[t]) for t in range(len(pred))])
    return np.linalg.norm(gt - aligned, axis=-1).mean(-1)


def _joint_object_contact(joints: np.ndarray, obj_vertices: np.ndarray,
                          threshold: float = CONTACT_THRESHOLD_M) -> np.ndarray:
    """Per-frame joint contact mask [T,J] via min distance to posed object verts."""
    masks = []
    for t in range(len(joints)):
        dist = cKDTree(obj_vertices[t]).query(joints[t])[0]
        masks.append(dist <= threshold)
    return np.asarray(masks, dtype=bool)


def contact_agreement(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Per-frame fraction of joints with matching contact labels [T]."""
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("contact masks must match")
    return (pred_mask == gt_mask).mean(axis=-1)


def tridi_object_reconstruction(pred_obj, ref_obj, ref_human, pred_contact, vertices,
                                contact_threshold: float = CONTACT_THRESHOLD_M):
    """Mode with generated object (010/011): OBJ_V2V / CENTER / CONTACT_*."""
    pred_v, ref_v = _pose_mesh(pred_obj, vertices), _pose_mesh(ref_obj, vertices)
    joints = np.asarray(ref_human[..., :3], dtype=np.float64)
    gt_contact = _joint_object_contact(joints, ref_v, contact_threshold)
    mesh_contact = _joint_object_contact(joints, pred_v, contact_threshold)
    diffused = np.asarray(pred_contact) >= 0.5
    if diffused.shape != gt_contact.shape:
        raise ValueError("diffused contact shape must match joints")
    return {
        "OBJ_V2V": float(np.linalg.norm(pred_v - ref_v, axis=-1).mean()),
        "OBJ_CENTER": float(np.linalg.norm(pred_v.mean(1) - ref_v.mean(1), axis=-1).mean()),
        "OBJ_CONTACT_MESHES": float(100 * contact_agreement(mesh_contact, gt_contact).mean()),
        "OBJ_CONTACT_DIFFUSED": float(100 * contact_agreement(diffused, gt_contact).mean()),
    }, {
        "per_frame_OBJ_V2V": np.linalg.norm(pred_v - ref_v, axis=-1).mean(-1),
        "per_frame_OBJ_CENTER": np.linalg.norm(pred_v.mean(1) - ref_v.mean(1), axis=-1),
        "object_pose12": _object_pose12(pred_obj),
        "reference_object_pose12": _object_pose12(ref_obj),
    }


def tridi_human_reconstruction(pred_human, ref_human, ref_obj, pred_contact, vertices,
                               contact_threshold: float = CONTACT_THRESHOLD_M):
    """Mode with generated human (100/101): MPJPE / PA / CONTACT_* (metres)."""
    pred_j = np.asarray(pred_human[..., :3], dtype=np.float64)
    ref_j = np.asarray(ref_human[..., :3], dtype=np.float64)
    ref_v = _pose_mesh(ref_obj, vertices)
    gt_contact = _joint_object_contact(ref_j, ref_v, contact_threshold)
    mesh_contact = _joint_object_contact(pred_j, ref_v, contact_threshold)
    diffused = np.asarray(pred_contact) >= 0.5
    if diffused.shape != gt_contact.shape:
        raise ValueError("diffused contact shape must match joints")
    mpjpe = get_mpjpe(pred_j, ref_j)
    mpjpe_pa = get_mpjpe_pa(pred_j, ref_j)
    return {
        "MPJPE": float(mpjpe.mean()),
        "MPJPE_PA": float(mpjpe_pa.mean()),
        "SBJ_CONTACT_MESHES": float(100 * contact_agreement(mesh_contact, gt_contact).mean()),
        "SBJ_CONTACT_DIFFUSED": float(100 * contact_agreement(diffused, gt_contact).mean()),
    }, {
        "per_frame_MPJPE": mpjpe,
        "per_frame_MPJPE_PA": mpjpe_pa,
        "human_joints_feat": _human_joints_feature(pred_j),
        "reference_human_joints_feat": _human_joints_feature(ref_j),
    }


def evaluate_tridi_sequence(states, controls, record, *, assets, start=0, mode: str,
                            contact_threshold: float = CONTACT_THRESHOLD_M):
    """Evaluate one window under TriDi reconstruction metrics for ``mode``."""
    controls.validate()
    valid = array(controls.valid_frames)[0]
    indices = start + np.flatnonzero(valid)
    pred_h = array(states["human"])[0, valid]
    pred_o = array(states["object"])[0, valid]
    pred_c = array(states["contact"])[0, valid]
    ref_h = np.asarray(record["human"])[indices]
    ref_o = np.asarray(record["object"])[indices]
    vertices, _, path = assets.mesh(record)
    generated = tuple(bit == "1" for bit in mode[:3])
    report = {"protocol": "TriDi-aligned", "mode": mode, "mesh_path": path,
              "contact_support": "representation_joints", "contact_threshold_m": contact_threshold,
              "metrics": {}, "features": {}}
    if generated[1] and not generated[0]:
        metrics, details = tridi_object_reconstruction(
            pred_o, ref_o, ref_h, pred_c, vertices, contact_threshold)
        report["metrics"], report["features"] = metrics, {
            "kind": "object_pose", "generated": details["object_pose12"],
            "reference": details["reference_object_pose12"],
        }
        report["target"] = "obj"
    elif generated[0] and not generated[1]:
        metrics, details = tridi_human_reconstruction(
            pred_h, ref_h, ref_o, pred_c, vertices, contact_threshold)
        report["metrics"], report["features"] = metrics, {
            "kind": "human_joints", "generated": details["human_joints_feat"],
            "reference": details["reference_human_joints_feat"],
        }
        report["target"] = "sbj"
    else:
        raise ValueError(f"TriDi profile currently supports object-only or human-only generation modes, got {mode}")
    report["status"] = "computed"
    return report


def _nn_distances(train: np.ndarray, query: np.ndarray, k: int = 1):
    if len(train) < 1 or len(query) < 1:
        raise ValueError("generation metrics require nonempty feature sets")
    model = NearestNeighbors(n_neighbors=min(k, len(train)), algorithm="auto", metric="euclidean")
    model.fit(train)
    distances, indices = model.kneighbors(query)
    return distances, indices


def generation_metrics(generated: np.ndarray, reference_test: np.ndarray,
                       reference_train: np.ndarray | None = None) -> dict[str, float]:
    """TriDi 1-NNA / COV / MMD / SD on frame-level feature matrices."""
    gen, ref = np.asarray(generated, dtype=np.float64), np.asarray(reference_test, dtype=np.float64)
    train = ref if reference_train is None else np.asarray(reference_train, dtype=np.float64)
    # COV: unique NN of S_g in S_r / |S_r|
    _, idx = _nn_distances(ref, gen, k=1)
    cov = len(np.unique(idx[:, 0])) / len(ref)
    # MMD: mean NN distance of S_r → S_g
    dist_r2g, _ = _nn_distances(gen, ref, k=1)
    mmd = float(dist_r2g[:, 0].mean())
    # SD: mean NN distance of S_g → S_train
    dist_g2t, _ = _nn_distances(train, gen, k=1)
    sd = float(dist_g2t[:, 0].mean())
    # 1-NNA on pooled set with leave-self-out via k=2
    pooled = np.concatenate([gen, ref], axis=0)
    labels = np.array([0] * len(gen) + [1] * len(ref))
    dist, idx = _nn_distances(pooled, pooled, k=2)
    nn_labels = labels[idx[:, 1]]
    nna = float((nn_labels == labels).mean())
    return {"1-NNA": nna, "COV": cov, "MMD": mmd, "SD": sd}


def collect_split_features(manifest: str, split: str, window: int, kind: str,
                           text_dim: int = 512) -> np.ndarray:
    """Build TriDi-style features from a manifest split (for generation refs)."""
    from .data import HOIDataset
    dataset = HOIDataset(manifest, split, window, window, text_dim)
    chunks = []
    for index in range(len(dataset)):
        record_index, start, end = dataset.windows[index]
        record = __import__("unified_hoi.data", fromlist=["load_record"]).load_record(
            dataset.records[record_index]["path"])
        if kind == "object_pose":
            chunks.append(_object_pose12(np.asarray(record["object"][start:end])))
        elif kind == "human_joints":
            chunks.append(_human_joints_feature(np.asarray(record["human"][start:end, :, :3])))
        else:
            raise ValueError(kind)
    if not chunks:
        raise ValueError(f"no {split} windows for generation reference features")
    return np.concatenate(chunks, axis=0)
