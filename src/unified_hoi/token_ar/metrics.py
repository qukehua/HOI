"""Copied native reconstruction metrics; independent from diffusion implementation."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from ..geometry import rotation6d_to_matrix


def rotation_matrices(state):
    """Reject degenerate encodings instead of silently using the model's fallback."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape[-1] != 9 or not np.isfinite(state).all():
        raise ValueError("pose states must be finite [...,9]")
    a, b = state[..., 3:6], state[..., 6:9]
    if (np.linalg.norm(a, axis=-1) < 1e-7).any() or (np.linalg.norm(np.cross(a, b), axis=-1) < 1e-7).any():
        raise ValueError("paper geometry metrics require nondegenerate rotations")
    return rotation6d_to_matrix(torch.from_numpy(state[..., 3:])).numpy()



def similarity(source, target):
    x, y = source - source.mean(0), target - target.mean(0)
    u, _, vh = np.linalg.svd(x.T @ y)
    z = np.eye(3)
    z[-1, -1] = np.sign(np.linalg.det(vh.T @ u.T))
    rotation = vh.T @ z @ u.T
    scale = np.trace(rotation @ x.T @ y) / max(float((x * x).sum()), 1e-12)
    return scale * x @ rotation.T + target.mean(0)



def contact_scores(pred, target):
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    if pred.shape != target.shape or not pred.size:
        raise ValueError("Contact predictions and targets must have the same nonempty shape")
    tp, fp, fn = int((pred & target).sum()), int((pred & ~target).sum()), int((~pred & target).sum())
    return {"accuracy_percent": float(100 * (pred == target).mean()),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "TP": tp, "FP": fp, "FN": fn,
            "TN": int((~pred & ~target).sum())}



def mesh_contacts(body_vertices, object_pose, object_vertices):
    rotation = rotation_matrices(object_pose)
    local = np.einsum("tji,tkj->tki", rotation, body_vertices - object_pose[:, None, :3])
    distance = cKDTree(object_vertices).query(local.reshape(-1, 3), workers=4)[0].reshape(local.shape[:-1])
    return distance <= .05



@torch.no_grad()
def reconstruction(states, batch, geometry, mode, vertices, codec=None, threshold=.5, frame_indices=None):
    if mode not in ("100", "010", "001", "110", "101", "011", "111"):
        raise ValueError("Invalid generated-bit evaluation mode")
    for key, bit in zip(("human", "object"), mode):
        if bit == "0" and not torch.allclose(states[key], batch[key], atol=1e-5, rtol=0):
            raise ValueError(f"Observed {key} was altered before evaluation")
    valid = batch["valid_frames"][0]
    if frame_indices is not None:
        selected = torch.as_tensor(frame_indices, device=valid.device, dtype=torch.long)
        if (not len(selected) or selected.min() < 0 or selected.max() >= len(valid)
                or len(selected.unique()) != len(selected) or not valid[selected].all()):
            raise ValueError("Invalid or padded evaluation frame indices")
        valid = torch.zeros_like(valid).scatter_(0, selected, True)
    gender = batch["gender"]
    pv, pj = geometry(states["human"][:, valid], gender)
    gv, gj = geometry(batch["human"][:, valid], gender)
    pv, pj, gv, gj = (v[0].detach().cpu().numpy().astype(np.float64) for v in (pv, pj, gv, gj))
    po = states["object"][0, valid].cpu().numpy()
    go = batch["object"][0, valid].cpu().numpy()
    gt_contact = mesh_contacts(gv, go, vertices)
    scores, details = {}, {}
    if mode[0] == "1":
        a, b = pj - pj[:, :1], gj - gj[:, :1]
        scores["MPJPE"] = float(np.linalg.norm(a - b, axis=-1).mean())
        scores["MPJPE_PA"] = float(np.mean([np.linalg.norm(similarity(x, y) - y, axis=-1).mean() for x, y in zip(a, b)]))
        contact = mesh_contacts(pv, po, vertices)
        details["SBJ_CONTACT"] = contact_scores(contact, gt_contact)
        scores["SBJ_CONTACT_MESHES"] = details["SBJ_CONTACT"]["accuracy_percent"]
    if mode[1] == "1":
        rp, rg = rotation_matrices(po), rotation_matrices(go)
        # Bound full-object vertex memory even for high-resolution OMOMO templates.
        v2v, center, angles = [], [], []
        for i in range(len(po)):
            pred = vertices @ rp[i].T + po[i, :3]
            target = vertices @ rg[i].T + go[i, :3]
            v2v.append(np.linalg.norm(pred - target, axis=-1).mean())
            center.append(np.linalg.norm(pred.mean(0) - target.mean(0)))
            angles.append(np.degrees(np.arccos(np.clip((np.trace(rp[i] @ rg[i].T) - 1) / 2, -1, 1))))
        scores.update(OBJ_V2V=float(np.mean(v2v)), OBJ_CENTER=float(np.mean(center)),
                      OBJ_ROTATION_DEG=float(np.mean(angles)))
        contact = mesh_contacts(pv, po, vertices)
        details["OBJ_CONTACT"] = contact_scores(contact, gt_contact)
        scores["OBJ_CONTACT_MESHES"] = details["OBJ_CONTACT"]["accuracy_percent"]
    if codec is not None and "contact" in states:
        if mode[2] == "0":
            # As in TriDi: observed contact uses the actual surface target, not AE reconstruction.
            predicted_contact = batch["surface_contact"][0, valid].cpu().numpy() >= .5
        else:
            predicted_contact = codec.decode(states["contact"])[0, valid].cpu().numpy() >= threshold
        details["CONTACT_TOKEN"] = contact_scores(predicted_contact, gt_contact)
        prefix = "OBJ" if mode[1] == "1" and mode[0] == "0" else "SBJ"
        scores[prefix + "_CONTACT_TOKEN"] = details["CONTACT_TOKEN"]["accuracy_percent"]
    for k, value in details.items():
        for name in ("precision", "recall", "f1"):
            scores[k + "_" + name.upper()] = value[name]
    scores["CONTACT_ALL_NEGATIVE_ACCURACY"] = float(100 * (~gt_contact).mean())
    return {"metrics": scores, "details": details, "frames": len(po), "dataset": geometry.dataset,
            "body_model": "smplh" if geometry.dataset == "behave" else "smplx",
            "evaluation_joints": 73, "contact_vertices": 690, "support_sha256": geometry.support_hash,
            "units": "metres; contacts percent; rotation degrees",
            "smplx_joint_mapping": "body22+hand30+vertex_landmarks21; face-only joints excluded" if geometry.dataset == "omomo" else None,
            "directly_comparable_to_paper_table": False}

