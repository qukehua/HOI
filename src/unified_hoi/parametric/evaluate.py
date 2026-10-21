"""Native SMPL-H / SMPL-X metrics with matched 73-joint semantics and surface contacts."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.utils.data import DataLoader

from ..data import read_manifest
from ..mesh_geometry import load_mesh_asset
from ..geometry import rotation6d_to_matrix
from ..runtime import move_batch, write_json
from .body import BodyGeometry, legacy_to_parameters, human_dim
from .data import ParametricDataset, load_record
from .diffusion import controls_for


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
    tp, fp, fn = int((pred & target).sum()), int((pred & ~target).sum()), int((~pred & target).sum())
    return {"accuracy_percent": float(100 * (pred == target).mean()),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "TP": tp, "FP": fp, "FN": fn}


def mesh_contacts(body_vertices, object_pose, object_vertices):
    rotation = rotation_matrices(object_pose)
    local = np.einsum("tji,tkj->tki", rotation, body_vertices - object_pose[:, None, :3])
    distance = cKDTree(object_vertices).query(local.reshape(-1, 3), workers=4)[0].reshape(local.shape[:-1])
    return distance <= .05


@torch.no_grad()
def reconstruction(states, batch, geometry, mode, vertices, codec=None, threshold=.5):
    if mode not in ("100", "010", "001", "110", "101", "011", "111"):
        raise ValueError("Invalid generated-bit evaluation mode")
    for key, bit in zip(("human", "object"), mode):
        if bit == "0" and not torch.allclose(states[key], batch[key], atol=1e-5, rtol=0):
            raise ValueError(f"Observed {key} was altered before evaluation")
    valid = batch["valid_frames"][0]
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
        details["CONTACT_DIFFUSED"] = contact_scores(predicted_contact, gt_contact)
        prefix = "OBJ" if mode[1] == "1" and mode[0] == "0" else "SBJ"
        scores[prefix + "_CONTACT_DIFFUSED"] = details["CONTACT_DIFFUSED"]["accuracy_percent"]
    for k, value in details.items():
        for name in ("precision", "recall", "f1"):
            scores[k + "_" + name.upper()] = value[name]
    return {"metrics": scores, "details": details, "frames": len(po), "dataset": geometry.dataset,
            "body_model": "smplh" if geometry.dataset == "behave" else "smplx",
            "evaluation_joints": 73, "contact_vertices": 690, "support_sha256": geometry.support_hash,
            "units": "metres; contacts percent; rotation degrees",
            "smplx_joint_mapping": "body22+hand30+vertex_landmarks21; face-only joints excluded" if geometry.dataset == "omomo" else None,
            "directly_comparable_to_paper_table": False}


def evaluate_legacy(prediction, reference, manifest, support, hands, mode, start=0, output=None):
    """Score old checkpoints with fixed source-hand convention, without filling generated poses from GT."""
    with np.load(prediction, allow_pickle=False) as archive:
        pred = dict(archive)
    ref = load_record(reference)
    dataset = str(ref["dataset"])
    geometry = BodyGeometry(dataset, support)
    load_record(reference, dataset, geometry.support_hash)
    if str(ref["body_model_sha256"]) != geometry.layers[str(ref["gender"])].asset_hash:
        raise ValueError("Legacy adapter requires the reference's exact native body assets")
    from .diffusion import MODES
    if mode not in MODES:
        raise ValueError("Invalid legacy adaptation mode")
    with np.load(hands, allow_pickle=False) as data:
        if str(data["dataset"]) != dataset:
            raise ValueError("Cross-dataset hand convention")
        hand = data["rotations"]
    entries = read_manifest(manifest)
    if {e["dataset"] for e in entries} != {dataset}:
        raise ValueError("Mixed datasets are prohibited for native evaluation")
    entry = next((e for e in entries if Path(e["path"]).resolve() == Path(reference).resolve()), None)
    if entry is None:
        raise ValueError("Reference not present in native manifest")
    frames = len(pred["human"])
    if start < 0 or start + frames > len(ref["human_params"]):
        raise ValueError("Legacy prediction/reference window mismatch")
    if "timestamps" in pred and not np.allclose(pred["timestamps"], ref["timestamps"][start:start + frames], atol=1e-6):
        raise ValueError("Legacy prediction timestamps do not match reference window")
    if mode[0] == "0":
        with torch.no_grad():
            _, reference_joints = geometry(torch.from_numpy(ref["human_params"][start:start + frames])[None],
                                          torch.tensor([int(str(ref["gender"]) == "male")]))
        if not np.allclose(pred["human"][..., :3], reference_joints[0, :, :22].numpy(), atol=.001, rtol=0):
            raise ValueError("Legacy observed human does not match native reference")
        human = ref["human_params"][start:start + frames]
    else:
        # Legacy model was conditioned on source skeleton/shape: disclose that protocol.
        converted = {"human": pred["human"], "betas": pred.get("betas", ref["betas"]),
                     "gender": pred.get("gender", ref["gender"])}
        human = legacy_to_parameters(converted, geometry, hand)
    states = {"human": torch.from_numpy(human)[None], "object": torch.from_numpy(pred["object"])[None]}
    batch = {"human": torch.from_numpy(ref["human_params"][start:start + frames])[None],
             "object": torch.from_numpy(ref["object"][start:start + frames])[None],
             "gender": torch.tensor([int(str(ref["gender"]) == "male")]),
             "valid_frames": torch.ones(1, frames, dtype=torch.bool)}
    vertices, _ = load_mesh_asset(str(Path(entry["object_mesh_path"]).resolve()))
    report = reconstruction(states, batch, geometry, mode, vertices * entry["object_mesh_scale"])
    report.update(adapter="legacy22_body_plus_fixed_hand_convention", source_shape_conditioned=True,
                  contact_diffused_available=False, hand_source=str(ref["hand_source"]),
                  generated_body_source="global_rotations_and_root_position; nonroot xyz are not IK targets")
    if output:
        write_json(output, report)
    return report


def benchmark(checkpoint, manifest, output, *, modes=("010", "100"), window=120, steps=50,
              seed=42, device="cpu", limit=None, text_conditioned=False, save_predictions=False):
    from .train import load_run, prepare_batch
    from .diffusion import MODES
    if not modes or any(mode not in MODES for mode in modes) or (limit is not None and limit < 1):
        raise ValueError("Specify valid native modes and a positive limit")
    model, diffusion, geometry, saved = load_run(checkpoint, device)
    config = saved["config"]
    dataset = ParametricDataset(manifest, "test", config["dataset"], geometry.support_hash, window,
                                text_condition=text_conditioned, text_dim=model.config.text_dim)
    if not len(dataset):
        raise ValueError("No test windows")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    reports, features = [], {}
    for index, batch in enumerate(DataLoader(dataset, batch_size=1)):
        if limit is not None and index >= limit:
            break
        batch = prepare_batch(model, move_batch(batch, device), geometry, diffusion.steps)
        if text_conditioned and not batch["text_available"].all():
            raise ValueError("Missing text condition")
        entry = dataset.records[int(batch["record_index"][0])]
        vertices, _ = load_mesh_asset(str(Path(entry["object_mesh_path"]).resolve()))
        vertices = vertices * entry["object_mesh_scale"]
        for mode in modes:
            masks = controls_for(batch, config["dataset"], mode)
            state = diffusion.sample(model, batch, masks, steps, seed + index * 1009)
            report = reconstruction(state, batch, geometry, mode, vertices, model.codec, float(model.contact_threshold))
            report.update(mode=mode, sequence_id=batch["sequence_id"][0], window_start=int(batch["window_start"][0]))
            reports.append(report)
            with (output / "per_clip.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
            if save_predictions:
                folder = output / mode
                folder.mkdir(exist_ok=True)
                n = report["frames"]
                record = load_record(entry["path"])
                start = int(batch["window_start"][0])
                np.savez_compressed(folder / f"{index:06d}.npz", human_params=state["human"][0, :n].cpu().numpy(),
                                    object=state["object"][0, :n].cpu().numpy(), contact_latent=state["contact"][0, :n].cpu().numpy(),
                                    contact_probability=model.codec.decode(state["contact"])[0, :n].cpu().numpy(),
                                    timestamps=record["timestamps"][start:start + n], dataset=config["dataset"],
                                    representation="parametric_v1", mode=mode, gender=record["gender"],
                                    window_start=start, sequence_id=record["sequence_id"],
                                    body_model_type=record["body_model_type"],
                                    body_model_sha256=record["body_model_sha256"],
                                    codec_sha256=saved["codec_metadata"]["file_sha256"],
                                    support_sha256=geometry.support_hash)
            features.setdefault(mode, []).append(report)
        print(f"Evaluated native window {index + 1}/{min(limit or len(dataset), len(dataset))}", flush=True)
    summary = {"dataset": config["dataset"], "checkpoint": str(checkpoint), "steps": steps, "seed": seed,
               "text_conditioned": text_conditioned, "single_sample": True, "support_sha256": geometry.support_hash,
               "codec_origin": saved["codec_metadata"]["origin"], "metrics": {}, "directly_comparable_to_paper_table": False}
    for mode, rows in features.items():
        summary["metrics"][mode] = {key: {"macro_mean_per_window": float(np.mean([r["metrics"][key] for r in rows])),
                                             "frame_weighted_mean": float(np.average([r["metrics"][key] for r in rows],
                                                                    weights=[r["frames"] for r in rows]))}
                                     for key in rows[0]["metrics"]}
    write_json(output / "summary.json", summary)
    return summary


def evaluate_prediction(prediction, checkpoint, reference, manifest, output=None, start=None, device="cpu"):
    """Re-evaluate a saved native prediction with the same codec, assets and reference window."""
    from .train import load_run
    model, _, geometry, saved = load_run(checkpoint, device)
    with np.load(prediction, allow_pickle=False) as archive:
        pred = dict(archive)
    dataset = geometry.dataset
    ref = load_record(reference, dataset, geometry.support_hash)
    for key, expected in (("representation", "parametric_v1"), ("dataset", dataset),
                          ("support_sha256", geometry.support_hash), ("gender", str(ref["gender"])),
                          ("body_model_sha256", str(ref["body_model_sha256"])),
                          ("codec_sha256", saved["codec_metadata"]["file_sha256"]),
                          ("sequence_id", str(ref["sequence_id"]))):
        if str(pred.get(key)) != expected:
            raise ValueError(f"Prediction provenance mismatch: {key}")
    if geometry.layers[str(ref["gender"])].asset_hash != str(ref["body_model_sha256"]):
        raise ValueError("Reference body assets differ from checkpoint")
    entries = read_manifest(manifest)
    if {e["dataset"] for e in entries} != {dataset}:
        raise ValueError("Mixed datasets are prohibited for native evaluation")
    entry = next((e for e in entries if Path(e["path"]).resolve() == Path(reference).resolve()), None)
    if entry is None:
        raise ValueError("Reference must belong to the native manifest")
    mode = str(pred["mode"])
    start = int(pred["window_start"]) if start is None else start
    n = len(pred["human_params"])
    if start < 0 or n < 1 or start + n > len(ref["human_params"]):
        raise ValueError("Prediction/reference window mismatch")
    for key, shape in (("human_params", (n, human_dim(dataset))), ("object", (n, 9)),
                       ("contact_latent", (n, 128)), ("timestamps", (n,))):
        if pred[key].shape != shape or not np.isfinite(pred[key]).all():
            raise ValueError(f"Invalid native prediction {key}")
    if not np.allclose(pred["timestamps"], ref["timestamps"][start:start + n], atol=1e-6, rtol=0):
        raise ValueError("Prediction timestamps do not match reference window")
    states = {k: torch.as_tensor(pred[source], dtype=torch.float32, device=device)[None]
              for k, source in (("human", "human_params"), ("object", "object"), ("contact", "contact_latent"))}
    batch = {k: torch.as_tensor(ref[source][start:start + n], dtype=torch.float32, device=device)[None]
             for k, source in (("human", "human_params"), ("object", "object"), ("surface_contact", "surface_contact"))}
    batch.update(gender=torch.tensor([int(str(ref["gender"]) == "male")], device=device),
                 valid_frames=torch.ones(1, n, dtype=torch.bool, device=device))
    with torch.no_grad():
        batch["contact"] = model.codec.encode(batch["surface_contact"])
        masks = controls_for(batch, dataset, mode)
        for key, mask in masks.items():
            if not torch.allclose(states[key][mask], batch[key][mask], atol=1e-5, rtol=0):
                raise ValueError(f"Observed {key} changed in saved prediction")
    vertices, _ = load_mesh_asset(str(Path(entry["object_mesh_path"]).resolve()))
    report = reconstruction(states, batch, geometry, mode, vertices * entry["object_mesh_scale"],
                            model.codec, float(model.contact_threshold))
    report.update(mode=mode, window_start=start, sequence_id=str(ref["sequence_id"]))
    if output:
        write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--manifest", required=True)
    bench.add_argument("--output", required=True)
    bench.add_argument("--modes", nargs="+", default=["010", "100"])
    bench.add_argument("--window", type=int, default=120)
    bench.add_argument("--steps", type=int, default=50)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--limit", type=int)
    bench.add_argument("--text-conditioned", action="store_true")
    bench.add_argument("--save-predictions", action="store_true")
    old = sub.add_parser("legacy")
    for name in ("prediction", "reference", "manifest", "support", "hands", "mode", "output"):
        old.add_argument("--" + name, required=True)
    old.add_argument("--start", type=int, default=0)
    native = sub.add_parser("prediction")
    for name in ("prediction", "checkpoint", "reference", "manifest", "output"):
        native.add_argument("--" + name, required=True)
    native.add_argument("--start", type=int)
    native.add_argument("--device", default="cpu")
    args = vars(parser.parse_args())
    command = args.pop("command")
    torch.set_num_threads(4)
    from ..runtime import device_for
    if "device" in args:
        args["device"] = device_for(args["device"])
    result = {"benchmark": benchmark, "legacy": evaluate_legacy, "prediction": evaluate_prediction}[command](**args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
