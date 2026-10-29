"""Batch-evaluate a token_ar checkpoint on a val/test split with the tridi_metrics_v2 protocol."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..mesh_geometry import load_mesh_asset
from ..runtime import move_batch, write_json
from . import MODALITIES, MODES
from .body import BodyGeometry, digest_file
from .checkpoint import load_run
from .codecs import load_codecs
from .data import MotionDataset
from .evaluate import ProbabilityDecoder
from .evaluation_protocol import (MODE_DESCRIPTIONS, add_protocol_arguments, aggregate_reports, fingerprint,
                                  frame_plan, protocol_options, window_selection)
from .metrics import generation_feature_kind, generation_features, generation_metrics, reconstruction
from .sample import complete
from .refinement import check_sampler, load_refiner


def collect_split_features(dataset, kind, geometry, target_fps, coordinate_frame, device):
    """Per-frame TriDi features from ground-truth windows of one split (no sampling)."""
    rows, frame_ids = [], []
    plan = frame_plan(dataset.records, target_fps)
    for index, (record_index, start, end) in enumerate(dataset.windows):
        selected, ids = window_selection(plan[record_index], start, end)
        if not len(selected):
            continue
        batch = move_batch(next(iter(DataLoader(dataset, batch_sampler=[[index]]))), device)
        valid = torch.as_tensor(selected, device=device)
        with torch.no_grad():
            _, joints = geometry(batch["human"][:, valid], batch["gender"])
        entry = dataset.records[record_index]
        vertices, _ = load_mesh_asset(str(Path(entry["object_mesh_path"]).resolve()))
        rows.append(generation_features(kind, joints=joints[0].cpu().numpy(),
                                        object_pose=batch["object"][0, valid].cpu().numpy(),
                                        vertices=vertices * entry["object_mesh_scale"],
                                        coordinate_frame=coordinate_frame))
        frame_ids.extend(ids)
    if not rows:
        raise ValueError("No windows for generation reference features")
    return np.concatenate(rows, 0), fingerprint(sorted(frame_ids))


def benchmark(checkpoint, output, *, modes=MODES, split="val", seed=42, temperature=0., top_k=50,
              text_conditioned=True, limit=None, device="cpu", dtype="bfloat16", profile="tridi",
              eval_fps=None, train_reference_fps=None, generation_frame="pelvis",
              body_models="data/smplx_models", refiner=None):
    if (not modes or any(m not in MODES for m in modes) or len(set(modes)) != len(modes)
            or split not in ("val", "test") or (limit is not None and limit < 1)):
        raise ValueError("Specify distinct H/O/C modes, a val/test split and a positive limit")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol = protocol_options(profile, eval_fps, train_reference_fps, generation_frame)
    device = torch.device(device)
    model, builder, saved = load_run(checkpoint, device, dtype)
    config = saved["config"]
    window = config.get("window", 120)
    codecs, _ = load_codecs(Path(checkpoint) / "codecs.pt", config["dataset"], saved["support_sha256"], device)
    geometry = BodyGeometry(config["dataset"], config["contact_support"], body_models).to(device)
    correction = None
    if refiner:
        correction, refinement_saved = load_refiner(refiner, saved, digest_file(Path(checkpoint) / "metadata.json"),
                                                    geometry, device)
        check_sampler(refinement_saved, temperature, top_k, text_conditioned)
    dataset = MotionDataset(config["manifest"], split, config["dataset"], geometry.support_hash, window)
    if not len(dataset):
        raise ValueError("No evaluation windows")
    plan = frame_plan(dataset.records, protocol["eval_fps"])
    output.mkdir(parents=True)
    reports, bags, ids, meshes = {}, {}, [], {}
    coarse_reports = {}
    reference_digest = hashlib.sha256()
    total = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(total):
        record_index, start, end = dataset.windows[index]
        selected, frame_ids = window_selection(plan[record_index], start, end)
        if not len(selected):
            continue
        batch = move_batch(next(iter(DataLoader(dataset, batch_sampler=[[index]]))), device)
        prompt = batch["text"][0] if text_conditioned else ""
        entry = dataset.records[record_index]
        mesh_path = str(Path(entry["object_mesh_path"]).resolve())
        if mesh_path not in meshes:
            vertices, _ = load_mesh_asset(mesh_path)
            meshes[mesh_path] = (vertices * entry["object_mesh_scale"], digest_file(mesh_path))
        vertices, mesh_hash = meshes[mesh_path]
        reference_digest.update(fingerprint([mesh_hash, entry["object_mesh_scale"]]).encode())
        n = end - start
        target = {"human": batch["human"][:, :n], "object": batch["object"][:, :n],
                  "surface_contact": batch["contact"][:, :n], "gender": batch["gender"],
                  "valid_frames": torch.ones(1, n, dtype=torch.bool, device=device)}
        valid = torch.as_tensor(selected, device=device)
        for key in ("human", "object", "surface_contact"):
            reference_digest.update(target[key][0, valid].cpu().numpy().astype("<f4").tobytes())
        ids.extend(frame_ids)
        with torch.no_grad():
            _, gt_joints = geometry(target["human"][:, valid], target["gender"])
        gt_joints, gt_pose = gt_joints[0].cpu().numpy(), target["object"][0, valid].cpu().numpy()
        for mode in modes:
            values, _ = complete(model, codecs, builder, batch, mode, prompt, seed + index * 1009,
                                 temperature, top_k)
            coarse_report = None
            if correction is not None:
                coarse_states = {key: values[key][None] for key in MODALITIES}
                coarse_report = reconstruction(coarse_states, target, geometry, mode, vertices,
                                               ProbabilityDecoder(), .5, selected)
                coarse_reports.setdefault(mode, []).append(coarse_report)
                with torch.no_grad():
                    values, _ = correction(values, mode, batch["object_bps"][0], batch["gender"][0],
                                            float(batch["fps"][0]), geometry,
                                            torch.tensor(vertices, device=device, dtype=torch.float32))
            states = {key: values[key][None] for key in ("human", "object", "contact")}
            report = reconstruction(states, target, geometry, mode, vertices, ProbabilityDecoder(), .5, selected)
            report.update(mode=mode, mode_description=MODE_DESCRIPTIONS[mode], sequence_id=entry["sequence_id"],
                          window_start=start, input_frames=n, selected_local_indices=selected.tolist())
            if coarse_report is not None:
                report["coarse_metrics"] = coarse_report["metrics"]
            with (output / "per_clip.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
            reports.setdefault(mode, []).append(report)
            if mode[0] == "1" or mode[1] == "1":
                kind = generation_feature_kind(mode)
                with torch.no_grad():
                    _, pred_joints = geometry(states["human"][:, valid], target["gender"])
                bag = bags.setdefault(mode, {"kind": kind, "generated": [], "reference": []})
                bag["generated"].append(generation_features(
                    kind, joints=pred_joints[0].cpu().numpy(), object_pose=states["object"][0, valid].cpu().numpy(),
                    vertices=vertices, coordinate_frame=protocol["generation_frame"]))
                bag["reference"].append(generation_features(
                    kind, joints=gt_joints, object_pose=gt_pose, vertices=vertices,
                    coordinate_frame=protocol["generation_frame"]))
        print(f"Evaluated window {index + 1}/{total}", flush=True)
    if not reports:
        raise ValueError("No frames selected; increase --limit or change --eval-fps")
    summary = {"dataset": config["dataset"], "checkpoint": str(checkpoint), "checkpoint_step": saved.get("step"),
               "split": split, "window": window, "seed": seed, "temperature": temperature, "top_k": top_k,
               "sampler": "autoregressive_vq", "text_conditioned": text_conditioned, "limit_windows": limit,
               "support_sha256": geometry.support_hash, "protocol": protocol,
               "modes": {m: MODE_DESCRIPTIONS[m] for m in modes}, "evaluated_frames": len(ids),
               "evaluation_frames_sha256": fingerprint(sorted(ids)),
               "reference_values_sha256": reference_digest.hexdigest(), "reference_train_frames_sha256": None,
               "metrics": {}, "reconstruction": {}, "contact_micro": {}, "generation": {}}
    if refiner:
        summary.update(sampler="autoregressive_vq_continuous_refinement", refiner_sha256=digest_file(refiner),
                       refiner=str(refiner), coarse_reconstruction={}, coarse_contact_micro={})
        for mode, rows in coarse_reports.items():
            _, primary, micro = aggregate_reports(rows)
            summary["coarse_reconstruction"][mode] = primary
            summary["coarse_contact_micro"][mode] = micro
    for mode, rows in reports.items():
        metrics, primary, micro = aggregate_reports(rows)
        summary["metrics"][mode], summary["reconstruction"][mode], summary["contact_micro"][mode] = metrics, primary, micro
    train_cache = {}
    if bags:
        train = MotionDataset(config["manifest"], "train", config["dataset"], geometry.support_hash, window)
    for mode, bag in bags.items():
        kind = bag["kind"]
        if kind not in train_cache:
            train_cache[kind] = collect_split_features(train, kind, geometry, protocol["train_reference_fps"],
                                                       protocol["generation_frame"], device)
        reference_train, train_hash = train_cache[kind]
        summary["reference_train_frames_sha256"] = train_hash
        scores = generation_metrics(np.concatenate(bag["generated"], 0), np.concatenate(bag["reference"], 0),
                                    reference_train)
        scores.update(feature_kind=kind, mode_description=MODE_DESCRIPTIONS[mode])
        summary["generation"][mode] = scores
        print(f"mode {mode} ({MODE_DESCRIPTIONS[mode]}): 1-NNA={scores['1-NNA']:.4f} "
              f"COV={scores['COV']:.4f} MMD={scores['MMD']:.4f}", flush=True)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    add_protocol_arguments(parser)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--no-text", dest="text_conditioned", action="store_false")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--body-models", default="data/smplx_models")
    parser.add_argument("--refiner", help="Optional continuous refiner; also reports paired coarse reconstruction")
    args = vars(parser.parse_args())
    torch.set_num_threads(4)
    summary = benchmark(**args)
    print(json.dumps({"output": args["output"], "reconstruction": summary["reconstruction"]}))


if __name__ == "__main__":
    main()
