"""Batch-evaluate the three HOI tasks on a val/test split."""
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
from .tasks import TASK_MODES, build_task_summary, evaluation_settings, report_key


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


def motion_text_table(evaluator, dataset, predictions, device, seed, view):
    """FID, R-precision Top-1/2/3 and Diversity for text-driven generations."""
    from .evaluate_motion_text import figure_table, score_motion_pairs
    generated = {}
    for key, predicted in predictions:
        if key in generated:
            raise ValueError(f"Duplicate motion-text window {key}")
        generated[key] = predicted
    pairs = []
    loader = DataLoader(dataset, batch_sampler=[[i] for i in range(len(dataset))],
                        generator=torch.Generator().manual_seed(0))
    for batch in loader:
        key = (batch["sequence_id"][0], int(batch["window_start"]))
        predicted = generated.get(key)
        if predicted is not None:
            pairs.append((batch, predicted))
    if len(pairs) < 2:
        raise ValueError("FID / R-precision needs at least two captioned generated windows")
    table = figure_table(score_motion_pairs(evaluator, pairs, device, seed=seed), view)
    table["evaluated_windows"] = len(pairs)
    return table


def benchmark(checkpoint, output, *, modes=None, split="val", seed=42, temperature=0., top_k=50,
              limit=None, device="cpu", dtype="bfloat16", profile="tridi",
              eval_fps=None, train_reference_fps=None, generation_frame="pelvis",
              body_models="data/smplx_models", refiner=None, motion_text_evaluator=None):
    if modes is None:
        modes = list(TASK_MODES)
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
    correction, cache_text = None, None
    if refiner:
        correction, refinement_saved = load_refiner(refiner, saved, digest_file(Path(checkpoint) / "metadata.json"),
                                                    geometry, device)
        cache_text = bool(refinement_saved["source"]["text_conditioned"])
        check_sampler(refinement_saved, temperature, top_k, cache_text)
    dataset = MotionDataset(config["manifest"], split, config["dataset"], geometry.support_hash, window)
    if not len(dataset):
        raise ValueError("No evaluation windows")
    evaluator_model, text_dataset, figure_view = None, None, config.get("fid_view", "joint")
    if motion_text_evaluator:
        from .motion_text import VIEWS, EvaluatorDataset, load_evaluator
        if "111" not in modes:
            raise ValueError("FID / R-precision / Diversity are the text-to-HOI metrics and need mode 111")
        if figure_view not in VIEWS or (limit is not None and limit < 2):
            raise ValueError("Motion-text table needs a known fid_view and at least two windows")
        evaluator_path = Path(motion_text_evaluator)
        if not evaluator_path.is_file():
            raise FileNotFoundError(f"Motion-text evaluator not found: {evaluator_path}")
        text_dataset = EvaluatorDataset(config["manifest"], split, config["dataset"], geometry.support_hash, window)
        evaluator_model, _ = load_evaluator(evaluator_path, config["dataset"], geometry.support_hash,
                                            text_dataset.records, device)
    plan = frame_plan(dataset.records, protocol["eval_fps"])
    output.mkdir(parents=True)
    reports, bags, ids, meshes = {}, {}, [], {}
    coarse_reports, figure_predictions = {}, []
    reference_digest = hashlib.sha256()
    total = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(total):
        record_index, start, end = dataset.windows[index]
        selected, frame_ids = window_selection(plan[record_index], start, end)
        if not len(selected):
            continue
        batch = move_batch(next(iter(DataLoader(dataset, batch_sampler=[[index]]))), device)
        caption = batch["text"][0]
        settings = [(mode, *row) for mode in modes for row in evaluation_settings(mode, caption)]
        if not settings:
            continue
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
        for mode, task, condition, prompt in settings:
            values, _ = complete(model, codecs, builder, batch, mode, prompt,
                                 seed + index * 1009 + (0 if condition == "with_text" else 17),
                                 temperature, top_k)
            key = report_key(mode, condition)
            refined = correction is not None and bool(str(prompt).strip()) == cache_text
            coarse_report = None
            if refined:
                coarse_states = {name: values[name][None] for name in MODALITIES}
                coarse_report = reconstruction(coarse_states, target, geometry, mode, vertices,
                                               ProbabilityDecoder(), .5, selected)
                coarse_reports.setdefault(key, []).append(coarse_report)
                with torch.no_grad():
                    values, _ = correction(values, mode, batch["object_bps"][0], batch["gender"][0],
                                            float(batch["fps"][0]), geometry,
                                            torch.tensor(vertices, device=device, dtype=torch.float32))
            states = {name: values[name][None] for name in ("human", "object", "contact")}
            report = reconstruction(states, target, geometry, mode, vertices, ProbabilityDecoder(), .5, selected)
            if evaluator_model is not None and task == "text_to_hoi":
                predicted = {name: values[name].detach().to("cpu")[None] for name in MODALITIES}
                predicted.update(valid_frames=torch.ones(1, values["human"].shape[0], dtype=torch.bool),
                                 fps=batch["fps"].detach().to("cpu"))
                figure_predictions.append(((batch["sequence_id"][0], start), predicted))
            report.update(mode=mode, condition=condition, task=task, refined=refined,
                          text_condition=bool(str(prompt).strip()), mode_description=MODE_DESCRIPTIONS[mode],
                          sequence_id=entry["sequence_id"], window_start=start, input_frames=n,
                          selected_local_indices=selected.tolist())
            if coarse_report is not None:
                report["coarse_metrics"] = coarse_report["metrics"]
            with (output / "per_clip.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
            reports.setdefault(key, []).append(report)
            if mode[0] == "1" or mode[1] == "1":
                kind = generation_feature_kind(mode)
                with torch.no_grad():
                    _, pred_joints = geometry(states["human"][:, valid], target["gender"])
                bag = bags.setdefault(key, {"kind": kind, "mode": mode, "generated": [], "reference": []})
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
               "sampler": "autoregressive_vq", "limit_windows": limit,
               "task_text": "text_to_hoi uses the caption; object_to_human is scored with and without it; human_to_object uses none",
               "support_sha256": geometry.support_hash, "protocol": protocol,
               "modes": {m: MODE_DESCRIPTIONS[m] for m in modes}, "evaluated_frames": len(ids),
               "evaluation_frames_sha256": fingerprint(sorted(ids)),
               "reference_values_sha256": reference_digest.hexdigest(), "reference_train_frames_sha256": None,
               "metrics": {}, "reconstruction": {}, "contact_micro": {}, "generation": {}}
    if refiner:
        summary.update(sampler="autoregressive_vq_continuous_refinement", refiner_sha256=digest_file(refiner),
                       refiner=str(refiner), refiner_text_conditioned=cache_text,
                       refiner_scope="settings whose text flag matches the cache",
                       coarse_reconstruction={}, coarse_contact_micro={})
        for mode, rows in coarse_reports.items():
            _, primary, micro = aggregate_reports(rows)
            summary["coarse_reconstruction"][mode] = primary
            summary["coarse_contact_micro"][mode] = micro
    aggregated = {}
    for key, rows in reports.items():
        metrics, primary, micro = aggregate_reports(rows)
        summary["metrics"][key], summary["reconstruction"][key], summary["contact_micro"][key] = metrics, primary, micro
        aggregated[key] = {"primary": primary, "micro": micro, "windows": len(rows),
                           "frames": int(sum(row["frames"] for row in rows))}
    train_cache = {}
    if bags:
        train = MotionDataset(config["manifest"], "train", config["dataset"], geometry.support_hash, window)
    for key, bag in bags.items():
        kind = bag["kind"]
        if kind not in train_cache:
            train_cache[kind] = collect_split_features(train, kind, geometry, protocol["train_reference_fps"],
                                                       protocol["generation_frame"], device)
        reference_train, train_hash = train_cache[kind]
        summary["reference_train_frames_sha256"] = train_hash
        scores = generation_metrics(np.concatenate(bag["generated"], 0), np.concatenate(bag["reference"], 0),
                                    reference_train)
        scores.update(feature_kind=kind, mode_description=MODE_DESCRIPTIONS[bag["mode"]])
        summary["generation"][key] = scores
    figure = None
    if evaluator_model is not None and figure_predictions:
        figure = motion_text_table(evaluator_model, text_dataset, figure_predictions, device, seed, figure_view)
        summary["motion_text"] = figure
        summary["motion_text_evaluator"] = str(motion_text_evaluator)
    summary["tasks"] = build_task_summary(aggregated, figure)
    _print_tasks(summary["tasks"])
    write_json(output / "summary.json", summary)
    return summary


def _print_tasks(tasks):
    text = tasks.get("text_to_hoi", {})
    if "FID" in text:
        precision = text["R_Precision"]
        print(f"text-to-HOI: FID={text['FID']:.4f} "
              f"R-Precision top1={precision['top1']:.4f} top2={precision['top2']:.4f} "
              f"top3={precision['top3']:.4f} Diversity={text['Diversity']:.4f} "
              f"(GT {text['Diversity_real']:.4f})", flush=True)
    human = tasks.get("object_to_human", {})
    for condition in ("without_text", "with_text"):
        row = human.get(condition)
        if not row:
            continue
        print(f"object-to-human {condition}: HandJPE={row['HandJPE']:.4f}m MPJPE={row['MPJPE']:.4f}m "
              f"C_prec={row['C_prec']:.4f} C_rec={row['C_rec']:.4f} C_acc={row['C_acc']:.4f} "
              f"c%={row['c_percent']:.4f}", flush=True)
    obj = tasks.get("human_to_object", {})
    if "Ec" in obj:
        print(f"human-to-object: Ec={obj['Ec']:.4f}m Ev2v={obj['Ev2v']:.4f}m", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    add_protocol_arguments(parser)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=None,
                        help="Default: 111 text-to-HOI, 101 object-to-human, 011 human-to-object")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--body-models", default="data/smplx_models")
    parser.add_argument("--refiner", help="Optional continuous refiner; also reports paired coarse reconstruction")
    parser.add_argument("--motion-text-evaluator",
                        help="Frozen evaluator last.pt. Adds FID, R-precision Top-1/2/3 and Diversity for mode 111")
    args = vars(parser.parse_args())
    torch.set_num_threads(4)
    summary = benchmark(**args)
    printed = {"output": args["output"], "tasks": {key: value for key, value in summary["tasks"].items()
                                                   if key != "protocol"}}
    print(json.dumps(printed))


if __name__ == "__main__":
    main()
