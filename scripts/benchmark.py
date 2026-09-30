"""Uni-HOI task evaluation on held-out windows, with separate control diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from unified_hoi.controls import GENERATED_MODES, PARTIAL_PATTERNS, sample_controls
from unified_hoi.data import HOIDataset, load_record
from unified_hoi.evaluate import evaluate_batch
from unified_hoi.paper_metrics import MODE_TASKS, add_asset_arguments, assets_from_args, evaluate_paper_sequence
from unified_hoi.runtime import device_for, load_model, move_batch, write_json


def _accumulate(groups, group, metrics, frames):
    for key, value in metrics.items():
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            groups.setdefault(group, {}).setdefault(key, []).append((value, frames))


def _summarize(groups):
    return {group: {key: {"macro_mean_per_window": float(np.mean([v for v, _ in rows])),
                         "frame_weighted_mean": float(np.average([v for v, _ in rows], weights=[w for _, w in rows])),
                         "supported_windows": len(rows)} for key, rows in metrics.items()}
            for group, metrics in groups.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("uni-hoi", "diagnostics"), default="uni-hoi")
    parser.add_argument("--modes", nargs="+", choices=list(GENERATED_MODES) + list(PARTIAL_PATTERNS),
                        help="Default uni-hoi: OMOMO 101, BEHAVE 011. Diagnostics: seven modes + mixed.")
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--projection-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-conditioned", action="store_true", help="Require real cached text for each evaluated window")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--samples-per-window", type=int, default=1,
                        help="For paper 101 only: select one best sample by MPJPE and use it for ALL metrics")
    signatures = ["none", "H", "O", "C", "H+O", "H+C", "O+C", "H+O+C"]
    parser.add_argument("--observed-signatures", nargs="+", choices=signatures,
                        help="Restrict actual observed combinations for control diagnostics")
    parser.add_argument("--limit", type=int, help="Explicit diagnostic subset; absent evaluates all test windows")
    parser.add_argument("--device", default="auto")
    add_asset_arguments(parser)
    args = parser.parse_args(argv)
    if (args.limit is not None and args.limit < 1) or args.samples_per_window < 1 or args.chamfer_samples < 1:
        raise ValueError("limit, samples-per-window and chamfer-samples must be positive")
    if args.profile == "uni-hoi" and args.modes and "111" in args.modes and not args.text_conditioned:
        raise ValueError("Uni-HOI table 1 needs --text-conditioned and real cached text; unconditional 111 is a different task")
    if args.samples_per_window > 1 and (args.profile != "uni-hoi" or args.modes not in (None, ["101"])):
        raise ValueError("best-of-K is supported only for uni-hoi object-to-human (101)")
    output = Path(args.output)
    if any((output / name).exists() for name in ("per_clip.jsonl", "summary.json", "feature_inputs.jsonl")):
        raise FileExistsError("Benchmark output exists; choose a new directory")
    torch.set_num_threads(4)
    device = device_for(args.device)
    model, diffusion, _ = load_model(args.checkpoint, device)
    dataset = HOIDataset(args.manifest, "test", args.window, args.window, model.config.text_dim)
    if not len(dataset):
        raise ValueError("No test windows")
    assets = assets_from_args(args)
    output.mkdir(parents=True, exist_ok=True)
    primary, diagnostics, grouped, signature_counts, task_status = {}, {}, {}, {}, {}
    feature_inputs = []
    excluded = [s for s in signatures if args.observed_signatures and s not in args.observed_signatures]
    with (output / "per_clip.jsonl").open("w", encoding="utf-8") as stream:
        for index, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
            if args.limit is not None and index >= args.limit:
                break
            dataset_name = batch["dataset"][0]
            modes = args.modes
            if modes is None:
                if args.profile == "diagnostics":
                    modes = list(GENERATED_MODES) + ["mixed"]
                else:
                    modes = {"omomo": ["101"], "behave": ["011"]}.get(dataset_name)
                    if modes is None:
                        raise ValueError(f"No Uni-HOI task mapping for {dataset_name}")
            if args.samples_per_window > 1 and modes != ["101"]:
                raise ValueError("best-of-K cannot be applied to human-to-object or text-to-HOI")
            if args.text_conditioned:
                if not bool(batch["text_available"].all()):
                    raise ValueError(f"Missing verified text embedding for {batch['sequence_id'][0]}; cannot run a text-conditioned task")
            else:
                batch["text_features"].zero_()
                batch["text_available"].zero_()
            record_index, start, end = dataset.windows[index]
            entry = dataset.records[record_index]
            record = load_record(entry["path"])
            batch = move_batch(batch, device)
            frames = int(batch["valid_frames"].sum())
            for mode in modes:
                seed = args.seed + index * 1009
                control = sample_controls(batch, batch["valid_frames"], mode=mode,
                                          generator=torch.Generator().manual_seed(seed), holdout_signatures=excluded)
                task = MODE_TASKS.get(mode)
                paper_enabled = args.profile == "uni-hoi" and task is not None
                if paper_enabled and task != "text-to-hoi":
                    assets.mesh(record)
                    if dataset_name == "omomo":
                        assets.omomo_metadata(record, np.arange(start, end))
                candidates = []
                sampling_seconds = evaluation_seconds = 0.
                for sample_index in range(args.samples_per_window):
                    sample_seed = seed + sample_index * 1000003
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    before = time.perf_counter()
                    result = diffusion.sample(model, batch, control, steps=args.steps, seed=sample_seed,
                                              projection_steps=args.projection_steps)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    sampling_seconds += time.perf_counter() - before
                    before = time.perf_counter()
                    paper = evaluate_paper_sequence(result, control, record, assets=assets, start=start, task=task,
                                                    text_conditioned=args.text_conditioned,
                                                    chamfer_samples=args.chamfer_samples, seed=seed) if paper_enabled else None
                    evaluation_seconds += time.perf_counter() - before
                    candidates.append((result, paper, sample_seed))
                chosen = int(np.argmin([p[1]["metrics"]["MPJPE_cm"] for p in candidates])) if len(candidates) > 1 else 0
                result, paper, selected_seed = candidates[chosen]
                diagnostic = evaluate_batch(result, control, batch)
                diagnostic["sample_and_projection_seconds"] = sampling_seconds
                diagnostic["paper_geometry_evaluation_seconds"] = evaluation_seconds
                signature = control.signatures[0]
                dgroup = f"{dataset_name}:{mode}"
                signature_counts.setdefault(dgroup, {})[signature] = signature_counts.setdefault(dgroup, {}).get(signature, 0) + 1
                row = {"dataset": dataset_name, "sequence_id": batch["sequence_id"][0], "window_start": start,
                       "mode": mode, "observed_signature": signature, "seed": selected_seed,
                       "samples_per_window": args.samples_per_window, "selected_sample_index": chosen,
                       "selection": "minimum_MPJPE_same_sample_for_all_metrics" if len(candidates) > 1 else "single_sample",
                       "text_conditioned": args.text_conditioned, "diagnostics": diagnostic,
                       "metrics": paper["metrics"] if paper else diagnostic}
                if paper:
                    row["paper_evaluation"] = paper
                    group = f"{dataset_name}:{task}:{'with_text' if args.text_conditioned else 'without_text'}"
                    _accumulate(primary, group, paper["metrics"], frames)
                    task_status.setdefault(group, {"status": paper["status"], "windows": 0,
                                                  "metric_names": list(paper["metrics"]),
                                                  "protocol_notes": paper["protocol_notes"]})["windows"] += 1
                elif args.profile == "diagnostics":
                    _accumulate(primary, dgroup, diagnostic, frames)
                _accumulate(diagnostics, dgroup, diagnostic, frames)
                _accumulate(grouped, dgroup + ":" + signature, diagnostic, frames)
                if args.save_predictions or (paper and task == "text-to-hoi"):
                    folder = output / "predictions"
                    folder.mkdir(exist_ok=True)
                    destination = folder / f"{index:06d}_{mode}.npz"
                    arrays = {k: v[0, :frames].detach().cpu().numpy() for k, v in result.items()}
                    arrays.update(fps=record["fps"], timestamps=record["timestamps"][start:end],
                                  text_conditioned=np.bool_(args.text_conditioned), dataset=np.array(dataset_name),
                                  sequence_id=record["sequence_id"], text=record["text"])
                    for name in ("human", "object", "contact"):
                        arrays[name + "_observed"] = control.values[name][0, :frames].cpu().numpy()
                        arrays[name + "_mask"] = control.masks[name][0, :frames].cpu().numpy()
                    np.savez_compressed(destination, **arrays)
                    row["prediction"] = destination.relative_to(output).as_posix()
                    if paper and task == "text-to-hoi":
                        feature_inputs.append({"sample_id": f"{dataset_name}:{batch['sequence_id'][0]}:{start}:{selected_seed}",
                                               "dataset": dataset_name, "prediction": row["prediction"],
                                               "prediction_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                                               "reference": str(entry["path"]), "reference_start": start,
                                               "reference_end": end, "text": str(record["text"])})
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            if (index + 1) % 20 == 0:
                print(f"Evaluated {index + 1}/{min(args.limit or len(dataset), len(dataset))} test windows", flush=True)
    if feature_inputs:
        (output / "feature_inputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in feature_inputs), encoding="utf-8")
    write_json(output / "summary.json", {"config": vars(args), "aggregation": "separate dataset/task groups; macro per window and frame-weighted means",
               "observed_signatures": signature_counts, "metrics": _summarize(primary), "task_status": task_status,
               "diagnostics_by_dataset_mode": _summarize(diagnostics), "diagnostics_by_signature": _summarize(grouped),
               "directly_comparable_to_paper_table": False, "physical_execution_validated": False,
               "protocol_notes": "Metric definitions follow Uni-HOI/referenced code; splits, 10fps windows, repetition and trained text evaluator are not verified against Uni-HOI.",
               "per_clip_results": "per_clip.jsonl", "feature_inputs": "feature_inputs.jsonl" if feature_inputs else None})
    print(output / "summary.json")


if __name__ == "__main__":
    main()
