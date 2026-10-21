"""Held-out evaluation. BEHAVE defaults to TriDi; other datasets default to Uni-HOI."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from unified_hoi.controls import (GENERATED_MODES, describe_mode, mode_to_target_name, parse_mode,
                                  sample_controls)
from unified_hoi.data import HOIDataset, load_record
from unified_hoi.body_metadata import body_metadata
from unified_hoi.body_geometry import BODY_GEOMETRY_KEYS
from unified_hoi.evaluate import evaluate_batch
from unified_hoi.paper_metrics import MODE_TASKS, add_asset_arguments, assets_from_args, evaluate_paper_sequence
from unified_hoi.runtime import device_for, load_model, move_batch, write_json

# Used when --modes is omitted. BEHAVE 011 is object generation; OMOMO 101 is human generation.
DEFAULT_DATASET_MODES = {"omomo": ["101"], "behave": ["011"]}


def _accumulate(groups, group, metrics, frames):
    for key, value in metrics.items():
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            groups.setdefault(group, {}).setdefault(key, []).append((value, frames))


def _summarize(groups):
    return {group: {key: {"macro_mean_per_window": float(np.mean([v for v, _ in rows])),
                         "frame_weighted_mean": float(np.average([v for v, _ in rows], weights=[w for _, w in rows])),
                         "supported_windows": len(rows)} for key, rows in metrics.items()}
            for group, metrics in groups.items()}


def _tridi_unsupported(modes):
    return [m for m in modes if m not in GENERATED_MODES or m.count("1") < 1
            or (m[0] == "1" and m[1] == "1")]


def _resolved_profile(requested, dataset_name):
    """``auto`` scores BEHAVE with TriDi and every other dataset with Uni-HOI."""
    if requested != "auto":
        return requested
    return "tridi" if dataset_name == "behave" else "uni-hoi"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("auto", "tridi", "uni-hoi", "diagnostics"), default="auto",
                        help="auto (default): BEHAVE uses tridi, other datasets use uni-hoi. "
                             "Native parametric checkpoints: both datasets use 73-joint/surface reconstruction metrics. "
                             "tridi: Reconstruction (OBJ_*/MPJPE/contacts) + Generation "
                             "(1-NNA/COV/MMD/SD). uni-hoi: paper task metrics. diagnostics: controls only.")
    parser.add_argument("--modes", nargs="+", type=parse_mode,
                        help="TriDi-style H/O/C modes (1=generate, 0=condition). "
                             "Accepts 011, sample_011, human-to-object, ... "
                             "Default: OMOMO 101, BEHAVE 011. Diagnostics: seven modes + mixed.")
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--projection-steps", type=int, default=0)
    parser.add_argument("--ik-steps", type=int, default=90)
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
    checkpoint_config = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("config", {})
    if checkpoint_config.get("representation") == "parametric_v1":
        from unified_hoi.parametric.evaluate import benchmark as native_benchmark
        if args.profile not in ("auto", "tridi") or args.projection_steps or args.samples_per_window != 1 or args.observed_signatures:
            raise ValueError("Native benchmark uses isolated surface metrics, one sample and full H/O/C modes; "
                             "legacy projection/profile/signature options are unsupported")
        torch.set_num_threads(4)
        return native_benchmark(args.checkpoint, args.manifest, args.output, modes=args.modes or ("010", "100"),
                                window=args.window, steps=args.steps, seed=args.seed, device=device_for(args.device),
                                limit=args.limit, text_conditioned=args.text_conditioned,
                                save_predictions=args.save_predictions)
    if (args.limit is not None and args.limit < 1) or args.samples_per_window < 1 or args.chamfer_samples < 1:
        raise ValueError("limit, samples-per-window and chamfer-samples must be positive")
    if args.profile == "uni-hoi" and args.modes and "111" in args.modes and not args.text_conditioned:
        raise ValueError("Uni-HOI table 1 needs --text-conditioned and real cached text; unconditional 111 is a different task")
    if args.samples_per_window > 1 and args.profile not in ("uni-hoi", "auto"):
        raise ValueError("best-of-K is supported only for uni-hoi object-to-human (101)")
    if args.samples_per_window > 1 and args.modes not in (None, ["101"]):
        raise ValueError("best-of-K is supported only for uni-hoi object-to-human (101)")
    if args.profile == "tridi" and args.modes:
        unsupported = _tridi_unsupported(args.modes)
        if unsupported:
            raise ValueError(f"tridi profile supports single-target modes 010/011/100/101; got {unsupported}")
    tridi_mod = None

    def load_tridi():
        nonlocal tridi_mod
        if tridi_mod is None:
            # Optional TriDi dependencies must not block Uni-HOI/mesh diagnostics.
            from unified_hoi import tridi_metrics
            tridi_mod = tridi_metrics
        return tridi_mod
    output = Path(args.output)
    if any((output / name).exists() for name in ("per_clip.jsonl", "summary.json", "feature_inputs.jsonl")):
        raise FileExistsError("Benchmark output exists; choose a new directory")
    torch.set_num_threads(4)
    device = device_for(args.device)
    model, diffusion, checkpoint = load_model(args.checkpoint, device)
    dataset = HOIDataset(args.manifest, "test", args.window, args.window, model.config.text_dim,
                         geometry_encoder=model.config.geometry_encoder,
                         body_geometry_version=checkpoint.get("config", {}).get("body_geometry_version"))
    if not len(dataset):
        raise ValueError("No test windows")
    assets = assets_from_args(args)
    output.mkdir(parents=True, exist_ok=True)
    primary, diagnostics, grouped, signature_counts, task_status = {}, {}, {}, {}, {}
    feature_inputs = []
    tridi_features = {}
    excluded = [s for s in signatures if args.observed_signatures and s not in args.observed_signatures]
    with (output / "per_clip.jsonl").open("w", encoding="utf-8") as stream:
        for index, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
            if args.limit is not None and index >= args.limit:
                break
            dataset_name = batch["dataset"][0]
            profile = _resolved_profile(args.profile, dataset_name)
            modes = args.modes
            if modes is None:
                if profile == "diagnostics":
                    modes = list(GENERATED_MODES) + ["mixed"]
                else:
                    modes = DEFAULT_DATASET_MODES.get(dataset_name)
                    if modes is None:
                        raise ValueError(f"No default task mapping for dataset {dataset_name}")
            if profile == "tridi":
                load_tridi()
                unsupported = _tridi_unsupported(modes)
                if unsupported:
                    raise ValueError(f"tridi profile supports single-target modes 010/011/100/101; got {unsupported}")
            if profile == "uni-hoi" and "111" in modes and not args.text_conditioned:
                raise ValueError("Uni-HOI table 1 needs --text-conditioned and real cached text; unconditional 111 is a different task")
            if args.samples_per_window > 1 and (profile != "uni-hoi" or modes != ["101"]):
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
                paper_enabled = profile == "uni-hoi" and task is not None
                tridi_enabled = profile == "tridi"
                if (paper_enabled and task not in (None, "text-to-hoi", "object-contact-to-human")) or tridi_enabled:
                    assets.mesh(record)
                    if dataset_name == "omomo" and paper_enabled:
                        assets.omomo_metadata(record, np.arange(start, end))
                candidates = []
                sampling_seconds = evaluation_seconds = 0.
                for sample_index in range(args.samples_per_window):
                    sample_seed = seed + sample_index * 1000003
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    before = time.perf_counter()
                    result = diffusion.sample(model, batch, control, steps=args.steps, seed=sample_seed,
                                              projection_steps=args.projection_steps, ik_steps=args.ik_steps)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    sampling_seconds += time.perf_counter() - before
                    before = time.perf_counter()
                    if paper_enabled:
                        paper = evaluate_paper_sequence(result, control, record, assets=assets, start=start, task=task,
                                                        text_conditioned=args.text_conditioned,
                                                        chamfer_samples=args.chamfer_samples, seed=seed)
                    elif tridi_enabled:
                        paper = tridi_mod.evaluate_tridi_sequence(result, control, record, assets=assets, start=start, mode=mode)
                    else:
                        paper = None
                    evaluation_seconds += time.perf_counter() - before
                    candidates.append((result, paper, sample_seed))
                if len(candidates) > 1:
                    key = "MPJPE_cm" if "MPJPE_cm" in candidates[0][1]["metrics"] else "MPJPE"
                    chosen = int(np.argmin([p[1]["metrics"][key] for p in candidates]))
                else:
                    chosen = 0
                result, paper, selected_seed = candidates[chosen]
                diagnostic = evaluate_batch(result, control, batch)
                diagnostic["sample_and_projection_seconds"] = sampling_seconds
                diagnostic["paper_geometry_evaluation_seconds"] = evaluation_seconds
                signature = control.signatures[0]
                dgroup = f"{dataset_name}:{mode}"
                signature_counts.setdefault(dgroup, {})[signature] = signature_counts.setdefault(dgroup, {}).get(signature, 0) + 1
                row = {"dataset": dataset_name, "sequence_id": batch["sequence_id"][0], "window_start": start,
                       "mode": mode, "mode_description": describe_mode(mode),
                       "sampling_target": mode_to_target_name(mode) if mode in GENERATED_MODES else None,
                       "task": task, "observed_signature": signature, "seed": selected_seed,
                       "samples_per_window": args.samples_per_window, "selected_sample_index": chosen,
                       "selection": "minimum_MPJPE_same_sample_for_all_metrics" if len(candidates) > 1 else "single_sample",
                       "text_conditioned": args.text_conditioned, "metric_profile": profile,
                       "diagnostics": diagnostic, "metrics": paper["metrics"] if paper else diagnostic}
                if paper:
                    row["paper_evaluation" if not tridi_enabled else "tridi_evaluation"] = {
                        k: v for k, v in paper.items() if k != "features"
                    }
                    group = (f"{dataset_name}:{mode}" if tridi_enabled
                             else f"{dataset_name}:{task}:{'with_text' if args.text_conditioned else 'without_text'}")
                    _accumulate(primary, group, paper["metrics"], frames)
                    task_status.setdefault(group, {"status": paper.get("status"), "windows": 0,
                                                  "metric_names": list(paper["metrics"]),
                                                  "protocol": paper.get("protocol"),
                                                  "protocol_notes": paper.get("protocol_notes", [])})["windows"] += 1
                    if tridi_enabled and "features" in paper:
                        bags = tridi_features.setdefault(mode, {"generated": [], "reference": []})
                        bags["generated"].append(paper["features"]["generated"])
                        bags["reference"].append(paper["features"]["reference"])
                        bags["kind"] = paper["features"]["kind"]
                elif profile == "diagnostics":
                    _accumulate(primary, dgroup, diagnostic, frames)
                _accumulate(diagnostics, dgroup, diagnostic, frames)
                _accumulate(grouped, dgroup + ":" + signature, diagnostic, frames)
                if args.save_predictions or (paper and task == "text-to-hoi"):
                    folder = output / "predictions"
                    folder.mkdir(exist_ok=True)
                    destination = folder / f"{index:06d}_{mode}.npz"
                    arrays = {k: v[0, :frames].detach().cpu().numpy() for k, v in result.items()}
                    arrays.update(body_metadata(record))
                    arrays.update({k: record[k] for k in BODY_GEOMETRY_KEYS if k in record})
                    arrays["human_position_decoding"] = np.array("root_rotation_FK")
                    arrays.update(rest_offsets=record["rest_offsets"], subject_id=record.get("subject_id", ""),
                                  fps=record["fps"], timestamps=record["timestamps"][start:end],
                                  text_conditioned=np.bool_(args.text_conditioned), dataset=np.array(dataset_name),
                                  sequence_id=record["sequence_id"], text=record["text"])
                    if model.config.geometry_encoder == "mesh_bps":
                        arrays.update(object_points=batch["object_points"][0].cpu().numpy(),
                                      object_bps=batch["object_bps"][0].cpu().numpy(),
                                      object_geometry_is_mesh=np.bool_(True), object_geometry_version=np.int32(1))
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
    generation = {}
    if any(bags.get("generated") for bags in tridi_features.values()):
        print("Computing TriDi generation metrics...", flush=True)
        for mode, bags in tridi_features.items():
            if not bags["generated"]:
                continue
            gen = np.concatenate(bags["generated"], axis=0)
            ref = np.concatenate(bags["reference"], axis=0)
            kind = bags["kind"]
            train = tridi_mod.collect_split_features(args.manifest, "train", args.window, kind, model.config.text_dim)
            generation[mode] = tridi_mod.generation_metrics(gen, ref, train)
            print(f"  mode {mode}: {generation[mode]}", flush=True)
    if feature_inputs:
        (output / "feature_inputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in feature_inputs), encoding="utf-8")
    if args.profile == "tridi":
        notes = ("TriDi-aligned reconstruction + generation; contact on representation joints "
                 "(not SMPL decimated verts). Units: metres for MPJPE/OBJ_*; contact as percent.")
    elif args.profile == "auto":
        notes = ("Default metrics are per dataset: BEHAVE uses TriDi reconstruction + generation "
                 "(mode 011: OBJ_V2V / OBJ_CENTER / CONTACT_* unless --modes is set); "
                 "other datasets use Uni-HOI. Pass --profile uni-hoi to score BEHAVE with E_ch/E_v2v.")
    else:
        notes = ("Metric definitions follow Uni-HOI/referenced code; splits, 10fps windows, repetition "
                 "and trained text evaluator are not verified against Uni-HOI.")
    write_json(output / "summary.json", {"config": vars(args), "aggregation": "separate dataset/task groups; macro per window and frame-weighted means",
               "observed_signatures": signature_counts, "metrics": _summarize(primary),
               "generation": generation, "task_status": task_status,
               "diagnostics_by_dataset_mode": _summarize(diagnostics), "diagnostics_by_signature": _summarize(grouped),
               "directly_comparable_to_paper_table": False, "physical_execution_validated": False,
               "protocol_notes": notes,
               "per_clip_results": "per_clip.jsonl", "feature_inputs": "feature_inputs.jsonl" if feature_inputs else None})
    print(output / "summary.json")


if __name__ == "__main__":
    main()
