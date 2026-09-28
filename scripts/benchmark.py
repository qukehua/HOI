"""Evaluate deterministic controls across test windows; save every result and macro means."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from unified_hoi.controls import GENERATED_MODES, PARTIAL_PATTERNS, sample_controls
from unified_hoi.data import HOIDataset
from unified_hoi.evaluate import evaluate_batch
from unified_hoi.runtime import device_for, load_model, move_batch, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", nargs="+", default=list(GENERATED_MODES) + ["mixed"],
                        choices=list(GENERATED_MODES) + list(PARTIAL_PATTERNS))
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--projection-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    signatures = ["none", "H", "O", "C", "H+O", "H+C", "O+C", "H+O+C"]
    parser.add_argument("--observed-signatures", nargs="+", choices=signatures,
                        help="Evaluate only these actual observed combinations, e.g. mixed H+O held-out controls")
    parser.add_argument("--limit", type=int, help="Explicit small diagnostic subset; absent evaluates all test windows")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "per_clip.jsonl").exists():
        raise FileExistsError("Benchmark output exists; choose a new directory")
    torch.set_num_threads(4)
    device = device_for(args.device)
    model, diffusion, _ = load_model(args.checkpoint, device)
    dataset = HOIDataset(args.manifest, "test", args.window, args.window, model.config.text_dim)
    if not len(dataset):
        raise ValueError("No test windows")
    values = {mode: {} for mode in args.modes}
    observed_signatures = {mode: {} for mode in args.modes}
    grouped = {}
    excluded = [s for s in signatures if args.observed_signatures and s not in args.observed_signatures]
    with (output / "per_clip.jsonl").open("w", encoding="utf-8") as stream:
        for index, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
            if args.limit is not None and index >= args.limit:
                break
            batch = move_batch(batch, device)
            for mode in args.modes:
                seed = args.seed + index * 1009
                control = sample_controls(batch, batch["valid_frames"], mode=mode,
                                          generator=torch.Generator().manual_seed(seed), holdout_signatures=excluded)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                before = time.perf_counter()
                result = diffusion.sample(model, batch, control, steps=args.steps, seed=seed,
                                          projection_steps=args.projection_steps)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - before
                report = evaluate_batch(result, control, batch)
                report["sample_and_projection_seconds"] = seconds
                signature = control.signatures[0]
                observed_signatures[mode][signature] = observed_signatures[mode].get(signature, 0) + 1
                row = {"sequence_id": batch["sequence_id"][0], "window_start": int(batch["window_start"][0]),
                       "mode": mode, "observed_signature": signature, "seed": seed, "metrics": report}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                for key, value in report.items():
                    if isinstance(value, (float, int)) and not isinstance(value, bool):
                        values[mode].setdefault(key, []).append(value)
                        grouped.setdefault(mode + ":" + signature, {}).setdefault(key, []).append(value)
            if (index + 1) % 20 == 0:
                print(f"Evaluated {index + 1}/{min(args.limit or len(dataset), len(dataset))} test windows", flush=True)
    summary = {mode: {key: {"macro_mean_per_window": float(np.mean(value)), "supported_windows": len(value)}
                      for key, value in records.items()} for mode, records in values.items()}
    by_signature = {group: {key: {"macro_mean_per_window": float(np.mean(value)), "supported_windows": len(value)}
                            for key, value in records.items()} for group, records in grouped.items()}
    write_json(output / "summary.json", {"config": vars(args), "aggregation": "macro mean over supported windows",
               "observed_signatures": observed_signatures, "metrics": summary, "metrics_by_signature": by_signature,
               "physical_execution_validated": False, "per_clip_results": "per_clip.jsonl"})
    print(output / "summary.json")


if __name__ == "__main__":
    main()
