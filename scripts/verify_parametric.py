"""CPU integration verification, never a formal training run.

Uses one source clip per split, full configured denoiser width/depth, two updates,
two DDIM steps, both reconstruction directions, saved-result replay and mesh export.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from unified_hoi.data import read_manifest
from unified_hoi.parametric.body import decode_full
from unified_hoi.parametric.evaluate import benchmark, evaluate_prediction
from unified_hoi.parametric.sample import sample
from unified_hoi.parametric.train import run_training
from unified_hoi.runtime import write_json
from unified_hoi.smpl_mesh import record_to_smpl


def verify(dataset, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = read_manifest(Path(f"data/processed/{dataset}_parametric.jsonl").resolve())
    subset = [next(row for row in rows if row["split"] == split) for split in ("train", "val", "test")]
    output.mkdir(parents=True)
    manifest = output / "subset.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in subset), encoding="utf-8")
    config = yaml.safe_load(Path(f"configs/train_{dataset}.yaml").read_text(encoding="utf-8"))
    config.update(manifest=str(manifest), output=str(output / "training"), device="cpu", cpu_threads=2,
                  window=8, stride=8, batch_size=1, max_steps=2, diffusion_steps=8, geometry_max_t=8,
                  geometry_frames=2, validate_every=1, validation_batches=1, validation_steps=2,
                  save_every=1, log_every=1, use_swanlab=False)
    checkpoint = run_training(config)
    result = benchmark(checkpoint, manifest, output / "benchmark", window=8, steps=2, limit=1,
                       save_predictions=True)
    reference = subset[2]["path"]
    replay = {}
    reports = [json.loads(line) for line in (output / "benchmark/per_clip.jsonl").read_text().splitlines()]
    for mode in ("010", "100"):
        replay[mode] = evaluate_prediction(output / "benchmark" / mode / "000000.npz", checkpoint,
                                           reference, manifest, output / f"replay_{mode}.json")
        expected = next(row for row in reports if row["mode"] == mode)
        for key, value in expected["metrics"].items():
            np.testing.assert_allclose(replay[mode]["metrics"][key], value, rtol=0, atol=1e-8)
    contact_path = sample(checkpoint, manifest, reference, output / "sample_001.npz", "001", frames=8, steps=2)
    evaluate_prediction(contact_path, checkpoint, reference, manifest, output / "replay_001.json")
    with np.load(contact_path, allow_pickle=False) as archive:
        prediction = dict(archive)
    vertices, faces, _ = record_to_smpl(prediction)
    full, _, joints = decode_full(prediction["human_params"], dataset, str(prediction["gender"]))
    np.testing.assert_allclose(vertices, full, atol=1e-7, rtol=0)
    np.testing.assert_allclose(prediction["joints73"], joints, atol=2e-6, rtol=0)
    np.savez_compressed(output / "full_mesh.npz", vertices=vertices, faces=faces, joints73=joints)
    # A different native dataset/codec must never be accepted as a reference.
    bad = dict(prediction, dataset="omomo" if dataset == "behave" else "behave")
    np.savez_compressed(output / "wrong_dataset.npz", **bad)
    try:
        evaluate_prediction(output / "wrong_dataset.npz", checkpoint, reference, manifest)
    except ValueError as exc:
        isolation_error = str(exc)
    else:
        raise AssertionError("Cross-dataset prediction accepted")
    report = {"dataset": dataset, "cpu_only": True, "formal_training": False, "training_steps": 2,
              "sample_steps": 2, "window": 8, "modes_evaluated": ["010", "100", "001"],
              "saved_metrics_replay_exact": True, "full_mesh_shape": list(vertices.shape),
              "surface_decode_matches_full_native": True, "cross_dataset_rejection": isolation_error,
              "training_files_only": True, "benchmark": result}
    write_json(output / "verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "benchmark"}), flush=True)


def verify_resume(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = read_manifest(Path("data/processed/behave_parametric.jsonl").resolve())
    subset = [next(row for row in rows if row["split"] == split) for split in ("train", "val")]
    output.mkdir(parents=True)
    manifest = output / "subset.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in subset), encoding="utf-8")
    config = yaml.safe_load(Path("configs/train_behave.yaml").read_text())
    config.update(manifest=str(manifest), device="cpu", cpu_threads=2, window=3, stride=3,
                  batch_size=1, max_steps=2, diffusion_steps=8, geometry_max_t=8, geometry_frames=1,
                  validate_every=1, validation_batches=1, validation_steps=1, use_swanlab=False,
                  save_every=1, log_every=1)
    config["model"].update(width=24, heads=4, layers=1)
    full = run_training(config | {"output": str(output / "full")})
    first = run_training(config | {"max_steps": 1, "output": str(output / "resumed")})
    resumed = run_training(config | {"output": str(output / "resumed")}, resume=first)
    a, b = [torch.load(path, weights_only=False, map_location="cpu") for path in (full, resumed)]
    maximum = 0.
    for key in ("model", "ema"):
        for name, value in a[key].items():
            torch.testing.assert_close(value, b[key][name], rtol=0, atol=0)
            maximum = max(maximum, float((value - b[key][name]).abs().max()))
    report = {"model_and_ema_bitwise_equal": True, "max_difference": maximum,
              "continuous_steps": 2, "resumed_after_step": 1, "cpu_only": True}
    write_json(output / "verification.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("behave", "omomo"))
    parser.add_argument("--resume-test", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.resume_test:
        verify_resume(args.output)
    elif args.dataset:
        verify(args.dataset, args.output)
    else:
        parser.error("Specify --dataset or --resume-test")
