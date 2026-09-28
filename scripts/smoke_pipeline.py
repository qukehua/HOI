"""Check all seven generation modes plus mixed conditions on a REAL held-out record.

Checks program behavior only. A four-step checkpoint cannot establish motion quality.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from unified_hoi.controls import GENERATED_MODES, compile_controls, sample_controls
from unified_hoi.data import read_manifest
from unified_hoi.evaluate import evaluate_batch
from unified_hoi.runtime import load_model, write_json
from unified_hoi.sample import prepare_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/smoke_real_omomo/last.pt")
    parser.add_argument("--manifest", default="data/processed/omomo_smoke_test/manifest.jsonl")
    parser.add_argument("--output", default="runs/smoke_real_omomo/samples")
    args = parser.parse_args()
    torch.set_num_threads(4)
    records = [r for r in read_manifest(args.manifest) if r["split"] == "test"]
    if not records:
        raise ValueError("Smoke sampling needs a held-out test record")
    batch, stamps, metadata = prepare_batch(records[0]["path"], frames=8)
    if len(stamps) != 8:
        raise ValueError("Choose a reference with at least eight frames")
    model, diffusion, checkpoint = load_model(args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name in (*GENERATED_MODES, "mixed"):
        target = output / f"{name}.npz"
        if target.exists():
            raise FileExistsError(f"Choose a fresh --output: {target}")
        if name == "mixed":
            spec = json.loads(Path("examples/mixed_controls.json").read_text(encoding="utf-8"))
            controls = compile_controls(spec, batch, batch["valid_frames"])
        else:
            controls = sample_controls(batch, batch["valid_frames"], mode=name,
                                       generator=torch.Generator().manual_seed(7))
        result = diffusion.sample(model, batch, controls, steps=min(4, diffusion.steps), seed=7,
                                  projection_steps=3 if name == "mixed" else 0)
        for key, value in result.items():
            if not torch.isfinite(value).all():
                raise AssertionError(f"Non-finite output for {name}/{key}")
            mask = controls.masks[key]
            if not torch.equal(value[mask], controls.values[key][mask]):
                raise AssertionError(f"Observed features changed in {name}/{key}")
        metrics[name] = evaluate_batch(result, controls, batch)
        arrays = {key: value[0].detach().numpy() for key, value in result.items()}
        for key in result:
            arrays[key + "_observed"] = controls.values[key][0].numpy()
            arrays[key + "_mask"] = controls.masks[key][0].numpy()
        arrays.update(timestamps=stamps, fps=metadata["fps"], object_points=metadata["object_points"],
                      rest_offsets=metadata["rest_offsets"], sequence_id=metadata["sequence_id"],
                      fps_verified_by_metadata=metadata.get("fps_verified_by_metadata", False))
        np.savez_compressed(target, **arrays)
    report = {"program_checks_passed": True, "condition_modes_checked": 8,
              "checkpoint_step": checkpoint["step"], "reference": records[0]["path"],
              "reference_split": "test", "generation_quality_validated": False,
              "physical_execution_validated": False, "metrics": metrics}
    write_json(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
