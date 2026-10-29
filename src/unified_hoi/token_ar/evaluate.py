"""Evaluate a token_ar prediction using copied native geometry metric formulas."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ..data import read_manifest
from ..mesh_geometry import load_mesh_asset
from ..runtime import write_json
from . import ARCHITECTURE, MODALITIES, MODES
from .body import BodyGeometry, digest_file
from .checkpoint import read_metadata
from .data import load_record
from .evaluation_protocol import frame_plan, window_selection
from .metrics import reconstruction


class ProbabilityDecoder:
    @staticmethod
    def decode(probabilities):
        return probabilities


def evaluate(prediction, reference, manifest, checkpoint, output, device="cpu", eval_fps=1.,
             contact_support=None, body_models="data/smplx_models"):
    if Path(output).exists():
        raise FileExistsError(output)
    saved = read_metadata(checkpoint)
    with np.load(prediction, allow_pickle=False) as archive:
        values = dict(archive)
    if str(values.get("architecture")) != ARCHITECTURE:
        raise ValueError("Expected a token_ar prediction")
    for key in ("codec_sha256", "support_sha256"):
        if str(values[key]) != saved[key]:
            raise ValueError(f"Prediction/checkpoint mismatch: {key}")
    if str(values["checkpoint_metadata_sha256"]) != digest_file(Path(checkpoint) / "metadata.json"):
        raise ValueError("Prediction was made by a different checkpoint")
    dataset = saved["config"]["dataset"]
    record = load_record(reference, dataset, saved["support_sha256"])
    rows = read_manifest(manifest)
    row = next((r for r in rows if Path(r["path"]).resolve() == Path(reference).resolve()), None)
    if row is None:
        raise ValueError("Reference must belong to the manifest")
    start, n, mode = int(values["window_start"]), len(values["human_params"]), str(values["mode"])
    if mode not in MODES or start < 0 or n < 1 or start + n > len(record["human_params"]):
        raise ValueError("Invalid prediction mode/window")
    if not np.array_equal(values["timestamps"], record["timestamps"][start:start + n]):
        raise ValueError("Reference timestamps differ from prediction")
    for key in ("dataset", "gender", "sequence_id", "body_model_sha256", "body_model_type"):
        if str(values[key]) != str(record[key]):
            raise ValueError(f"Prediction/reference mismatch: {key}")
    support = contact_support or saved["config"]["contact_support"]
    geometry = BodyGeometry(dataset, support, body_models).to(device)
    if geometry.support_hash != saved["support_sha256"] or geometry.layers[str(record["gender"])].asset_hash != str(record["body_model_sha256"]):
        raise ValueError("Body/contact assets changed")
    names = {"human": "human_params", "object": "object", "contact": "surface_contact"}
    targets = {k: torch.tensor(record[v][start:start + n], device=device, dtype=torch.float32)[None]
               for k, v in names.items()}
    states = {k: torch.tensor(values[v], device=device, dtype=torch.float32)[None]
              for k, v in {"human": "human_params", "object": "object", "contact": "contact_probability"}.items()}
    for key, bit in zip(MODALITIES, mode):
        if states[key].shape != targets[key].shape or not torch.isfinite(states[key]).all():
            raise ValueError(f"Invalid prediction shape/values for {key}")
        if bit == "0" and not torch.equal(states[key], targets[key]):
            raise ValueError(f"Observed {key} was changed")
    if ((states["contact"] < 0) | (states["contact"] > 1)).any():
        raise ValueError("Contact probability outside [0,1]")
    targets["surface_contact"] = targets["contact"]
    targets["valid_frames"] = torch.ones(1, n, dtype=torch.bool, device=device)
    targets["gender"] = torch.tensor([int(str(record["gender"]) == "male")], device=device)
    split_rows = [r for r in rows if r["split"] == row["split"]]
    plan = frame_plan(split_rows, eval_fps)
    ref_index = split_rows.index(row)
    selected, frame_ids = window_selection(plan[ref_index], start, start + n)
    if not len(selected):
        raise ValueError("No scoring frames in this window; use --eval-fps 0 to score all frames")
    vertices, _ = load_mesh_asset(row["object_mesh_path"])
    report = reconstruction(states, targets, geometry, mode, vertices * row["object_mesh_scale"],
                            ProbabilityDecoder(), float(values["contact_decode_threshold"]), selected)
    report.update(architecture=ARCHITECTURE, mode=mode, eval_fps=eval_fps, text_prompt=str(values["text_prompt"]),
                  frame_indices=selected.tolist(), frame_ids=frame_ids,
                  checkpoint=str(checkpoint), sampler="autoregressive_vq")
    if "refiner_sha256" in values:
        report.update(sampler="autoregressive_vq_continuous_refinement",
                      refiner_sha256=str(values["refiner_sha256"]), refiner_path=str(values["refiner_path"]))
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prediction", "reference", "manifest", "checkpoint", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-fps", type=float, default=1.)
    parser.add_argument("--contact-support")
    parser.add_argument("--body-models", default="data/smplx_models")
    print(evaluate(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
