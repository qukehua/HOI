"""Create a separate FK/mesh-consistent dataset; source archives are never edited."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import torch

from unified_hoi.body_geometry import canonicalize_body, file_sha256, validate_body_geometry
from unified_hoi.data import load_record, read_manifest
from unified_hoi.mesh_geometry import attach_mesh_geometry
from unified_hoi.smpl_mesh import DEFAULT_MODEL_PATH


def build_manifest(source, destination, models=DEFAULT_MODEL_PATH, resume=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.exists():
        raise FileExistsError(f"Use a new output manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data_dir = destination.parent / destination.stem
    if data_dir.exists() and not resume:
        raise FileExistsError(f"{data_dir} exists; --resume verifies already converted archives")
    entries = read_manifest(source)
    data_dir.mkdir(parents=True, exist_ok=True)
    converted, splits, hashes = [], Counter(), set()
    frames, sums, squares = 0, {}, {}
    max_shift = 0.
    changed_labels = total_labels = 0
    for index, entry in enumerate(entries):
        original = Path(entry["path"]).resolve()
        source_hash = file_sha256(original)
        output = data_dir / f"{index:06d}_{original.stem}.npz"
        if output.exists():
            record = load_record(output)
            validate_body_geometry(record, model_path=models)
            if str(record.get("source_archive_sha256")) != source_hash:
                raise ValueError(f"Source archive changed: {original}")
        else:
            record = load_record(original)
            old_contact = record["contact"].copy()
            if entry.get("object_mesh_path"):
                record = attach_mesh_geometry(record, entry["object_mesh_path"], entry["object_mesh_scale"])
            record = canonicalize_body(record, models)
            record["source_contact"] = old_contact
            record["source_archive_sha256"] = np.array(source_hash)
            record["source_record_path"] = np.array(os.path.relpath(original, output.parent))
            validate_body_geometry(record)
            temporary = output.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **record)
            os.replace(temporary, output)
        shift = np.linalg.norm(record["human"][..., :3] - record["source_body_xyz"], axis=-1)
        max_shift = max(max_shift, float(shift.max()))
        changed_labels += int(np.count_nonzero(record["contact"] != record["source_contact"]))
        total_labels += record["contact"].size
        hashes.add(str(record["body_model_sha256"]))
        copied = deepcopy(entry)
        for key in ("object_mesh_path", "text_features_path"):
            if copied.get(key):
                copied[key] = Path(os.path.relpath(Path(copied[key]).resolve(), destination.parent)).as_posix()
        for variant in copied.get("text_variants", []):
            if variant.get("text_features_path"):
                variant["text_features_path"] = Path(os.path.relpath(
                    Path(variant["text_features_path"]).resolve(), destination.parent)).as_posix()
        copied.update(path=output.relative_to(destination.parent).as_posix(), body_geometry_version=1)
        converted.append(copied)
        splits[entry["split"]] += 1
        if entry["split"] == "train":
            frames += len(record["human"])
            for key in ("human", "object"):
                values = np.asarray(record[key], np.float64)
                sums[key] = sums.get(key, 0) + values.sum(0)
                squares[key] = squares.get(key, 0) + np.square(values).sum(0)
        if (index + 1) % 200 == 0:
            print(f"{source.name}: {index + 1}/{len(entries)} converted/verified", flush=True)
    report = {"source": str(source), "manifest": str(destination), "body_geometry_version": 1,
              "sequences": len(converted), "splits": dict(splits), "model_sha256": sorted(hashes),
              "max_source_to_fk_joint_shift_m": max_shift,
              "changed_contact_labels": changed_labels, "total_contact_labels": total_labels,
              "normalization_train_frames": frames,
              "normalization_note": "Whole-sequence train-only audit statistics. Training refits on its own training windows; never reuse old checkpoint statistics."}
    for key in sums:
        mean = sums[key] / frames
        report[key + "_mean"] = mean.tolist()
        report[key + "_std"] = np.maximum(np.sqrt(np.maximum(squares[key] / frames - mean**2, 0)), .05).tolist()
    destination.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary = destination.with_suffix(".tmp.jsonl")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted), encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({k: report[k] for k in ("manifest", "sequences", "splits", "max_source_to_fk_joint_shift_m")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--resume", action="store_true", help="Verify and reuse partial output after interruption")
    args = parser.parse_args()
    torch.set_num_threads(2)
    build_manifest(args.manifest, args.output, args.models, args.resume)


if __name__ == "__main__":
    main()
