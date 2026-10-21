"""Build isolated native-parameter records and surface-contact supervision."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

from ..data import read_manifest
from ..mesh_geometry import load_mesh_asset
from ..geometry import rotation6d_to_matrix
from ..smpl_mesh import DEFAULT_MODEL_PATH
from . import REPRESENTATION
from .body import BodyGeometry, digest_file, legacy_to_parameters, make_support


@lru_cache(maxsize=64)
def object_tree(path, scale):
    vertices, faces = load_mesh_asset(path)
    return cKDTree(vertices[np.unique(faces)] * scale)


def fixed_hands(dataset, raw="data/raw/behave"):
    if dataset == "omomo":
        return np.tile(np.eye(3, dtype=np.float32), (30, 1, 1)), "unavailable_neutral_convention"
    files = sorted(Path(raw).glob("*/smpl_fit_all.npz"))
    if not files:
        raise FileNotFoundError("Original BEHAVE SMPL-H fits are required")
    reference = None
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            hand = archive["poses"].reshape(-1, 52, 3)[:, 22:]
            if reference is None:
                reference = hand[0].copy()
            if not np.allclose(hand, reference[None], atol=1e-7, rtol=0):
                raise ValueError("BEHAVE hands vary: implement timestamped full hand import, never discard it")
    return Rotation.from_rotvec(reference.reshape(-1, 3)).as_matrix().astype(np.float32), "constant_source_fit"


def prepare(manifest, output, dataset, assets, models=DEFAULT_MODEL_PATH, resume=False):
    manifest, output, assets = Path(manifest).resolve(), Path(output).resolve(), Path(assets).resolve()
    rows = read_manifest(manifest)
    if not rows or {r["dataset"] for r in rows} != {dataset}:
        raise ValueError("Manifest must contain exactly the requested dataset")
    if output.exists():
        raise FileExistsError(output)
    support = assets / "contact_support.npz"
    make_support(dataset, support, models)
    hands, hand_source = fixed_hands(dataset)
    hand_file = assets / "hand_convention.npz"
    if hand_file.exists():
        with np.load(hand_file, allow_pickle=False) as old:
            if not np.array_equal(old["rotations"], hands) or str(old["dataset"]) != dataset:
                raise ValueError("Hand convention changed")
    else:
        np.savez_compressed(hand_file, rotations=hands, dataset=dataset, source=hand_source)
    geometry = BodyGeometry(dataset, support, models)
    destination = output.parent / output.stem
    destination.mkdir(parents=True, exist_ok=True)
    result, positives, count, total_frames = [], 0, 0, 0
    with torch.no_grad():
        for index, entry in enumerate(rows):
            source = Path(entry["path"])
            target = destination / f"{index:06d}_{source.stem}.npz"
            source_hash = digest_file(source)
            if target.exists():
                if not resume:
                    raise FileExistsError(f"Partial dataset exists; use --resume: {target}")
                with np.load(target, allow_pickle=False) as saved:
                    if (str(saved["source_sha256"]) != source_hash
                            or str(saved["support_sha256"]) != geometry.support_hash
                            or str(saved["dataset"]) != dataset
                            or str(saved["body_model_sha256"]) != geometry.layers[str(saved["gender"])].asset_hash):
                        raise ValueError("Partial dataset provenance mismatch")
                    contact = saved["surface_contact"]
            else:
                with np.load(source, allow_pickle=False) as archive:
                    record = dict(archive)
                human = legacy_to_parameters(record, geometry, hands)
                tree = object_tree(str(Path(entry["object_mesh_path"]).resolve()), float(entry["object_mesh_scale"]))
                contacts, distances = [], []
                for start in range(0, len(human), 64):
                    human_tensor = torch.from_numpy(human[start:start + 64])[None]
                    gender = torch.tensor([int(str(record["gender"]) == "male")])
                    vertices, _ = geometry(human_tensor, gender)
                    obj = np.asarray(record["object"])[start:start + 64]
                    rotation = rotation6d_to_matrix(torch.from_numpy(obj[:, 3:])).numpy()
                    local = np.einsum("tji,tkj->tki", rotation, vertices[0].numpy() - obj[:, None, :3])
                    distance = tree.query(local.reshape(-1, 3), workers=4)[0].reshape(-1, 690)
                    distances.append(distance.astype(np.float16))
                    contacts.append((distance <= .05).astype(np.uint8))
                contact = np.concatenate(contacts)
                arrays = {k: record[k] for k in ("object", "timestamps", "fps", "sequence_id", "subject_id",
                          "object_id", "text", "gender", "betas", "body_model_type")}
                for key in ("text_variants", "source_sequence_id", "source_to_world", "source_object_scale"):
                    if key in record:
                        arrays[key] = record[key]
                arrays.update(human_params=human, surface_contact=contact,
                              surface_distance=np.concatenate(distances), representation=REPRESENTATION,
                              dataset=dataset, source_sha256=source_hash, support_sha256=geometry.support_hash,
                              body_model_sha256=geometry.layers[str(record["gender"])].asset_hash,
                              hand_source=hand_source, contact_threshold=np.float32(.05))
                temporary = target.with_suffix(".partial.npz")
                np.savez_compressed(temporary, **arrays)
                temporary.replace(target)
            positives += int(contact.sum())
            count += contact.size
            total_frames += len(contact)
            row = dict(entry)
            for field in ("object_mesh_path", "text_features_path"):
                if row.get(field):
                    row[field] = os.path.relpath(row[field], output.parent).replace("\\", "/")
            row["text_variants"] = [dict(v) for v in entry.get("text_variants", [])]
            for variant in row["text_variants"]:
                if variant.get("text_features_path"):
                    variant["text_features_path"] = os.path.relpath(variant["text_features_path"], output.parent).replace("\\", "/")
            row.update(path=target.relative_to(output.parent).as_posix(), representation=REPRESENTATION,
                       support_sha256=geometry.support_hash)
            result.append(row)
            if (index + 1) % 100 == 0:
                print(f"{dataset}: {index + 1}/{len(rows)} native records", flush=True)
    temporary = output.with_suffix(".partial.jsonl")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result), encoding="utf-8")
    temporary.replace(output)
    report = dict(dataset=dataset, records=len(rows), frames=total_frames,
                  positive_surface_contact_fraction=positives / count, hand_source=hand_source,
                  support_sha256=geometry.support_hash, representation=REPRESENTATION,
                  model_hashes={k: v.asset_hash for k, v in geometry.layers.items()},
                  evaluation_joints=73, contact_vertices=690, original_archives_modified=False)
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=("behave", "omomo"), required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--models", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    prepare(args.manifest, args.output, args.dataset, args.assets, args.models, args.resume)


if __name__ == "__main__":
    main()
