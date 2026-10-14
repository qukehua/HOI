"""Attach canonical mesh assets to existing HOI manifests without rewriting motion archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import trimesh

from unified_hoi.data import read_manifest
from unified_hoi.mesh_geometry import GEOMETRY_VERSION, validate_mesh
from unified_hoi.preprocess import _mesh_for_behave, surface_points


def prepare_manifest(manifest, output, *, omomo_objects="OMOMO/data/captured_objects",
                     behave_objects="data/raw/behave/objects"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    entries = read_manifest(manifest)
    if not entries:
        raise ValueError("empty input manifest")
    assets, rows, expected_points = {}, [], {}
    output.parent.mkdir(parents=True, exist_ok=True)
    asset_folder = output.parent / "object_meshes"
    for number, entry in enumerate(entries, 1):
        dataset, obj = entry["dataset"], entry.get("object_id")
        with np.load(entry["path"], allow_pickle=False) as record:
            obj = str(record["object_id"].item()) if obj is None else str(obj)
            if str(record["dataset"].item()) != dataset or str(record["object_id"].item()) != obj:
                raise ValueError("manifest/archive mesh identity mismatch")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", obj):
                raise ValueError(f"invalid object identifier: {obj}")
            if dataset == "omomo":
                if obj in {"mop", "vacuum"}:
                    raise ValueError("articulated objects require separate rigid-part meshes")
                source = Path(omomo_objects) / f"{obj}_cleaned_simplified.obj"
                if "source_object_scale" not in record:
                    raise ValueError("OMOMO mesh requires recorded source_object_scale")
                scale = float(record["source_object_scale"])
                if "source_object_scale" in entry and not np.isclose(entry["source_object_scale"], scale):
                    raise ValueError("manifest/archive object scale mismatch")
                centered = False
            elif dataset == "behave":
                source, scale, centered = _mesh_for_behave(behave_objects, obj), 1., True
            else:
                raise ValueError(f"unsupported mesh dataset: {dataset}")
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("object scale must be finite and positive")
            key = (dataset, obj)
            if key not in assets:
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                mesh = trimesh.load(str(source), process=False, force="mesh")
                vertices = np.asarray(mesh.vertices, np.float64).copy()
                faces = np.asarray(mesh.faces, np.int64).copy()
                center = vertices.mean(0) if centered else np.zeros(3)
                vertices -= center
                validate_mesh(vertices, faces)
                asset_folder.mkdir(exist_ok=True)
                target = asset_folder / f"{dataset}_{obj}_{digest[:16]}_v{GEOMETRY_VERSION}.npz"
                if target.exists():
                    with np.load(target, allow_pickle=False) as cached:
                        if (str(cached["source_sha256"].item()) != digest
                                or not np.array_equal(cached["vertices"], vertices.astype(np.float32))
                                or not np.array_equal(cached["faces"], faces)):
                            raise ValueError(f"existing mesh cache mismatch: {target}")
                else:
                    np.savez_compressed(target, vertices=vertices.astype(np.float32), faces=faces,
                                        geometry_version=np.int32(GEOMETRY_VERSION), source_sha256=digest,
                                        dataset=dataset, object_id=obj, source_mesh=str(source.resolve()),
                                        original_vertex_mean=center,
                                        origin_convention="vertex_mean_centered" if centered else "annotation_origin")
                assets[key] = target
            # Verify the mesh frame against the existing conversion, including OMOMO scale.
            point_key = (key, len(record["object_points"]))
            if point_key not in expected_points:
                expected_points[point_key] = surface_points(source, point_key[1], center=centered)
            if not np.allclose(record["object_points"], expected_points[point_key] * scale, atol=2e-5, rtol=2e-5):
                raise ValueError(f"mesh does not match archive object origin/scale/sampling: {entry['path']}")
        row = dict(entry)
        for field in ("path", "text_features_path"):
            if row.get(field):
                row[field] = os.path.relpath(Path(row[field]).resolve(), output.parent).replace("\\", "/")
        row["text_variants"] = [dict(variant) for variant in entry.get("text_variants", [])]
        for variant in row["text_variants"]:
            if variant.get("text_features_path"):
                variant["text_features_path"] = os.path.relpath(Path(variant["text_features_path"]).resolve(),
                                                               output.parent).replace("\\", "/")
        row.update(object_mesh_path=os.path.relpath(assets[key], output.parent).replace("\\", "/"),
                   object_mesh_scale=scale, object_geometry_version=GEOMETRY_VERSION)
        rows.append(row)
        if number % 500 == 0:
            print(f"Validated {number}/{len(entries)} mesh associations", flush=True)
    # Publish only after every record passes; old manifests and motion files stay intact.
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(output)
    return {"manifest": str(output), "records": len(rows), "mesh_assets": len(assets)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--omomo-objects", default="OMOMO/data/captured_objects")
    parser.add_argument("--behave-objects", default="data/raw/behave/objects")
    args = parser.parse_args()
    print(json.dumps(prepare_manifest(args.manifest, args.output, omomo_objects=args.omomo_objects,
                                     behave_objects=args.behave_objects), ensure_ascii=False))


if __name__ == "__main__":
    main()
