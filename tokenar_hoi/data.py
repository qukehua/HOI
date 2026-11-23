"""Shared manifest reader with sequence split isolation and relative paths."""
import json
from pathlib import Path


def read_manifest(path: str | Path) -> list[dict]:
    """Resolve paths and prevent sequence leakage, including overlapping windows."""
    path = Path(path)
    records, assignments = [], {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        entry = json.loads(line)
        for field in ("path", "dataset", "sequence_id", "split", "fps"):
            if field not in entry:
                raise ValueError(f"manifest line {number} missing {field}")
        if entry["split"] not in {"train", "val", "test"}:
            raise ValueError(f"invalid split on manifest line {number}")
        key = (entry["dataset"], entry.get("source_sequence_id", entry["sequence_id"]))
        if key in assignments and assignments[key] != entry["split"]:
            raise ValueError(f"sequence split leakage: {key}")
        assignments[key] = entry["split"]
        resolved = Path(entry["path"])
        entry["path"] = str(resolved if resolved.is_absolute() else path.parent / resolved)
        if entry.get("object_mesh_path"):
            mesh_path = Path(entry["object_mesh_path"])
            entry["object_mesh_path"] = str(mesh_path if mesh_path.is_absolute() else path.parent / mesh_path)
        if entry.get("text_features_path"):
            feature_path = Path(entry["text_features_path"])
            entry["text_features_path"] = str(feature_path if feature_path.is_absolute()
                                              else path.parent / feature_path)
        for variant in entry.get("text_variants", []):
            if variant.get("text_features_path"):
                feature_path = Path(variant["text_features_path"])
                variant["text_features_path"] = str(feature_path if feature_path.is_absolute()
                                                    else path.parent / feature_path)
        records.append(entry)
    return records
