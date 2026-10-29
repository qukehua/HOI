"""Verified AR prediction caches with native targets loaded only for supervision."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ..data import read_manifest
from ..mesh_geometry import load_mesh_asset, mesh_features
from . import MODALITIES, MODES
from .body import digest_file, human_dim
from .data import load_record
from .prepare_refinement import CACHE_KIND


class RefinementDataset:
    def __init__(self, cache, split, manifest=None):
        if split not in ("train", "val"):
            raise ValueError("Refinement training only accepts train/val")
        self.root = Path(cache)
        self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        finished = json.loads((self.root / "COMPLETE").read_text(encoding="utf-8"))
        if (self.metadata.get("kind") != CACHE_KIND
                or finished["metadata_sha256"] != digest_file(self.root / "metadata.json")
                or finished["index_sha256"] != digest_file(self.root / "index.jsonl")):
            raise ValueError("Incomplete or changed refinement cache")
        manifest = manifest or self.metadata["manifest"]
        if digest_file(manifest) != self.metadata["manifest_sha256"]:
            raise ValueError("Refinement manifest changed")
        rows = read_manifest(manifest)
        self.records = [r for r in rows if r["split"] == split]
        all_entries = [json.loads(line) for line in (self.root / "index.jsonl").read_text(encoding="utf-8").splitlines()]
        if finished["entries"] != len(all_entries):
            raise ValueError("Cache entry count changed")
        self.entries = [r for r in all_entries if r["split"] == split]
        seen, checked = set(), {}
        for row in self.entries:
            key = (row["record_index"], row["start"], row["end"], row["mode"])
            if row["mode"] not in MODES or key in seen:
                raise ValueError("Duplicate or invalid refinement task")
            seen.add(key)
            entry = self.records[row["record_index"]]
            if (entry["sequence_id"] != row["sequence_id"]
                    or entry.get("source_sequence_id", entry["sequence_id"]) != row["source_sequence_id"]):
                raise ValueError("Cache source/split changed")
            for path, expected in ((entry["path"], row["reference_sha256"]),
                                   (entry["object_mesh_path"], row["object_mesh_sha256"])):
                if path not in checked:
                    checked[path] = digest_file(path)
                if checked[path] != expected:
                    raise ValueError("Native target/mesh changed since AR caching")
            if digest_file(self.root / row["file"]) != row["sha256"]:
                raise ValueError("Cached AR prediction changed")
        if not self.entries:
            raise ValueError("Empty refinement split")

    def __len__(self):
        return len(self.entries)

    @lru_cache(maxsize=8)
    def record(self, index):
        return load_record(self.records[index]["path"], self.metadata["dataset"], self.metadata["support_sha256"])

    def __getitem__(self, index):
        row = self.entries[index]
        entry, record = self.records[row["record_index"]], self.record(row["record_index"])
        start, end = row["start"], row["end"]
        if not 0 <= start < end <= len(record["human_params"]):
            raise ValueError("Invalid cached window bounds")
        target = {key: torch.from_numpy(record[source][start:end].astype(np.float32).copy())
                  for key, source in (("human", "human_params"), ("object", "object"), ("contact", "surface_contact"))}
        with np.load(self.root / row["file"], allow_pickle=False) as archive:
            coarse = {k: torch.from_numpy(archive[k].astype(np.float32)) for k in MODALITIES}
        for key, dim, bit in zip(MODALITIES, (human_dim(self.metadata["dataset"]), 9, 690), row["mode"]):
            if coarse[key].shape != (end - start, dim) or not torch.isfinite(coarse[key]).all():
                raise ValueError("Invalid cached prediction")
            if bit == "0" and not torch.equal(coarse[key], target[key]):
                raise ValueError(f"Observed {key} differs from native data")
        bps, _ = mesh_features(str(Path(entry["object_mesh_path"]).resolve()), entry["object_mesh_scale"])
        vertices, _ = load_mesh_asset(str(Path(entry["object_mesh_path"]).resolve()))
        gender_name = str(record["gender"])
        if str(record["body_model_sha256"]) != self.metadata["body_assets"][gender_name]:
            raise ValueError("Native body asset provenance mismatch")
        return dict(coarse=coarse, target=target, mode=row["mode"], bps=torch.from_numpy(bps.copy()),
                    gender=torch.tensor(int(gender_name == "male")), fps=float(record["fps"]),
                    vertices=torch.from_numpy((vertices * entry["object_mesh_scale"]).copy()))


def to_device(item, device):
    return {k: to_device(v, device) if isinstance(v, dict) else v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in item.items()}
