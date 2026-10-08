"""Sequence-first, timestamp-aware loading of the common HOI archive format."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ARRAY_KEYS = ("human", "object", "contact", "object_points", "rest_offsets")
META_KEYS = ("sequence_id", "dataset", "subject_id", "object_id", "text")


def scalar(value):
    return np.asarray(value).item()


def validate_record(record: dict) -> None:
    """Reject corrupt representations; never silently invent time or coordinates."""
    human = np.asarray(record["human"])
    if human.ndim != 3 or human.shape[1:] != (22, 9) or len(human) < 1:
        raise ValueError("human must have shape [T,22,9], T >= 1")
    n = len(human)
    for key, shape in (("object", (n, 9)), ("contact", (n, 22)),
                       ("rest_offsets", (22, 3))):
        if np.asarray(record[key]).shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
    points = np.asarray(record["object_points"])
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("object_points must have shape [K,3], K > 0")
    for key in ARRAY_KEYS:
        if not np.isfinite(record[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    if not np.isin(record["contact"], [0, 1]).all():
        raise ValueError("contact must be binary (unknown controls use a separate mask)")
    for key in ("human", "object"):
        r6 = np.asarray(record[key])[..., 3:9].reshape(-1, 2, 3)
        if not (np.allclose(np.linalg.norm(r6, axis=-1), 1, atol=2e-3)
                and np.allclose((r6[:, 0] * r6[:, 1]).sum(-1), 0, atol=2e-3)):
            raise ValueError(f"{key} rotation6d must contain two orthonormal matrix rows")
    fps = float(scalar(record["fps"]))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    stamps = np.asarray(record["timestamps"], dtype=np.float64)
    if stamps.shape != (n,) or not np.isfinite(stamps).all():
        raise ValueError("timestamps must be finite [T] seconds")
    if n > 1:
        dt = np.diff(stamps)
        if not np.all(dt > 0):
            raise ValueError("timestamps must strictly increase")
        # BEHAVE rounds timestamps to milliseconds. A missing frame is NOT padding.
        tolerance = min(0.0021, 0.1 / fps)
        if not np.all(np.abs(dt - 1 / fps) <= tolerance):
            raise ValueError("timestamps contain gaps or disagree with fps; split continuous runs first")
    for key in META_KEYS:
        if key not in record or not isinstance(scalar(record[key]), str):
            raise ValueError(f"{key} must be a scalar string")
    if not scalar(record["sequence_id"]):
        raise ValueError("sequence_id cannot be empty")


def load_record(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    validate_record(result)
    return result


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


class HOIDataset(Dataset):
    """Select the split FIRST, then take windows inside each whole sequence.

    Short final windows are padded by repeating the last frame. valid_frames marks
    padding. No interpolation, upsampling, or guessed semantic labels is performed.
    All records in a loader must use the same object-point count and frame rate.
    """

    def __init__(self, manifest, split, window, stride=None, text_dim=512, text_condition=True):
        if window <= 0 or (stride is not None and stride <= 0):
            raise ValueError("window and stride must be positive")
        self.window, self.stride, self.text_dim = int(window), int(stride or window), int(text_dim)
        self.text_condition = text_condition
        self.split = split
        self.text_generator = None
        self.records = [r for r in read_manifest(manifest) if r["split"] == split]
        self.windows = []
        point_counts, frame_rates = set(), set()
        for idx, entry in enumerate(self.records):
            rec = load_record(entry["path"])
            for key in ("dataset", "sequence_id"):
                if str(scalar(rec[key])) != str(entry[key]):
                    raise ValueError(f"archive/manifest {key} mismatch: {entry['path']}")
            fps = float(scalar(rec["fps"]))
            if not np.isclose(fps, entry["fps"]):
                raise ValueError("archive/manifest fps mismatch")
            frame_rates.add(round(fps, 5))
            point_counts.add(len(rec["object_points"]))
            length = len(rec["human"])
            self.windows.extend((idx, start, min(start + self.window, length))
                                for start in range(0, length, self.stride))
        if len(point_counts) > 1:
            raise ValueError("mixed object point counts; preprocess every sequence with the same K")
        if len(frame_rates) > 1:
            raise ValueError("mixed frame rates; use separate loaders or explicit downsampling")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        rec_idx, start, end = self.windows[index]
        entry = self.records[rec_idx]
        record = load_record(entry["path"])
        length = end - start
        output = {}
        for key in ("human", "object", "contact"):
            value = np.asarray(record[key][start:end], dtype=np.float32)
            if length < self.window:
                value = np.concatenate((value, np.repeat(value[-1:], self.window - length, axis=0)))
            output[key] = torch.from_numpy(value.copy())
        for key in ("object_points", "rest_offsets"):
            output[key] = torch.as_tensor(np.asarray(record[key], dtype=np.float32).copy())
        output["valid_frames"] = torch.arange(self.window) < length
        output["fps"] = torch.tensor(float(scalar(record["fps"])), dtype=torch.float32)
        output["fps_verified_by_metadata"] = torch.tensor(bool(scalar(record.get("fps_verified_by_metadata", False))))
        output["floor_y"] = torch.tensor(float(scalar(record.get("floor_y", 0.))), dtype=torch.float32)
        output["floor_available"] = torch.tensor("floor_y" in record)
        output["contact_threshold"] = torch.tensor(float(scalar(record.get("contact_threshold", .05))), dtype=torch.float32)
        stamps = np.asarray(record["timestamps"][start:end], dtype=np.float64)
        output["timestamps"] = torch.from_numpy(np.pad(stamps, (0, self.window - length), mode="edge"))
        features = np.zeros(self.text_dim, dtype=np.float32)
        available = False
        selected_text = str(scalar(record["text"]))
        feature_path = entry.get("text_features_path")
        variants = entry.get("text_variants", [])
        if self.text_condition and variants:
            index = (int(torch.randint(len(variants), (), generator=self.text_generator).item())
                     if self.split == "train" else 0)
            selected = variants[index]
            selected_text = selected["text"]
            verified_texts = np.asarray(record.get("text_variants", [scalar(record["text"])]))
            if selected_text not in verified_texts.tolist():
                raise ValueError("manifest text variant is not present in the annotated archive")
            feature_path = selected.get("text_features_path")
            if not feature_path:
                raise ValueError("text variants require cached features; run scripts/cache_text.py")
        if self.text_condition and feature_path:
            cached = np.load(feature_path, allow_pickle=False)
            if cached.shape != (self.text_dim,) or not np.isfinite(cached).all():
                raise ValueError(f"text embedding must be finite [{self.text_dim}]")
            features = np.asarray(cached, dtype=np.float32)
            available = bool(selected_text.strip())
            if not available:
                raise ValueError("text cache supplied for a record with no verified text")
        output["text_features"] = torch.from_numpy(features.copy())
        output["text_available"] = torch.tensor(available)
        for key in META_KEYS:
            output[key] = str(scalar(record[key]))
        output["text"] = selected_text
        output["source_sequence_id"] = entry.get("source_sequence_id", output["sequence_id"])
        output["window_start"] = start
        return output
