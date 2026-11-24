"""Native-parameter windows, with strict dataset and contact-support isolation."""
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data import read_manifest
from ..mesh_geometry import mesh_features
from . import REPRESENTATION
from .body import human_dim, beta_count
from .contact_offset import CONTACT_CAP, CONTACT_DIM, CONTACT_VERTICES
from ..body_metadata import DATASET_MODELS


def load_record(path, dataset=None, support_hash=None, *, contact_support=None, body_models=None,
                fps=None, object_mesh=None, object_mesh_scale=1., sequence_id=None):
    from .interact_convert import convert_interact, interact_files
    if interact_files(path) is not None:
        r = convert_interact(path, dataset, contact_support, fps=fps, object_mesh=object_mesh,
                             object_mesh_scale=object_mesh_scale, body_models=body_models or "data/smplx_models",
                             sequence_id=sequence_id)
    else:
        with np.load(path, allow_pickle=False) as archive:
            r = dict(archive)
    name = str(r.get("dataset"))
    if name not in ("behave", "omomo") or (dataset and name != dataset):
        raise ValueError("Dataset isolation violation")
    if str(r.get("representation")) != REPRESENTATION:
        raise ValueError("Expected parametric_v1 record; prepare native data first")
    if str(r.get("body_model_type")) != DATASET_MODELS[name] or str(r.get("gender")) not in ("male", "female"):
        raise ValueError("Native body model/gender mismatch")
    if support_hash and str(r["support_sha256"]) != support_hash:
        raise ValueError("Surface-contact support mismatch")
    t = len(r["human_params"])
    for k, shape in (("human_params", (t, human_dim(name))), ("object", (t, 9)),
                     ("timestamps", (t,))):
        if r[k].shape != shape or not np.isfinite(r[k]).all():
            raise ValueError(f"Invalid native record {k}")
    if "contact_offset" not in r:
        raise ValueError("contact_offset must be finite (T, 690, 4) object-frame targets")
    offset = np.asarray(r["contact_offset"], np.float32)
    if offset.shape == (t, CONTACT_VERTICES, 4):
        offset = offset.reshape(t, CONTACT_DIM)
    if offset.shape != (t, CONTACT_DIM) or not np.isfinite(offset).all():
        raise ValueError("contact_offset must be finite (T, 690, 4) object-frame targets")
    distance = offset.reshape(t, CONTACT_VERTICES, 4)[..., 3]
    if (distance < 0).any() or (distance > CONTACT_CAP + 1e-4).any():
        raise ValueError("Contact distances must lie in [0, 5cm]")
    r["contact_offset"] = np.ascontiguousarray(offset)
    if t < 1 or (t > 1 and not (np.diff(r["timestamps"]) > 0).all()):
        raise ValueError("Invalid native record timestamps")
    fps = float(r["fps"])
    if not np.isfinite(fps) or fps <= 0 or (t > 1 and not np.all(
            np.abs(np.diff(r["timestamps"]) - 1 / fps) <= min(.0021, .1 / fps))):
        raise ValueError("Native timestamps contain gaps or disagree with fps")
    for rotation in (r["human_params"][:, beta_count(name):-3], r["object"][:, 3:]):
        rows = rotation.reshape(-1, 2, 3)
        if (not np.allclose(np.linalg.norm(rows, axis=-1), 1, atol=2e-3)
                or not np.allclose((rows[:, 0] * rows[:, 1]).sum(-1), 0, atol=2e-3)):
            raise ValueError("Native rotation6d rows must be orthonormal")
    if name == "omomo":
        source = str(r.get("hand_source"))
        hands = r["human_params"][:, beta_count(name) + 132:beta_count(name) + 312].reshape(t, 30, 6)
        neutral = source == "unavailable_neutral_convention"
        if source not in ("unavailable_neutral_convention", "interact_corrected"):
            raise ValueError("OMOMO hand_source must be unavailable_neutral_convention or interact_corrected")
        if neutral and not np.allclose(hands, [1, 0, 0, 0, 1, 0], atol=1e-6, rtol=0):
            raise ValueError("OMOMO neutral-hand records must use identity hand rotations")
    return r


class ParametricDataset(Dataset):
    def __init__(self, manifest, split, dataset, support_hash, window=120, stride=None,
                 text_condition=True, text_dim=512, include_geometry=True, contact_support=None,
                 body_models=None):
        if window < 1 or (stride is not None and stride < 1):
            raise ValueError("Window and stride must be positive")
        rows = read_manifest(manifest)
        if {r["dataset"] for r in rows} != {dataset}:
            raise ValueError("Mixed datasets are prohibited for parametric training/evaluation")
        self.records = [r for r in rows if r["split"] == split]
        self.dataset, self.support_hash, self.window = dataset, support_hash, window
        self.text_condition, self.text_dim, self.split = text_condition, text_dim, split
        self.include_geometry = include_geometry
        self.text_generator = None
        self.windows = []
        self.loaded = []
        for i, row in enumerate(self.records):
            if os.environ.get("RANK", "0") == "0" and contact_support and (i == 0 or (i + 1) % 25 == 0 or i + 1 == len(self.records)):
                print(f"loading {split} {i + 1}/{len(self.records)} {row.get('sequence_id', '')}", flush=True)
            r = load_record(row["path"], dataset, support_hash, contact_support=contact_support,
                            body_models=body_models, fps=row.get("fps"), object_mesh=row.get("object_mesh_path"),
                            object_mesh_scale=row.get("object_mesh_scale", 1.), sequence_id=row.get("sequence_id"))
            if not row.get("object_mesh_path") and r.get("object_mesh_path") is not None:
                row["object_mesh_path"] = str(np.asarray(r["object_mesh_path"]).item())
                row.setdefault("object_mesh_scale", float(np.asarray(r.get("object_mesh_scale", 1.)).item()))
            if not row.get("text") and r.get("text") is not None:
                text = str(np.asarray(r["text"]).item()).strip()
                if text:
                    row["text"] = text
            self.loaded.append(r)
            self.windows.extend((i, start, min(start + window, len(r["human_params"])))
                                for start in range(0, len(r["human_params"]), stride or window))

    def __len__(self):
        return len(self.windows)

    def normalization_batches(self, chunk=512):
        """Each training frame once; statistics do not need mesh BPS or text loading."""
        if self.split != "train":
            raise ValueError("Normalization may only use training records")
        for record in self.loaded:
            for start in range(0, len(record["human_params"]), chunk):
                values = {key: torch.from_numpy(record[source][start:start + chunk].astype(np.float32))[None]
                          for source, key in (("human_params", "human"), ("object", "object"),
                                              ("contact_offset", "contact"))}
                values["valid_frames"] = torch.ones(values["human"].shape[:2], dtype=torch.bool)
                yield values

    def __getitem__(self, index):
        record_index, start, end = self.windows[index]
        entry = self.records[record_index]
        r = self.loaded[record_index]
        output = {}
        for source, key in (("human_params", "human"), ("object", "object"),
                            ("contact_offset", "contact_offset")):
            value = r[source][start:end].astype(np.float32)
            value = np.concatenate((value, np.repeat(value[-1:], self.window - len(value), 0)))
            output[key] = torch.from_numpy(value)
        if self.include_geometry:
            mesh_path = str(Path(entry["object_mesh_path"]).resolve())
            scale = entry["object_mesh_scale"]
            if mesh_path.endswith(".npz"):
                bps, vertices = mesh_features(mesh_path, scale)
            else:
                from .interact_convert import surface_features
                bps, vertices = surface_features(mesh_path, scale)
            output["object_bps"] = torch.from_numpy(bps.copy())
            indices = np.linspace(0, len(vertices) - 1, 512).astype(int)
            output["object_points"] = torch.from_numpy(vertices[indices].copy())
        # Paths collate as strings; full meshes are loaded only for selected loss frames.
        output["object_mesh_path"] = str(Path(entry["object_mesh_path"]).resolve())
        output["object_mesh_scale"] = float(entry["object_mesh_scale"])
        output["valid_frames"] = torch.arange(self.window) < end - start
        output["gender"] = torch.tensor(int(str(r["gender"]) == "male"))
        output["fps"] = torch.tensor(float(r["fps"]), dtype=torch.float32)
        features = np.zeros(self.text_dim, np.float32)
        feature_path = entry.get("text_features_path")
        variants = entry.get("text_variants", [])
        if self.text_condition and variants:
            i = int(torch.randint(len(variants), (), generator=self.text_generator)) if self.split == "train" else 0
            feature_path = variants[i].get("text_features_path")
        if self.text_condition and feature_path:
            features = np.load(feature_path, allow_pickle=False)
            if features.shape != (self.text_dim,) or not np.isfinite(features).all():
                raise ValueError("Invalid text features")
        output["text_features"] = torch.from_numpy(features.astype(np.float32).copy())
        output["text_available"] = torch.tensor(bool(self.text_condition and feature_path))
        output["dataset"] = self.dataset
        output["body_model_sha256"] = str(r["body_model_sha256"])
        output["sequence_id"] = str(r["sequence_id"])
        output["record_index"], output["window_start"] = record_index, start
        return output
