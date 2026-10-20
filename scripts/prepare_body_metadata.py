"""Backfill original body metadata without recompressing/changing motion arrays.

Uses trusted local OMOMO joblib and BEHAVE NPZ/info.json. Default is a dry run;
--write atomically appends only missing fields and refuses conflicting metadata.
"""
import argparse
from collections import Counter
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

import joblib
import numpy as np

from unified_hoi.body_metadata import body_metadata
from unified_hoi.data import read_manifest


def append_metadata(path, metadata, write=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        for key, value in metadata.items():
            if key in archive and not np.array_equal(archive[key], value):
                raise ValueError(f"Conflicting {key} in {path}")
        missing = {k: v for k, v in metadata.items() if k not in archive}
    if not missing or not write:
        return bool(missing)
    with ZipFile(path) as archive:
        original = {i.filename: (i.CRC, i.file_size) for i in archive.infolist()}
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".body.npz", delete=False) as file:
        temporary = Path(file.name)
    try:
        shutil.copyfile(path, temporary)
        with ZipFile(temporary, "a", compression=ZIP_DEFLATED) as archive:
            for key, value in missing.items():
                buffer = BytesIO()
                np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
                archive.writestr(key + ".npy", buffer.getvalue())
        with ZipFile(temporary) as archive:
            for name, checksum in original.items():
                info = archive.getinfo(name)
                if (info.CRC, info.file_size) != checksum:
                    raise ValueError(f"Original array changed: {name}")
        with np.load(temporary, allow_pickle=False) as archive:
            for key, value in metadata.items():
                if not np.array_equal(archive[key], value):
                    raise ValueError(f"Metadata verification failed: {key}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True,
                        help="Intermediate mesh manifests from a fresh rebuild, before body-FK conversion")
    parser.add_argument("--omomo-raw", type=Path, default=Path("OMOMO/data"))
    parser.add_argument("--behave-raw", type=Path, default=Path("data/raw/behave"))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("docs/data_reports/body_metadata_backfill_report.json"))
    args = parser.parse_args()
    entries = [entry for manifest in args.manifests for entry in read_manifest(manifest)]
    omomo = {}
    if any(entry["dataset"] == "omomo" for entry in entries):
        for split in ("train", "test"):
            records = joblib.load(args.omomo_raw / f"{split}_diffusion_manip_seq_joints24.p")
            for source in records.values():
                name = str(source["seq_name"])
                if name in omomo:
                    raise ValueError(f"Duplicate OMOMO sequence: {name}")
                omomo[name] = (body_metadata({"dataset": "omomo", "gender": source["gender"],
                                             "betas": source["betas"]}), np.asarray(source["rest_offsets"])[:22])
            del records
    behave, seen, counts, changed = {}, set(), Counter(), Counter()
    for entry in entries:
        path = Path(entry["path"]).resolve()
        if path in seen:
            continue
        seen.add(path)
        dataset, source = entry["dataset"], entry["source_sequence_id"]
        if dataset == "omomo":
            metadata, offsets = omomo[source]
            with np.load(path, allow_pickle=False) as record:
                if not np.allclose(record["rest_offsets"][1:], offsets[1:], atol=1e-6, rtol=0):
                    raise ValueError(f"Raw/processed skeleton mismatch: {path}")
        elif dataset == "behave":
            if source not in behave:
                folder = args.behave_raw / source
                info = json.loads((folder / "info.json").read_text())
                with np.load(folder / "smpl_fit_all.npz", allow_pickle=False) as record:
                    betas = np.asarray(record["betas"]).reshape(-1, 10)
                    if not np.allclose(betas, betas[:1], atol=1e-4, rtol=1e-4):
                        raise ValueError(f"Time-varying body shape: {source}")
                    behave[source] = body_metadata({"dataset": dataset, "gender": info["gender"], "betas": betas[:1]})
            metadata = behave[source]
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        changed[dataset] += append_metadata(path, metadata, args.write)
        counts[dataset] += 1
        if len(seen) % 1000 == 0:
            print(f"Checked {len(seen)}/{len(entries)} archives", flush=True)
    report = {"write": args.write, "checked": dict(counts), "updated" if args.write else "would_update": dict(changed),
              "original_array_members_preserved": True, "body_models": {"omomo": "smplx16", "behave": "smplh10"}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
