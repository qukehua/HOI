"""Align published HOI-Diff captions to BEHAVE time intervals without changing source splits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from unified_hoi.data import load_record, read_manifest, validate_record


COMMIT = "a9c5c2d091b5b88ffacbfadbef3537876dbbc49b"


def action_segments(annotation):
    """Follow the published action boundaries, flushing at each source boundary too."""
    labels = {row["id"]: row["description"] for row in annotation["label_description"]}
    sources = defaultdict(list)
    for row in annotation["action_label"]:
        sources[row["name"]].append(row)
    result = {}
    for source, rows in sources.items():
        times = [float(row["frame"][1:]) for row in rows]
        if not all(b > a for a, b in zip(times, times[1:])):
            raise ValueError(f"Unordered action timestamps: {source}")
        run = []

        def flush():
            if run:
                start, end = int(float(run[0]["frame"][1:])), int(float(run[-1]["frame"][1:]))
                key = f"{source}_{start}"
                if key in result:
                    raise ValueError(f"Duplicate action interval: {key}")
                result[key] = {"source": source, "start": start, "end": end,
                               "action": labels[run[0]["label"]]}

        for row in rows:
            if labels[row["label"]] in {"no_interaction", "action_transition"}:
                flush()
                run = []
                continue
            if run and row["label"] != run[-1]["label"]:
                flush()
                run = []
            run.append(row)
        flush()
    return result


def parse_captions(path):
    groups, rejected = defaultdict(list), []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            caption, tokens, start, end = line.rsplit("#", 3)
            start, end = float(start), float(end)
            if (not caption.strip() or not np.isfinite([start, end]).all() or start < 0
                    or (not (start == end == 0) and end <= start)):
                raise ValueError("Invalid caption or relative time interval")
            groups[(start, end)].append({"text": caption.strip(), "tokens": tokens,
                                         "annotation_line": line_number})
        except ValueError as exc:
            rejected.append({"file": path.name, "line": line_number, "reason": str(exc), "raw": line})
    return groups, rejected


def select_times(times, start, end, relative, origin):
    # Upstream process_behave.py uses strict +/- 0.5 s action boundaries.
    mask = (times > start - .5) & (times < end + .5)
    if relative != (0., 0.):
        lo, hi = relative
        # Text time tags are seconds inside the extracted action clip, not source seconds.
        mask &= (times >= origin + lo - 1e-8) & (times < origin + hi - 1e-8)
    return mask


def prepare(manifest, annotations, raw_root, output):
    manifest, annotations, raw_root, output = map(Path, (manifest, annotations, raw_root, output))
    output = output.resolve()
    if (output / "manifest.jsonl").exists():
        raise FileExistsError(output / "manifest.jsonl")
    entries = read_manifest(manifest)
    sources = defaultdict(list)
    for entry in entries:
        if entry["dataset"] != "behave":
            raise ValueError("Expected a BEHAVE-only manifest")
        sources[entry.get("source_sequence_id", entry["sequence_id"])].append(entry)
    segments = action_segments(json.loads((annotations / "process/action_label.json").read_text()))
    all_texts = sorted((annotations / "dataset/behave_t2m/texts").glob("*.txt"))
    download_report = json.loads((annotations / "download_report.json").read_text())
    if len(all_texts) != download_report["text_files"] or download_report["commit"] != COMMIT:
        raise ValueError("Download is incomplete or not the supported release")
    (output / "sequences").mkdir(parents=True, exist_ok=True)
    report = {"upstream": "https://github.com/neu-vi/HOI-Diff", "commit": COMMIT,
              "source_manifest": str(manifest), "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "downloaded_text_files": len(all_texts), "downloaded_caption_lines": 0,
              "local_source_sequences": len(sources), "unmatched_sources": [], "unmatched_text_files": [],
              "rejected_caption_lines": [], "empty_intervals": [], "clamped_time_tags": [],
              "caption_policy": "all valid captions cached; sample uniformly at train time; first at val/test",
              "temporal_policy": "strict action +/-0.5 seconds; nonzero caption tags are relative seconds from first raw 30fps clip frame; intersect with available frames",
              "split_policy": "inherit original source_sequence_id assignments; no split redistribution"}
    by_source = defaultdict(list)
    for path in all_texts:
        report["downloaded_caption_lines"] += len(path.read_text(encoding="utf-8").splitlines())
        if path.stem not in segments:
            raise ValueError(f"No action interval for {path.name}")
        source = segments[path.stem]["source"]
        if source not in sources:
            report["unmatched_text_files"].append(path.name)
        else:
            by_source[source].append(path)
    rows = []
    coverage = {split: Counter() for split in ("train", "val", "test")}
    for source, source_entries in sources.items():
        if source not in by_source:
            report["unmatched_sources"].append(source)
        # Only numeric/string timestamps are needed from raw data; no upstream code is executed.
        with np.load(raw_root / source / "object_fit_all.npz", allow_pickle=False) as archive:
            raw_times = np.array([float(str(x)[1:]) for x in archive["frame_times"]])
        records = [(entry, load_record(entry["path"])) for entry in source_entries]
        covered = [np.zeros(len(record["timestamps"]), dtype=bool) for _, record in records]
        source_count = len(rows)
        for path in by_source[source]:
            segment = segments[path.stem]
            start, end = segment["start"], segment["end"]
            raw_clip = raw_times[(raw_times > start - .5) & (raw_times < end + .5)]
            if not len(raw_clip):
                report["empty_intervals"].append({"file": path.name, "reason": "No raw frames in action interval"})
                continue
            origin = float(raw_clip[0])
            caption_groups, rejected = parse_captions(path)
            report["rejected_caption_lines"].extend(rejected)
            for group_index, (relative, captions) in enumerate(caption_groups.items()):
                produced = False
                if relative != (0., 0.) and origin + relative[1] > raw_clip[-1] + 1 / 30 + 1e-3:
                    report["clamped_time_tags"].append({"file": path.name, "relative_seconds": list(relative),
                                                       "raw_clip_seconds": float(raw_clip[-1] - origin + 1 / 30)})
                for record_index, (entry, record) in enumerate(records):
                    mask = select_times(record["timestamps"], start, end, relative, origin)
                    if not mask.any():
                        continue
                    produced = True
                    covered[record_index] |= mask
                    sequence_id = f"{path.stem}__text{group_index}__part{record_index}"
                    clip = dict(record)
                    for key in ("human", "object", "contact", "timestamps"):
                        clip[key] = record[key][mask]
                    clip.update(sequence_id=np.array(sequence_id), source_sequence_id=np.array(source),
                                text=np.array(captions[0]["text"]),
                                text_variants=np.array([caption["text"] for caption in captions]))
                    validate_record(clip)
                    target = output / "sequences" / f"behave_{sequence_id}.npz"
                    if target.exists():
                        raise FileExistsError(target)
                    np.savez_compressed(target, **clip)
                    row = {key: value for key, value in entry.items()
                           if key not in {"text_features_path", "text_encoder", "text_variants"}}
                    row.update(path=os.path.relpath(target, output).replace("\\", "/"),
                               sequence_id=sequence_id, source_sequence_id=source, frames=len(clip["timestamps"]),
                               text=captions[0]["text"], text_variants=captions,
                               text_annotation_file=path.relative_to(annotations).as_posix(), text_annotation_commit=COMMIT,
                               action_interval_seconds=[start - .5, end + .5],
                               caption_interval_seconds=list(relative), raw_clip_origin_seconds=origin,
                               start_seconds=float(clip["timestamps"][0]), end_seconds=float(clip["timestamps"][-1]))
                    rows.append(row)
                    split = entry["split"]
                    coverage[split].update(clips=1, caption_variants=len(captions), frames=len(clip["timestamps"]))
                if not produced:
                    report["empty_intervals"].append({"file": path.name, "relative_seconds": list(relative),
                                                       "caption_lines": [x["annotation_line"] for x in captions],
                                                       "reason": "No local frames inside both action and caption intervals"})
        split = source_entries[0]["split"]
        coverage[split].update(source_sequences=1, sources_with_text=int(len(rows) > source_count),
                               original_frames=sum(len(mask) for mask in covered),
                               covered_unique_frames=sum(int(mask.sum()) for mask in covered))
    if not rows:
        raise ValueError("No aligned text clips")
    destination = output / "manifest.jsonl"
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    read_manifest(destination)  # Enforce source split isolation on the output too.
    report.update(aligned_clips=len(rows), aligned_caption_variants=sum(len(r["text_variants"]) for r in rows),
                  coverage_by_split=coverage, short_clips_under_one_second=sum(r["frames"] < r["fps"] for r in rows))
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("aligned_clips", "aligned_caption_variants", "coverage_by_split")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/behave_source/manifest.jsonl")
    parser.add_argument("--annotations", default=f"data/annotations/hoi_diff/{COMMIT}")
    parser.add_argument("--raw-root", default="data/raw/behave")
    parser.add_argument("--output", default="data/processed/behave")
    args = parser.parse_args()
    prepare(args.manifest, args.annotations, args.raw_root, args.output)


if __name__ == "__main__":
    main()
