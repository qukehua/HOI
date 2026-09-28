"""Combine sequence-level manifests without breaking relative paths or split integrity."""
import argparse
import json
from pathlib import Path
from unified_hoi.data import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    records, seen, splits = [], set(), {}
    for source in args.inputs:
        for entry in read_manifest(source):
            key = (entry["dataset"], entry["sequence_id"])
            if key in seen:
                raise ValueError(f"Duplicate sequence: {key}")
            seen.add(key)
            source_key = (entry["dataset"], entry.get("source_sequence_id", entry["sequence_id"]))
            if source_key in splits and splits[source_key] != entry["split"]:
                raise ValueError(f"Source sequence leaks across splits: {source_key}")
            splits[source_key] = entry["split"]
            # Keep the output portable if the entire project tree moves.
            import os
            for field in ("path", "text_features_path"):
                if entry.get(field):
                    entry[field] = os.path.relpath(Path(entry[field]).resolve(), output.parent).replace("\\", "/")
            records.append(entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in records), encoding="utf-8")
    print(f"{len(records)} records: {output}")


if __name__ == "__main__":
    main()
