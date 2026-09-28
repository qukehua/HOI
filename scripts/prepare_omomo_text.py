"""Read official OMOMO's shipped text ZIP as data; produce a sequence/text mapping."""
import argparse
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="external/omomo/omomo_text_anno.zip")
    parser.add_argument("--output", default="data/annotations/omomo_text.json")
    args = parser.parse_args()
    result = {}
    with zipfile.ZipFile(args.archive) as archive:
        for member in archive.namelist():
            if member.endswith(".json") and not member.startswith("__MACOSX/"):
                for key, value in json.loads(archive.read(member)).items():
                    if not isinstance(value, str) or (key in result and result[key] != value):
                        raise ValueError(f"Invalid or conflicting text annotation: {key}")
                    result[key] = value
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(result)} official annotations: {output}")


if __name__ == "__main__":
    main()
