"""Download the complete, pinned HOI-Diff BEHAVE text release and verify Git blobs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


COMMIT = "a9c5c2d091b5b88ffacbfadbef3537876dbbc49b"
REPOSITORY = "https://github.com/neu-vi/HOI-Diff"
TEXT_DIRECTORY = "dataset/behave_t2m/texts/"
SUPPORT_FILES = {"LICENSE", "README.md", "process/README.md", "process/action_label.json",
                 "process/process_behave.py", "dataset/dataset_split.json"}


def wanted(path):
    return path in SUPPORT_FILES or (path.startswith(TEXT_DIRECTORY) and path.endswith(".txt"))


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "HOI-annotation-preparation"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def download(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tree_url = f"https://api.github.com/repos/neu-vi/HOI-Diff/git/trees/{COMMIT}?recursive=1"
    tree = json.loads(fetch(tree_url))
    if tree.get("truncated"):
        raise ValueError("Upstream Git tree is truncated; cannot verify a complete release")
    expected = {x["path"]: x for x in tree["tree"] if x["type"] == "blob" and wanted(x["path"])}
    if not any(path.startswith(TEXT_DIRECTORY) for path in expected):
        raise ValueError("No BEHAVE text annotations in upstream tree")
    archive_url = f"https://codeload.github.com/neu-vi/HOI-Diff/zip/{COMMIT}"
    print(f"Downloading {len(expected)} annotation/support files from {COMMIT}", flush=True)
    raw = fetch(archive_url)
    verified = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        prefix = f"HOI-Diff-{COMMIT}/"
        for member in archive.infolist():
            if not member.filename.startswith(prefix):
                continue
            relative = member.filename[len(prefix):]
            if relative not in expected:
                continue
            target = (output / relative).resolve()
            if not target.is_relative_to(output):
                raise ValueError(f"Unsafe archive member: {relative}")
            data = archive.read(member)
            blob_sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if blob_sha != expected[relative]["sha"]:
                raise ValueError(f"Git blob checksum mismatch: {relative}")
            if target.exists() and target.read_bytes() != data:
                raise FileExistsError(f"Existing local file differs from pinned release: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)
            verified[relative] = {"bytes": len(data), "git_blob_sha1": blob_sha,
                                  "sha256": hashlib.sha256(data).hexdigest()}
    if set(verified) != set(expected):
        raise ValueError(f"Missing files: {sorted(set(expected) - set(verified))}")
    report = {"repository": REPOSITORY, "commit": COMMIT, "archive_url": archive_url,
              "archive_sha256": hashlib.sha256(raw).hexdigest(), "tree_url": tree_url,
              "text_files": sum(path.startswith(TEXT_DIRECTORY) for path in verified),
              "support_files": len(SUPPORT_FILES), "files": verified}
    (output / "download_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified all {report['text_files']} text files: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"data/annotations/hoi_diff/{COMMIT}")
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
