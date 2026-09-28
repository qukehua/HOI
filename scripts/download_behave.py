"""Download only TriDi-required BEHAVE assets from the official host.

By downloading/using BEHAVE its research-only license applies:
https://virtualhumans.mpi-inf.mpg.de/behave/license.html
SMPL-H models are NOT redistributed here; obtain them from their official portal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request
import zipfile

BASE = "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/"
FILES = ("objects.zip", "behave-30fps-params-v1.tar", "split.json")


def download(url: str, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
    expected = int(head.headers.get("Content-Length", 0))
    head.close()
    if not target.exists() or (expected and target.stat().st_size != expected):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                offset = 0
            with partial.open("ab" if offset else "wb") as stream:
                received = offset
                last_print = time.monotonic()
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    received += len(chunk)
                    if time.monotonic() - last_print > 15:
                        print(f"{target.name}: {received / 1e6:.1f}/{expected / 1e6:.1f} MB", flush=True)
                        last_print = time.monotonic()
        if expected and partial.stat().st_size != expected:
            raise IOError(f"Incomplete download: {partial}")
        partial.replace(target)
    with target.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"url": url, "path": str(target.resolve()), "bytes": target.stat().st_size,
            "sha256_local": digest, "publisher_checksum_available": False}


def extract(archive: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve()
    def check(name: str) -> Path:
        dest = (root / name).resolve()
        if not dest.is_relative_to(root):
            raise ValueError(f"Unsafe archive member: {name}")
        return dest
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                check(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are not supported")
            z.extractall(root)
    else:
        with tarfile.open(archive) as t:
            for member in t:
                dest = check(member.name)
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(member) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    raise ValueError(f"Unsupported archive member: {member.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/behave"))
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    records = []
    for filename in FILES:
        destination = args.output / ("archives" if filename != "split.json" else "") / filename
        for attempt in range(3):
            try:
                records.append(download(BASE + filename, destination))
                break
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
        if not args.no_extract and filename != "split.json":
            print(f"Extracting {filename}", flush=True)
            extract(destination, args.output)
    (args.output / "download_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Downloaded and verified byte lengths: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
