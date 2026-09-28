"""Fetch exact upstream commits without changing an existing user's checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from unified_hoi.integrations.upstream import UPSTREAMS, checkout_report, verify_tridi_primitives  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "external")
    parser.add_argument("--only", choices=list(UPSTREAMS), action="append")
    parser.add_argument("--verify-only", action="store_true", help="No network or checkout changes")
    args = parser.parse_args(argv)
    reports = []
    for name in args.only or UPSTREAMS:
        spec = UPSTREAMS[name]
        path = args.directory.resolve() / name
        if path.exists():
            report = checkout_report(path, name)
            if not report.get("pinned") or report.get("dirty") or report.get("remote") != spec["url"]:
                raise RuntimeError(f"Existing checkout was left unchanged: {json.dumps(report)}")
        elif args.verify_only:
            raise FileNotFoundError(f"Missing checkout: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            # The destination is absent. Never reset, clean, pull, or remove an existing tree.
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--", spec["url"], str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", spec["commit"]], check=True)
            report = checkout_report(path, name)
            if not report.get("pinned") or report.get("dirty"):
                raise RuntimeError(f"New checkout failed verification: {report}")
        if name == "tridi":
            report["vendored_primitives_match"] = verify_tridi_primitives(path)
            if not all(report["vendored_primitives_match"].values()):
                raise RuntimeError("Vendored TriDi definitions differ from the pinned official source")
        reports.append(report)
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
