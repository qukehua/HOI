"""Validate prepared native data, training its isolated contact codec if missing."""
import argparse
from pathlib import Path

import torch

from unified_hoi.parametric.contact import train_codec, load_codec
from unified_hoi.parametric.body import digest_file
from unified_hoi.data import read_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("behave", "omomo"), required=True)
    p.add_argument("--codec-steps", type=int, default=3000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    root = Path("data/parametric_assets") / args.dataset
    manifest = Path(f"data/processed/{args.dataset}_parametric.jsonl")
    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing {manifest}. Restore the prepared native manifest and records from your transfer/backup. "
            "This checkout no longer contains the legacy raw-data conversion pipeline.")
    if not (root / "contact_ae/best.pt").exists():
        train_codec(manifest, root / "contact_ae", args.dataset, root / "contact_support.npz",
                    steps=args.codec_steps, device=args.device)
    support_hash = digest_file(root / "contact_support.npz")
    rows = read_manifest(manifest)
    if (not rows or any(row["dataset"] != args.dataset or row.get("support_sha256") != support_hash
                        or not Path(row["path"]).exists() for row in rows)):
        raise ValueError("Prepared manifest is incomplete or has mismatched dataset/contact support")
    load_codec(root / "contact_ae/best.pt", args.dataset, support_hash)
    print(f"Ready: {manifest}; isolated contact codec: {root / 'contact_ae/best.pt'}")


if __name__ == "__main__":
    main()
