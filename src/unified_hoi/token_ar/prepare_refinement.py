"""Cache actual frozen AR rollouts on train/val for continuous refinement."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..runtime import move_batch, write_json
from . import MODALITIES, MODES
from .body import BodyGeometry, digest_file
from .checkpoint import load_run
from .codecs import load_codecs
from .data import MotionDataset
from .sample import complete

CACHE_KIND = "token_ar_refinement_cache_v1"


def prepare(checkpoint, output, device="cuda:0", dtype="bfloat16", modes=MODES,
            temperature=0., top_k=50, text_conditioned=True, seed=42, limit=None,
            body_models="data/smplx_models", resume=False):
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("Specify distinct H/O/C modes")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    output, checkpoint = Path(output), Path(checkpoint)
    if output.exists() and not resume:
        raise FileExistsError("Choose a new cache or use --resume")
    if resume and not (output / "metadata.json").exists():
        raise ValueError("Cannot resume a cache without metadata")
    model, builder, saved = load_run(checkpoint, device, dtype)
    model.requires_grad_(False)
    cfg = saved["config"]
    codecs, _ = load_codecs(checkpoint / "codecs.pt", cfg["dataset"], saved["support_sha256"], device)
    geometry = BodyGeometry(cfg["dataset"], cfg["contact_support"], body_models)
    if geometry.support_hash != saved["support_sha256"]:
        raise ValueError("Contact support changed")
    if digest_file(cfg["manifest"]) != saved["manifest_sha256"]:
        raise ValueError("AR manifest changed")
    metadata = dict(kind=CACHE_KIND, dataset=cfg["dataset"], manifest=cfg["manifest"],
                    manifest_sha256=saved["manifest_sha256"], contact_support=cfg["contact_support"],
                    support_sha256=saved["support_sha256"], codec_sha256=saved["codec_sha256"],
                    ar_metadata_sha256=digest_file(checkpoint / "metadata.json"),
                    body_assets={k: v.asset_hash for k, v in geometry.layers.items()},
                    window=cfg.get("window", 120), train_stride=cfg.get("stride", 60),
                    temperature=temperature, top_k=top_k, text_conditioned=text_conditioned,
                    dtype=dtype, seed=seed, modes=list(modes), limit_per_split=limit,
                    normalization={key: {name: getattr(codecs.models[key], name).cpu().tolist()
                                         for name in ("mean", "std")} for key in ("human", "object")})
    if resume and json.loads((output / "metadata.json").read_text(encoding="utf-8")) != metadata:
        raise ValueError("Cache settings/source changed; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metadata.json", metadata)
    index_path = output / "index.jsonl"
    done = {}
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["split"], row["window_index"], row["mode"])
            if key in done or digest_file(output / row["file"]) != row["sha256"]:
                raise ValueError("Duplicate or corrupted cache entry")
            done[key] = row
    count = 0
    for split in ("train", "val"):
        dataset = MotionDataset(cfg["manifest"], split, cfg["dataset"], saved["support_sha256"],
                                metadata["window"], metadata["train_stride"] if split == "train" else None)
        total = len(dataset) if limit is None else min(limit, len(dataset))
        if not total:
            raise ValueError("Need nonempty train and val splits")
        for index in range(total):
            if all((split, index, mode) in done for mode in modes):
                count += len(modes)
                continue
            # Deterministic caption selection makes resume equivalent to an uninterrupted cache run.
            torch.manual_seed(seed + index * 1009 + (100000000 if split == "val" else 0))
            batch = move_batch(next(iter(DataLoader(dataset, batch_sampler=[[index]]))), device)
            record_index, start, end = dataset.windows[index]
            gender = "male" if int(batch["gender"][0]) else "female"
            if batch["body_model_sha256"][0] != metadata["body_assets"][gender]:
                raise ValueError("Native reference body assets changed")
            prompt = batch["text"][0] if text_conditioned else ""
            for mode in modes:
                if (split, index, mode) in done:
                    count += 1
                    continue
                values, _ = complete(model, codecs, builder, batch, mode, prompt,
                                     seed + index * 1009, temperature, top_k)
                filename = f"{split}_{index:06d}_{mode}.npz"
                path = output / filename
                temporary = path.with_suffix(".partial")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **{k: values[k].cpu().numpy() for k in MODALITIES})
                temporary.replace(path)
                entry = dataset.records[record_index]
                row = dict(file=filename, sha256=digest_file(path), split=split, window_index=index,
                           record_index=record_index, start=start, end=end, mode=mode,
                           sequence_id=entry["sequence_id"], source_sequence_id=entry.get("source_sequence_id", entry["sequence_id"]),
                           reference_sha256=digest_file(entry["path"]),
                           object_mesh_sha256=digest_file(entry["object_mesh_path"]), text_prompt=prompt)
                with index_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                count += 1
            print(f"Cached {split} window {index + 1}/{total}", flush=True)
    write_json(output / "COMPLETE", dict(entries=count, index_sha256=digest_file(index_path),
                                         metadata_sha256=digest_file(output / "metadata.json")))
    return {"output": str(output), "entries": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--no-text", dest="text_conditioned", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Smoke only: maximum windows per split")
    parser.add_argument("--body-models", default="data/smplx_models")
    parser.add_argument("--resume", action="store_true")
    torch.set_num_threads(4)
    print(prepare(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
