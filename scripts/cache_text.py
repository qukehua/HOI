"""Cache frozen CLIP text features for verified annotations; no fake language vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import numpy as np
import torch
from unified_hoi.data import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New manifest path")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = CLIPTextModelWithProjection.from_pretrained(args.model).to(args.device).eval()
    cache = output.parent / "text_features"
    cache.mkdir(parents=True, exist_ok=True)
    entries = read_manifest(args.manifest)
    encoded = set()

    def cache_caption(caption):
        text = caption.get("text", "").strip()
        if text:
            key = hashlib.sha256((args.model + "\n" + text).encode()).hexdigest()
            target = cache / (key + ".npy")
            if not target.exists():
                tokens = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=77).to(args.device)
                with torch.no_grad():
                    embedding = model(**tokens).text_embeds[0].cpu().numpy().astype(np.float32)
                np.save(target, embedding)
            embedding = np.load(target, allow_pickle=False)
            if embedding.shape != (model.config.projection_dim,) or not np.isfinite(embedding).all() or not embedding.any():
                raise ValueError(f"Invalid cached embedding: {target}")
            caption["text_features_path"] = os.path.relpath(target, output.parent).replace("\\", "/")
            caption["text_encoder"] = args.model
            encoded.add(key)

    for number, entry in enumerate(entries, 1):
        entry["path"] = os.path.relpath(Path(entry["path"]).resolve(), output.parent).replace("\\", "/")
        cache_caption(entry)
        for variant in entry.get("text_variants", []):
            cache_caption(variant)
        if number % 100 == 0:
            print(f"Cached {number}/{len(entries)} records ({len(encoded)} unique descriptions)", flush=True)
    output.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
