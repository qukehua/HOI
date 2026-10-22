"""Autoregressively complete missing H/O/C modalities, then decode motion VQ tokens."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import read_manifest
from ..runtime import move_batch
from . import ARCHITECTURE, MODALITIES, MODES
from .body import BodyGeometry, digest_file, standard_parameters
from .checkpoint import load_run
from .codecs import canonicalize, load_codecs
from .data import MotionDataset, load_record


@torch.no_grad()
def complete(model, codecs, builder, batch, mode, text=None, seed=42, temperature=1., top_k=50):
    if mode not in MODES or len(batch["human"]) != 1:
        raise ValueError("Sampling expects one window and a valid H/O/C mask")
    n = int(batch["valid_frames"][0].sum())
    token_count = (n + codecs.config.stride - 1) // codecs.config.stride
    # Do not even run the encoders on unavailable ground-truth modalities.
    known = {k: codecs.models[k].encode(batch[k], batch["valid_frames"])[0].tolist()
             for k, bit in zip(MODALITIES, mode) if bit == "0"}
    prompt = batch["text"][0] if text is None else text
    example, order = builder.example(known, mode, prompt, n, float(batch["fps"][0]),
                                     "male" if batch["gender"][0] else "female", codecs.config.dataset,
                                     training=False, token_count=token_count)
    generated = model.generate_codes(example["input_ids"], order, batch["object_bps"], seed, temperature, top_k)
    values = {}
    for k, bit in zip(MODALITIES, mode):
        values[k] = (codecs.models[k].decode(torch.tensor(generated[k], device=batch[k].device), n)
                     if bit == "1" else batch[k][0, :n].clone())
    h, o = canonicalize(values["human"], values["object"], codecs.config.dataset)
    if mode[0] == "1":
        values["human"] = h
    if mode[1] == "1":
        values["object"] = o
    return values, {k: generated[k] if mode[i] == "1" else known[k] for i, k in enumerate(MODALITIES)}


def sample(checkpoint, manifest, reference, output, mode, frames=120, start=0, seed=42,
           device="cpu", dtype="float32", text_conditioned=False, text=None, temperature=1., top_k=50,
           contact_support=None, body_models="data/smplx_models"):
    destination = Path(output)
    if destination.suffix != ".npz" or destination.exists():
        raise ValueError("Choose a new .npz output file")
    if frames < 1 or start < 0:
        raise ValueError("Invalid window")
    model, builder, saved = load_run(checkpoint, device, dtype)
    dataset_name = saved["config"]["dataset"]
    support = contact_support or saved["config"]["contact_support"]
    if digest_file(support) != saved["support_sha256"]:
        raise ValueError("Contact surface support changed")
    codecs, _ = load_codecs(Path(checkpoint) / "codecs.pt", dataset_name, saved["support_sha256"], device)
    geometry = BodyGeometry(dataset_name, support, body_models).to(device)
    entry = next((r for r in read_manifest(manifest) if Path(r["path"]).resolve() == Path(reference).resolve()), None)
    if entry is None:
        raise ValueError("Reference must belong to the native manifest")
    dataset = MotionDataset(manifest, entry["split"], dataset_name, geometry.support_hash, frames)
    record = load_record(reference, dataset_name, geometry.support_hash)
    if geometry.layers[str(record["gender"])].asset_hash != str(record["body_model_sha256"]):
        raise ValueError("Body assets do not match the native reference")
    end = min(start + frames, len(record["human_params"]))
    if end <= start:
        raise ValueError("Window outside reference sequence")
    index = next(i for i, row in enumerate(dataset.records) if Path(row["path"]).resolve() == Path(reference).resolve())
    dataset.windows = [(index, start, end)]
    batch = move_batch(next(iter(DataLoader(dataset, batch_size=1))), device)
    if text is None:
        text = batch["text"][0] if text_conditioned else ""
        if text_conditioned and not text.strip():
            raise ValueError("Requested text condition is missing")
    values, tokens = complete(model, codecs, builder, batch, mode, text, seed, temperature, top_k)
    with torch.no_grad():
        vertices, joints = geometry(values["human"][None], batch["gender"])
    arrays = {"human_params": values["human"].cpu().numpy(), "object": values["object"].cpu().numpy(),
              "contact_probability": values["contact"].cpu().numpy(),
              "contact_vertices": vertices[0].cpu().numpy(), "joints73": joints[0].cpu().numpy()}
    for key, bit in zip(MODALITIES, mode):
        arrays[key + "_tokens"] = np.asarray(tokens[key], np.int64)
        arrays[key + "_mask"] = np.full(values[key].shape, bit == "0", bool)
        arrays[key + "_observed"] = batch[key][0, :end - start].cpu().numpy() if bit == "0" else np.zeros(values[key].shape, np.float32)
    arrays.update(standard_parameters(arrays["human_params"], dataset_name))
    arrays.update(architecture=ARCHITECTURE, representation="parametric_v1", dataset=dataset_name, mode=mode,
                  gender=record["gender"], fps=record["fps"], timestamps=record["timestamps"][start:end],
                  body_model_type=record["body_model_type"], body_model_sha256=record["body_model_sha256"],
                  support_sha256=geometry.support_hash, codec_sha256=saved["codec_sha256"],
                  sequence_id=record["sequence_id"], window_start=start, hand_source=record["hand_source"],
                  text_prompt=text, seed=seed, temperature=temperature, top_k=top_k,
                  contact_decode_threshold=.5, contact_decoded_binary=arrays["contact_probability"] >= .5,
                  checkpoint_metadata_sha256=digest_file(Path(checkpoint) / "metadata.json"))
    with np.load(support, allow_pickle=False) as archive:
        arrays["contact_vertex_indices"] = archive["indices"].copy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "reference", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--text-conditioned", action="store_true")
    parser.add_argument("--text", help="Explicit user prompt; overrides the reference caption")
    parser.add_argument("--contact-support")
    parser.add_argument("--body-models", default="data/smplx_models")
    torch.set_num_threads(4)
    print(sample(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
