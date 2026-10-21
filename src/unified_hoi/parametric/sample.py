"""Sample native human parameters and contact latents from a manifest reference."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import read_manifest
from ..runtime import move_batch
from .body import split_human, standard_parameters
from .data import ParametricDataset, load_record
from .diffusion import controls_for
from .train import load_run, prepare_batch


def sample(checkpoint, manifest, reference, output, mode, frames=120, start=0, steps=50,
           seed=42, device="cpu", text_conditioned=False):
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("Native output must end in .npz")
    if output.exists():
        raise FileExistsError(output)
    model, diffusion, geometry, saved = load_run(checkpoint, device)
    entry = next((e for e in read_manifest(manifest) if Path(e["path"]).resolve() == Path(reference).resolve()), None)
    if entry is None:
        raise ValueError("Reference must belong to the native manifest")
    dataset = ParametricDataset(manifest, entry["split"], saved["config"]["dataset"], geometry.support_hash,
                                frames, text_condition=text_conditioned, text_dim=model.config.text_dim)
    index = next(i for i, e in enumerate(dataset.records) if Path(e["path"]).resolve() == Path(reference).resolve())
    record = load_record(reference, saved["config"]["dataset"], geometry.support_hash)
    end = min(start + frames, len(record["human_params"]))
    if start < 0 or end <= start:
        raise ValueError("Reference window outside clip")
    dataset.windows = [(index, start, end)]
    batch = prepare_batch(model, move_batch(next(iter(DataLoader(dataset, batch_size=1))), device), geometry, diffusion.steps)
    if text_conditioned and not batch["text_available"].all():
        raise ValueError("No verified text features for requested text condition")
    masks = controls_for(batch, saved["config"]["dataset"], mode)
    result = diffusion.sample(model, batch, masks, steps, seed)
    with torch.no_grad():
        surface, joints = geometry(result["human"], batch["gender"])
        beta, pose, trans = split_human(result["human"], saved["config"]["dataset"])
        values = {"human_params": result["human"], "object": result["object"],
                  "contact_latent": result["contact"], "contact_probability": model.codec.decode(result["contact"]),
                  "joints73": joints, "contact_vertices": surface, "betas": beta,
                  "local_pose6d": pose, "smpl_transl": trans}
        arrays = {key: value[0, :end - start].cpu().numpy() for key, value in values.items()}
        for key in ("human", "object", "contact"):
            arrays[key + "_observed"] = torch.where(masks[key], batch[key], 0.)[0, :end - start].cpu().numpy()
            arrays[key + "_mask"] = masks[key][0, :end - start].cpu().numpy()
        arrays["surface_contact_observed"] = (record["surface_contact"][start:end] if mode[2] == "0"
                                              else np.zeros((end - start, 690), np.uint8))
    arrays.update(representation="parametric_v1", dataset=saved["config"]["dataset"],
                  mode=mode, gender=record["gender"], fps=record["fps"], timestamps=record["timestamps"][start:end],
                  support_sha256=geometry.support_hash, body_model_type=record["body_model_type"],
                  body_model_sha256=record["body_model_sha256"], sequence_id=record["sequence_id"], window_start=start,
                  codec_sha256=saved["codec_metadata"]["file_sha256"],
                  hand_source=record["hand_source"], codec_origin=saved["codec_metadata"]["origin"])
    arrays.update(standard_parameters(arrays["human_params"], saved["config"]["dataset"]))
    with np.load(saved["config"]["contact_support"], allow_pickle=False) as support:
        arrays["contact_vertex_indices"] = support["indices"].copy()
    arrays["contact_decode_threshold"] = np.float32(float(model.contact_threshold))
    arrays["contact_decoded_binary"] = arrays["contact_probability"] >= float(model.contact_threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return str(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "reference", "output", "mode"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--text-conditioned", action="store_true")
    torch.set_num_threads(4)
    from ..runtime import device_for
    args = vars(p.parse_args())
    args["device"] = device_for(args["device"])
    print(sample(**args))


if __name__ == "__main__":
    main()
