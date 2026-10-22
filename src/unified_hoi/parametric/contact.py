"""Dataset-specific trained 690 -> 128 -> 690 contact autoencoders.

Same dimensions as TriDi; independently trained latent spaces, explicitly not
claimed to be the unavailable official pretrained latent coordinates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..data import read_manifest
from ..runtime import save_checkpoint, seed_everything, write_json
from .body import digest_file


class ContactAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(690, 512), nn.SiLU(), nn.Linear(512, 256),
                                     nn.SiLU(), nn.Linear(256, 128))
        self.decoder = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 512),
                                     nn.SiLU(), nn.Linear(512, 690))

    def encode(self, contact):
        return self.encoder(contact)

    def decode_logits(self, latent):
        return self.decoder(latent)

    def decode(self, latent):
        return self.decode_logits(latent).sigmoid()


def load_codec(path, dataset, support_hash):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("dataset") != dataset or saved.get("support_sha256") != support_hash
            or saved.get("codec_version") != 1 or saved.get("steps", 0) < 1):
        raise ValueError("Contact AE dataset/support/version mismatch or untrained codec")
    codec = ContactAE()
    codec.load_state_dict(saved["model"])
    return codec.eval().requires_grad_(False), saved


def train_codec(manifest, output, dataset, support, steps=3000, batch_size=256, device="cpu", seed=42):
    if steps < 1 or batch_size < 1:
        raise ValueError("Positive training budget required")
    output = Path(output)
    if (output / "best.pt").exists():
        raise FileExistsError(output / "best.pt")
    seed_everything(seed)
    torch.set_num_threads(4)
    support_hash = digest_file(support)
    groups = {"train": [], "val": []}
    rows = read_manifest(manifest)
    if {r["dataset"] for r in rows} != {dataset}:
        raise ValueError("AE training must isolate datasets")
    for row in rows:
        if row["split"] not in groups:
            continue
        with np.load(row["path"], allow_pickle=False) as archive:
            if str(archive["support_sha256"]) != support_hash:
                raise ValueError("Contact support mismatch")
            groups[row["split"]].append(archive["surface_contact"])
    if not all(groups.values()):
        raise ValueError("Contact AE requires nonempty train and validation splits")
    train = torch.from_numpy(np.concatenate(groups["train"]))
    val = torch.from_numpy(np.concatenate(groups["val"]))
    del groups
    positive = train.float().sum(0)
    weights = ((len(train) - positive) / positive.clamp_min(1)).clamp(1, 30).to(device)
    model = ContactAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, best_step = float("inf"), None, 0
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for step in range(1, steps + 1):
        model.train()
        target = train[torch.randint(len(train), (batch_size,), generator=generator)].float().to(device)
        z = model.encode(target)
        loss = F.binary_cross_entropy_with_logits(model.decoder(z), target, pos_weight=weights) + 1e-4 * z.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 250 == 0 or step == steps:
            model.eval()
            losses = []
            with torch.no_grad():
                for part in val.split(1024):
                    target = part.float().to(device)
                    value = F.binary_cross_entropy_with_logits(model.decoder(model.encode(target)), target,
                                                               pos_weight=weights, reduction="sum")
                    losses.append(float(value))
            score = sum(losses) / val.numel()
            if score < best:
                best, best_step = score, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = {"step": step, "train_loss": float(loss.detach()), "validation_bce": score}
            history.append(row)
            print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    # Threshold selection uses validation only, never test contacts.
    counts = np.zeros((17, 3), np.int64)
    thresholds = np.linspace(.1, .9, 17)
    with torch.no_grad():
        for part in val.split(1024):
            probability = model.decode(model.encode(part.float().to(device))).cpu().numpy()
            target = part.numpy().astype(bool)
            for i, threshold in enumerate(thresholds):
                pred = probability >= threshold
                counts[i] += ((pred & target).sum(), (pred & ~target).sum(), (~pred & target).sum())
    scores = 2 * counts[:, 0] / np.maximum(2 * counts[:, 0] + counts[:, 1] + counts[:, 2], 1)
    index = int(scores.argmax())
    payload = {"model": best_state, "dataset": dataset, "codec_version": 1, "steps": best_step,
               "support_sha256": support_hash, "threshold": float(thresholds[index]),
               "origin": "independently_trained_on_dataset_train_split",
               "validation_f1": float(scores[index]), "seed": seed}
    save_checkpoint(output / "best.pt", payload)
    write_json(output / "report.json", {k: v for k, v in payload.items() if k != "model"} |
               {"train_frames": len(train), "validation_frames": len(val), "history": history,
                "official_tridi_latent_coordinates": False})
    return str(output / "best.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset", choices=("behave", "omomo"), required=True)
    p.add_argument("--support", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    print(train_codec(args.manifest, args.output, args.dataset, args.support, args.steps,
                      args.batch_size, args.device))


if __name__ == "__main__":
    main()
