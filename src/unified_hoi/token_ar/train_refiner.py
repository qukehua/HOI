"""Train a small continuous refiner on cached frozen AR errors (single GPU/CPU)."""
import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from ..runtime import save_checkpoint, seed_everything, write_json
from .body import BodyGeometry, digest_file
from .refinement import ContinuousRefiner, KIND, RefinerConfig, refinement_loss, refiner_payload
from .refinement_data import RefinementDataset, to_device


@torch.no_grad()
def validate(model, dataset, geometry, config, device):
    model.eval()
    totals = {}
    for i in range(len(dataset)):
        item = to_device(dataset[i], device)
        loss, stats, _ = refinement_loss(model, **item, geometry=geometry,
                                         weights=config.get("loss_weights"),
                                         geometry_frames=config.get("geometry_frames", 16))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite refinement validation loss")
        frames = len(item["coarse"]["human"])
        row = totals.setdefault(item["mode"], {"frames": 0, "loss": 0., **{k: 0. for k in stats}})
        row["frames"] += frames
        for key, value in {"loss": float(loss), **stats}.items():
            row[key] += value * frames
    results = {mode: {k: v / row["frames"] for k, v in row.items() if k != "frames"}
               for mode, row in totals.items()}
    return sum(row["loss"] for row in results.values()) / len(results), results


def run(config, resume=None):
    config = dict(config)
    if config.get("kind") != KIND:
        raise ValueError("Use a continuous-refinement training configuration")
    config.setdefault("schedule_steps", config["max_steps"])
    for key in ("max_steps", "schedule_steps", "gradient_accumulation_steps", "validate_every", "log_every"):
        if config.get(key, 1) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= config.get("warmup_steps", 100) < config["schedule_steps"]:
        raise ValueError("warmup_steps must be smaller than schedule_steps")
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Choose a fresh refiner output or explicitly resume")
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = config.get("device", "cuda:0")
    train = RefinementDataset(config["cache"], "train", config.get("manifest"))
    val = RefinementDataset(config["cache"], "val", config.get("manifest"))
    source = train.metadata | {"cache_index_sha256": digest_file(Path(config["cache"]) / "index.jsonl")}
    geometry = BodyGeometry(source["dataset"], config.get("contact_support", source["contact_support"]),
                            config.get("body_models", "data/smplx_models")).to(device).eval().requires_grad_(False)
    if (geometry.support_hash != source["support_sha256"]
            or {k: v.asset_hash for k, v in geometry.layers.items()} != source["body_assets"]):
        raise ValueError("Refiner geometry does not match cached AR data")
    model = ContinuousRefiner(RefinerConfig(dataset=source["dataset"], **config.get("model", {}))).to(device)
    for key in ("human", "object"):
        for name in ("mean", "std"):
            value = torch.tensor(source["normalization"][key][name], device=device)
            if not torch.isfinite(value).all() or (name == "std" and (value <= 0).any()):
                raise ValueError("Invalid cached normalization")
            getattr(model, key + "_" + name).copy_(value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-4),
                                 weight_decay=config.get("weight_decay", .01))
    warmup, duration = config.get("warmup_steps", 100), config["schedule_steps"]

    def factor(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        fraction = min(1., (step - warmup) / max(duration - warmup, 1))
        return .1 + .9 * (1 + math.cos(math.pi * fraction)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    step, best, bad_validations = 0, float("inf"), 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        mutable = {"max_steps", "output", "device", "cpu_threads", "log_every", "cache", "manifest", "contact_support", "body_models"}
        if (saved.get("kind") != KIND or saved["source"] != source
                or {k: v for k, v in saved["config"].items() if k not in mutable}
                != {k: v for k, v in config.items() if k not in mutable}):
            raise ValueError("Resume changed refiner data or training semantics")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        generator.set_state(saved["sampling_rng"])
        step, best, bad_validations = saved["step"], saved["best_val"], saved["bad_validations"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    if not resume:
        score, details = validate(model, val, geometry, config, device)
        write_json(output / "initial_validation.json", dict(loss=score, modes=details,
                    note="Zero-initialized residual; unrefined AR baseline up to rotation projection/epsilon."))
        best = score
        save_checkpoint(output / "best.pt", refiner_payload(model, source, config=config, step=0,
                        best_val=best, bad_validations=0, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), sampling_rng=generator.get_state(), validation=details))
    accumulation = config.get("gradient_accumulation_steps", 8)
    while step < config["max_steps"]:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, means = 0., {}
        for _ in range(accumulation):
            index = int(torch.randint(len(train), (), generator=generator))
            item = to_device(train[index], device)
            loss, stats, _ = refinement_loss(model, **item, geometry=geometry,
                                             weights=config.get("loss_weights"),
                                             geometry_frames=config.get("geometry_frames", 16))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite refiner training loss")
            (loss / accumulation).backward()
            total += float(loss.detach()) / accumulation
            for key, value in stats.items():
                means[key] = means.get(key, 0.) + value / accumulation
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.))
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite refiner gradient")
        optimizer.step()
        scheduler.step()
        step += 1
        if step == 1 or step % config.get("log_every", 20) == 0:
            print(json.dumps(dict(step=step, loss=total, lr=scheduler.get_last_lr()[0], **means)), flush=True)
        if step % config.get("validate_every", 250) == 0 or step == config["max_steps"]:
            score, details = validate(model, val, geometry, config, device)
            improved = score < best
            best = min(best, score)
            bad_validations = 0 if improved else bad_validations + 1
            payload = refiner_payload(model, source, config=config, step=step, best_val=best,
                                      bad_validations=bad_validations, optimizer=optimizer.state_dict(),
                                      scheduler=scheduler.state_dict(), sampling_rng=generator.get_state(), validation=details)
            save_checkpoint(output / "last.pt", payload)
            if improved:
                save_checkpoint(output / "best.pt", payload)
            row = dict(step=step, validation_loss=score, best_val=best, modes=details)
            with (output / "validation.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
            patience = config.get("early_stopping_patience", 8)
            if patience > 0 and bad_validations >= patience:
                print("Early stopping on full-cache validation geometry loss", flush=True)
                break
    return dict(output=str(output), step=step, best_val=best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--cache")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key in ("cache", "output", "device", "max_steps"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    print(run(config, args.resume))


if __name__ == "__main__":
    main()
