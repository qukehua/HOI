"""Train one model on mixed observed-feature tasks, with restartable checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from .controls import sample_controls
from .data import HOIDataset
from .diffusion import HOIDiffusion
from .model import ModelConfig, UnifiedHOIDenoiser
from .runtime import (device_for, move_batch, restore_rng, rng_state, save_checkpoint,
                      seed_everything, write_json)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional until installed
    tqdm = None


def make_loader(dataset, config, epoch, shuffle=True):
    generator = torch.Generator().manual_seed(config.get("seed", 42) + epoch)
    sampler = None
    if shuffle and config.get("balance_datasets", True):
        labels = [dataset.records[i]["dataset"] for i, _, _ in dataset.windows]
        count = Counter(labels)
        sampler = WeightedRandomSampler([1 / count[label] for label in labels], len(labels),
                                        replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=config.get("batch_size", 8),
                      shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=config.get("workers", 0), generator=generator, drop_last=False)


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source, 1 - decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def _scalar_metrics(metrics):
    return {key: value for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}


def init_wandb(config, output, run_info):
    if not config.get("use_wandb", False):
        return None
    try:
        import wandb
    except ImportError:
        print("WandB is enabled but wandb is not installed. "
              "Install it with `pip install wandb` or `pip install '.[wandb]'`.", flush=True)
        return None

    run_name = config.get("wandb_run_name") or output.name
    kwargs = {
        "project": config.get("wandb_project", "unified-hoi"),
        "name": run_name,
        "mode": config.get("wandb_mode", "online"),
        "config": {**config, "run_info": run_info},
        "dir": str(output),
    }
    entity = config.get("wandb_entity")
    if entity not in (None, "", "null", "None"):
        kwargs["entity"] = entity
    if config.get("wandb_id"):
        kwargs["id"] = config["wandb_id"]
        kwargs["resume"] = config.get("wandb_resume", "allow")
    try:
        return wandb.init(**kwargs)
    except Exception as exc:
        print(f"WandB initialization failed, continuing without WandB: {exc}", flush=True)
        return None


def log_wandb(run, payload, step):
    if run is None:
        return
    run.log(payload, step=step)


def run_training(config, resume=None):
    config = deepcopy(config)
    output = Path(config.get("output", "runs/unified"))
    if not resume and (output / "last.pt").exists():
        raise FileExistsError(f"Training checkpoint already exists in {output}; use --resume or a new --output")
    model_config = ModelConfig(**config.get("model", {}))
    config["model"] = model_config.asdict()
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = device_for(config.get("device", "auto"))
    dataset = HOIDataset(config["manifest"], "train", config.get("window", 120),
                         config.get("stride", 60), model_config.text_dim)
    if not len(dataset):
        raise ValueError("No training windows; preprocess data and check train split first")
    validation = HOIDataset(config["manifest"], "val", config.get("window", 120),
                            config.get("window", 120), model_config.text_dim)
    model = UnifiedHOIDenoiser(model_config)
    if not resume:
        model.normalizer.fit(make_loader(dataset, config, 0, shuffle=False))
    model.to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-4),
                                 weight_decay=config.get("weight_decay", .01))
    diffusion = HOIDiffusion(config.get("diffusion_steps", 1000))
    start_epoch = next_batch = step = 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        # Changing these changes the data order, loss or conditional task distribution.
        for key in ("model", "manifest", "window", "stride", "batch_size", "seed", "learning_rate",
                    "weight_decay", "balance_datasets", "holdout_signatures", "allowed_patterns",
                    "diffusion_steps", "geometry_weight", "text_dropout", "ema_decay", "gradient_clip"):
            if saved["config"].get(key) != config.get(key):
                raise ValueError(f"Resume configuration changed {key}; start a new run instead")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        start_epoch, next_batch, step = saved["epoch"], saved["next_batch"], saved["step"]
        restore_rng(saved["rng"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    run_info = {
        "parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
        "train_sequences": len(dataset.records), "train_windows": len(dataset),
        "validation_windows": len(validation), "torch": torch.__version__,
        "training_from_scratch": True, "physical_execution_validated": False,
        "reference_repos": {"tridi": "afa9631dc2b3a250588ab64026eeaa37f18f0d38",
                           "kimodo": "58e781898b3d7e328a676a75d3e338c45dce3ad9"}}
    write_json(output / "run_info.json", run_info)
    max_steps = config.get("max_steps", 100000)
    wandb_run = init_wandb(config, output, run_info)
    show_progress = config.get("progress", True) and tqdm is not None
    if config.get("progress", True) and tqdm is None:
        print("Progress bar requested but tqdm is not installed. "
              "Install it with `pip install tqdm`.", flush=True)

    def checkpoint(epoch, batch_index):
        save_checkpoint(output / "last.pt", {"schema_version": 1, "config": config, "model": model.state_dict(),
                        "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "epoch": epoch, "next_batch": batch_index, "rng": rng_state()})

    if step >= max_steps:
        if wandb_run is not None:
            wandb_run.finish()
        return str(Path(resume).resolve())

    progress = None
    if show_progress:
        progress = tqdm(total=max_steps, initial=step, desc="train", unit="step", dynamic_ncols=True)

    def emit(message):
        if progress is not None:
            progress.write(message)
        else:
            print(message, flush=True)

    try:
        epoch = start_epoch
        while step < max_steps:
            loader = make_loader(dataset, config, epoch)
            for index, cpu_batch in enumerate(loader):
                if epoch == start_epoch and index < next_batch:
                    continue
                model.train()
                batch = move_batch(cpu_batch, device)
                controls = sample_controls(batch, batch["valid_frames"],
                                           holdout_signatures=config.get("holdout_signatures", ()),
                                           allowed_patterns=config.get("allowed_patterns"))
                keep_text = torch.rand(len(batch["human"]), 1, device=device) >= config.get("text_dropout", .1)
                batch["text_features"] = batch["text_features"] * keep_text
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = diffusion.training_loss(model, batch, controls,
                                                       geometry_weight=config.get("geometry_weight", .05))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}; checkpoint remains last finite step")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.))
                if not torch.isfinite(grad):
                    raise FloatingPointError(f"Non-finite gradient at step {step}")
                optimizer.step()
                update_ema(ema, model, config.get("ema_decay", .999))
                step += 1
                metrics.update(step=step, epoch=epoch, gradient_norm=float(grad),
                               signatures=list(controls.signatures))
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        h=f"{metrics.get('denoise_human', 0):.3f}",
                        o=f"{metrics.get('denoise_object', 0):.3f}",
                        epoch=epoch,
                        refresh=False,
                    )
                if step % config.get("log_every", 20) == 0 or step == 1:
                    emit(json.dumps(metrics))
                    with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics) + "\n")
                    log_wandb(wandb_run, {f"train/{key}": value
                                          for key, value in _scalar_metrics(metrics).items()
                                          if key not in {"step"}}, step=step)
                if len(validation) and step % config.get("validate_every", 1000) == 0:
                    original_rng = rng_state()
                    seed_everything(config.get("seed", 42) + 1000000)
                    losses = []
                    with torch.no_grad():
                        for val_index, val in enumerate(make_loader(validation, config, 0, shuffle=False)):
                            if val_index >= config.get("validation_batches", 20):
                                break
                            val = move_batch(val, device)
                            control = sample_controls(val, val["valid_frames"],
                                                      holdout_signatures=config.get("holdout_signatures", ()),
                                                      allowed_patterns=config.get("allowed_patterns"))
                            _, result = diffusion.training_loss(ema, val, control,
                                                               config.get("geometry_weight", .05))
                            losses.append(result["loss"])
                    restore_rng(original_rng)
                    val_payload = {"step": step, "loss": sum(losses) / len(losses),
                                   "batches": len(losses)}
                    with (output / "validation.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(val_payload) + "\n")
                    emit(json.dumps({"validation": val_payload}))
                    log_wandb(wandb_run, {"val/loss": val_payload["loss"],
                                          "val/batches": val_payload["batches"],
                                          "train/epoch": epoch}, step=step)
                if step % config.get("save_every", 1000) == 0 or step == max_steps:
                    checkpoint(epoch, index + 1)
                if step >= max_steps:
                    break
            epoch += 1
            next_batch = 0
    finally:
        if progress is not None:
            progress.close()
        if wandb_run is not None:
            wandb_run.finish()
    return str((output / "last.pt").resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", help="Trusted checkpoint produced by this project")
    parser.add_argument("--wandb", dest="use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--progress", dest="progress", action="store_true",
                        help="Show a tqdm training progress bar")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="Disable the tqdm training progress bar")
    parser.set_defaults(use_wandb=None, progress=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for field in ("manifest", "output", "max_steps"):
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    if args.use_wandb is not None:
        config["use_wandb"] = args.use_wandb
    if args.progress is not None:
        config["progress"] = args.progress
    for field, key in (("wandb_project", "wandb_project"), ("wandb_run_name", "wandb_run_name"),
                       ("wandb_entity", "wandb_entity"), ("wandb_mode", "wandb_mode")):
        value = getattr(args, field)
        if value is not None:
            config[key] = value
    print(run_training(config, args.resume))


if __name__ == "__main__":
    main()
