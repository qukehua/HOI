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


def init_swanlab(config, output, run_info):
    if not config.get("use_swanlab", False):
        return None
    try:
        import swanlab
    except ImportError:
        print("SwanLab is enabled but swanlab is not installed. "
              "Install it with `pip install swanlab` or `pip install '.[swanlab]'`.", flush=True)
        return None

    api_key = config.get("swanlab_api_key")
    if api_key in (None, "", "null", "None"):
        import os
        api_key = os.environ.get("SWANLAB_API_KEY")
    if api_key:
        try:
            swanlab.login(api_key=api_key, save=True)
        except Exception as exc:
            print(f"SwanLab login failed, continuing without SwanLab: {exc}", flush=True)
            return None

    run_name = config.get("swanlab_run_name") or output.name
    logged_config = {key: value for key, value in config.items() if key != "swanlab_api_key"}
    kwargs = {
        "project": config.get("swanlab_project", "HOI"),
        "experiment_name": run_name,
        "mode": config.get("swanlab_mode", "online"),
        "config": {**logged_config, "run_info": run_info},
        "logdir": str(output / "swanlog"),
    }
    workspace = config.get("swanlab_workspace")
    if workspace not in (None, "", "null", "None"):
        kwargs["workspace"] = workspace
    if config.get("swanlab_id"):
        kwargs["id"] = config["swanlab_id"]
        kwargs["resume"] = config.get("swanlab_resume", "allow")
    try:
        return swanlab.init(**kwargs)
    except Exception as exc:
        print(f"SwanLab initialization failed, continuing without SwanLab: {exc}", flush=True)
        return None


def log_swanlab(run, payload, step):
    if run is None:
        return
    run.log(payload, step=step)


def finish_swanlab(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass


def run_training(config, resume=None):
    config = deepcopy(config)
    # Omitted switches preserve the historical behavior: use caches when present.
    text_condition = config.get("text_condition", True)
    if not isinstance(text_condition, bool):
        raise ValueError("text_condition must be a YAML boolean (true or false)")
    output = Path(config.get("output", "runs/unified"))
    if not resume and (output / "last.pt").exists():
        raise FileExistsError(f"Training checkpoint already exists in {output}; use --resume or a new --output")
    model_config = ModelConfig(**config.get("model", {}))
    config["model"] = model_config.asdict()
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = device_for(config.get("device", "auto"))
    dataset = HOIDataset(config["manifest"], "train", config.get("window", 120),
                         config.get("stride", 60), model_config.text_dim,
                         text_condition=text_condition)
    if not len(dataset):
        raise ValueError("No training windows; preprocess data and check train split first")
    if config.get("text_condition") is True and not any(
            entry.get("text_features_path") for entry in dataset.records):
        raise ValueError("text_condition is enabled but the train split has no text_features_path; "
                         "run scripts/cache_text.py and use its output manifest, "
                         "or set text_condition: false")
    validation = HOIDataset(config["manifest"], "val", config.get("window", 120),
                            config.get("window", 120), model_config.text_dim,
                            text_condition=text_condition)
    model = UnifiedHOIDenoiser(model_config)
    if not resume:
        model.normalizer.fit(make_loader(dataset, config, 0, shuffle=False))
    model.to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-4),
                                 weight_decay=config.get("weight_decay", .01))
    diffusion = HOIDiffusion(config.get("diffusion_steps", 1000))
    start_epoch = next_batch = step = 0
    best_val_loss = float("inf")
    best_step = None
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["config"].get("text_condition", True) != text_condition:
            raise ValueError("Resume configuration changed text_condition; start a new run instead")
        # Changing these changes the data order, loss or conditional task distribution.
        for key in ("model", "manifest", "window", "stride", "batch_size", "seed", "learning_rate",
                    "weight_decay", "balance_datasets", "holdout_signatures", "allowed_patterns",
                    "diffusion_steps", "geometry_weight", "losses", "text_dropout", "ema_decay",
                    "gradient_clip"):
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
        if saved.get("best_val_loss") is not None:
            best_val_loss = float(saved["best_val_loss"])
            best_step = saved.get("best_step")
    output.mkdir(parents=True, exist_ok=True)
    if best_step is None and (output / "best_info.json").exists():
        best_info = json.loads((output / "best_info.json").read_text(encoding="utf-8"))
        best_val_loss = float(best_info["loss"])
        best_step = best_info.get("step")
    write_json(output / "config.json", config)
    save_best = config.get("save_best", True)
    run_info = {
        "parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
        "train_sequences": len(dataset.records), "train_windows": len(dataset),
        "validation_windows": len(validation), "torch": torch.__version__,
        "text_condition": text_condition,
        "training_from_scratch": True, "physical_execution_validated": False,
        "save_best": save_best,
        "reference_repos": {"tridi": "afa9631dc2b3a250588ab64026eeaa37f18f0d38",
                           "kimodo": "58e781898b3d7e328a676a75d3e338c45dce3ad9"}}
    write_json(output / "run_info.json", run_info)
    max_steps = config.get("max_steps", 100000)
    swanlab_run = init_swanlab(config, output, run_info)
    show_progress = config.get("progress", True) and tqdm is not None
    if config.get("progress", True) and tqdm is None:
        print("Progress bar requested but tqdm is not installed. "
              "Install it with `pip install tqdm`.", flush=True)
    if save_best and not len(validation):
        print("save_best is enabled but the validation split is empty; "
              "best.pt will not be written.", flush=True)

    def checkpoint_payload(epoch, batch_index):
        return {"schema_version": 1, "config": config, "model": model.state_dict(),
                "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "epoch": epoch, "next_batch": batch_index, "rng": rng_state(),
                "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
                "best_step": best_step}

    def checkpoint(epoch, batch_index):
        save_checkpoint(output / "last.pt", checkpoint_payload(epoch, batch_index))

    def save_best_checkpoint(epoch, batch_index, val_loss):
        nonlocal best_val_loss, best_step
        best_val_loss = float(val_loss)
        best_step = step
        payload = checkpoint_payload(epoch, batch_index)
        payload["is_best"] = True
        save_checkpoint(output / "best.pt", payload)
        write_json(output / "best_info.json", {
            "step": best_step, "epoch": epoch, "loss": best_val_loss,
            "batches": config.get("validation_batches", 20),
            "metric": "val/loss", "path": str((output / "best.pt").resolve())})
        if swanlab_run is not None:
            log_swanlab(swanlab_run, {"val/best_loss": best_val_loss, "val/best_step": best_step},
                        step=best_step)

    if step >= max_steps:
        finish_swanlab(swanlab_run)
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
                loss, metrics = diffusion.training_loss(
                    model, batch, controls,
                    geometry_weight=config.get("geometry_weight", .05),
                    loss_flags=config.get("losses"))
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
                    postfix = {
                        "loss": f"{metrics['loss']:.4f}",
                        "h": f"{metrics.get('denoise_human', 0):.3f}",
                        "o": f"{metrics.get('denoise_object', 0):.3f}",
                        "epoch": epoch,
                    }
                    if best_step is not None:
                        postfix["best"] = f"{best_val_loss:.4f}"
                    progress.set_postfix(**postfix, refresh=False)
                if step % config.get("log_every", 20) == 0 or step == 1:
                    emit(json.dumps(metrics))
                    with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics) + "\n")
                    log_swanlab(swanlab_run, {f"train/{key}": value
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
                            _, result = diffusion.training_loss(
                                ema, val, control,
                                geometry_weight=config.get("geometry_weight", .05),
                                loss_flags=config.get("losses"))
                            losses.append(result["loss"])
                    restore_rng(original_rng)
                    val_loss = sum(losses) / len(losses)
                    improved = save_best and val_loss < best_val_loss
                    val_payload = {"step": step, "loss": val_loss, "batches": len(losses),
                                   "best_val_loss": min(best_val_loss, val_loss) if save_best else None,
                                   "improved": improved}
                    with (output / "validation.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(val_payload) + "\n")
                    emit(json.dumps({"validation": val_payload}))
                    swanlab_payload = {"val/loss": val_payload["loss"],
                                       "val/batches": val_payload["batches"],
                                       "train/epoch": epoch}
                    if val_payload["best_val_loss"] is not None:
                        swanlab_payload["val/best_loss"] = val_payload["best_val_loss"]
                    log_swanlab(swanlab_run, swanlab_payload, step=step)
                    if improved:
                        save_best_checkpoint(epoch, index + 1, val_loss)
                        emit(json.dumps({"best_checkpoint": {"step": best_step, "loss": best_val_loss,
                                                             "path": str(output / "best.pt")}}))
                if step % config.get("save_every", 1000) == 0 or step == max_steps:
                    checkpoint(epoch, index + 1)
                if step >= max_steps:
                    break
            epoch += 1
            next_batch = 0
    finally:
        if progress is not None:
            progress.close()
        finish_swanlab(swanlab_run)
    return str((output / "last.pt").resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", help="Trusted checkpoint produced by this project")
    parser.add_argument("--swanlab", dest="use_swanlab", action="store_true",
                        help="Enable SwanLab logging")
    parser.add_argument("--no-swanlab", dest="use_swanlab", action="store_false",
                        help="Disable SwanLab logging")
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-run-name")
    parser.add_argument("--swanlab-workspace")
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline", "disabled"))
    parser.add_argument("--progress", dest="progress", action="store_true",
                        help="Show a tqdm training progress bar")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="Disable the tqdm training progress bar")
    parser.set_defaults(use_swanlab=None, progress=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for field in ("manifest", "output", "max_steps"):
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    if args.use_swanlab is not None:
        config["use_swanlab"] = args.use_swanlab
    if args.progress is not None:
        config["progress"] = args.progress
    for field, key in (("swanlab_project", "swanlab_project"),
                       ("swanlab_run_name", "swanlab_run_name"),
                       ("swanlab_workspace", "swanlab_workspace"),
                       ("swanlab_mode", "swanlab_mode")):
        value = getattr(args, field)
        if value is not None:
            config[key] = value
    print(run_training(config, args.resume))


if __name__ == "__main__":
    main()
