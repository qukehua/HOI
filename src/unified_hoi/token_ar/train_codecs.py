"""Pretrain the three temporal VQ-VAEs on one dataset's train split."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from ..runtime import move_batch, save_checkpoint, seed_everything, write_json
from . import ARCHITECTURE
from .body import digest_file
from .codecs import CodecBundle, CodecConfig, codec_payload
from .data import MotionDataset

def init_swanlab(config, output):
    if not config.get("use_swanlab", False):
        return None
    try:
        import swanlab
    except ImportError:
        print("SwanLab is enabled but swanlab is not installed.", flush=True)
        return None
    run_name = config.get("swanlab_run_name") or output.name
    kwargs = {
        "project": config.get("swanlab_project", "HOI"),
        "experiment_name": run_name,
        "mode": config.get("swanlab_mode", "online"),
        "config": {k: v for k, v in config.items() if k != "swanlab_api_key"},
        "logdir": str(output / "swanlog"),
    }
    workspace = config.get("swanlab_workspace")
    if workspace not in (None, "", "null", "None"):
        kwargs["workspace"] = workspace
    try:
        return swanlab.init(**kwargs)
    except Exception as exc:
        print(f"SwanLab initialization failed, continuing without SwanLab: {exc}", flush=True)
        return None


def log_swanlab(run, payload, step):
    if run is not None:
        run.log(payload, step=step)


def finish_swanlab(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


_MUTABLE = {
    "max_steps", "output", "device", "log_every", "cpu_threads", "use_swanlab",
    "swanlab_project", "swanlab_run_name", "swanlab_mode", "swanlab_workspace",
    "swanlab_id", "swanlab_resume", "swanlab_api_key",
}


@torch.no_grad()
def validate(model, loader, device, batches):
    model.eval()
    total, count = 0., 0
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        loss, _ = model.loss(move_batch(batch, device))
        total += float(loss) * len(batch["human"])
        count += len(batch["human"])
    if not count:
        raise ValueError("Empty codec validation split")
    return total / count


def run(config, resume=None):
    if config.get("architecture") != ARCHITECTURE:
        raise ValueError("Use a token_ar codec configuration")
    if "modality_steps" in config:
        from .vq_training import run_independent
        return run_independent(config, resume)
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Choose a new codec output directory or explicitly resume")
    if int(config["max_steps"]) < 1 or int(config.get("validation_batches", 16)) < 1:
        raise ValueError("Training/validation budgets must be positive")
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    support_hash = digest_file(config["contact_support"])
    kwargs = dict(manifest=config["manifest"], dataset=config["dataset"], support_hash=support_hash,
                  window=config.get("window", 120), include_geometry=False)
    train = MotionDataset(split="train", stride=config.get("stride", 60), **kwargs)
    val = MotionDataset(split="val", **kwargs)
    if not len(train) or not len(val):
        raise ValueError("Need nonempty train and val splits")
    cfg = CodecConfig(dataset=config["dataset"], **config.get("codec", {}))
    model = CodecBundle(cfg)
    model.fit_normalization(train)
    device = torch.device(config.get("device", "cuda"))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 2e-4))
    metadata = {"support_sha256": support_hash, "manifest_sha256": digest_file(config["manifest"]),
                "contact_threshold": .5, "config": config}
    step, epoch, skip, best = 0, 0, 0, float("inf")
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        for key in ("support_sha256", "manifest_sha256"):
            if saved[key] != metadata[key]:
                raise ValueError(f"Resume changed {key}")
        previous = {k: v for k, v in saved["config"].items() if k not in _MUTABLE}
        current = {k: v for k, v in config.items() if k not in _MUTABLE}
        if saved.get("architecture") != ARCHITECTURE or CodecConfig(**saved["codec_config"]) != cfg or previous != current:
            raise ValueError("Resume changed architecture, tokenizer or training semantics")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, epoch, skip, best = saved["step"], saved["epoch"], saved["next_batch"], saved["best_val"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    swan = init_swanlab(config, output)
    validation = DataLoader(val, batch_size=config.get("batch_size", 32), shuffle=False, num_workers=0)
    try:
        return _train(config, model, optimizer, train, device, output, validation, metadata, swan,
                      step, epoch, skip, best)
    finally:
        finish_swanlab(swan)


def _train(config, model, optimizer, train, device, output, validation, metadata, swan, step, epoch, skip, best):
    last_saved = None
    while step < config["max_steps"]:
        generator = torch.Generator().manual_seed(config.get("seed", 42) + epoch)
        order = torch.randperm(len(train), generator=generator).tolist()
        size = config.get("batch_size", 32)
        batches = [order[i:i + size] for i in range(0, len(order), size)]
        loader = DataLoader(train, batch_sampler=batches[skip:], generator=generator, num_workers=0)
        for index, batch in enumerate(loader, start=skip):
            model.train()
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, stats = model.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite VQ loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            step += 1
            reset_every = config.get("reset_unused_every", 100)
            if model.config.quantizer == "gradient" and reset_every and step % reset_every == 0:
                for key, codec in model.models.items():
                    unused = codec.usage == 0
                    codec.reset_unused(batch[key], batch["valid_frames"])
                    state = optimizer.state.get(codec.codebook.weight, {})
                    for field in ("exp_avg", "exp_avg_sq"):
                        if field in state:
                            state[field][unused] = 0
            if step % config.get("log_every", 20) == 0 or step == 1:
                row = {"step": step, "loss": float(loss.detach()), **stats}
                print(json.dumps(row), flush=True)
                log_swanlab(swan, {"train/loss": row["loss"], **{f"train/{k}": v for k, v in stats.items()}}, step)
            if step % config.get("validate_every", 1000) == 0 or step == config["max_steps"]:
                score = validate(model, validation, device, config.get("validation_batches", 16))
                improved = score < best
                best = min(best, score)
                payload = codec_payload(model, {**metadata, "optimizer": optimizer.state_dict(), "step": step,
                                                "epoch": epoch, "next_batch": index + 1, "best_val": best,
                                                "torch_rng": torch.get_rng_state(),
                                                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
                save_checkpoint(output / "last.pt", payload)
                if improved:
                    save_checkpoint(output / "best.pt", payload)
                last_saved = step
                print(json.dumps({"step": step, "validation_loss": score, "best_val": best}), flush=True)
                log_swanlab(swan, {"val/loss": score, "val/best": best}, step)
            if step >= config["max_steps"]:
                break
        epoch, skip = epoch + 1, 0
    return {"output": str(output), "step": step, "saved_step": last_saved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output")
    parser.add_argument("--swanlab", dest="use_swanlab", action="store_true")
    parser.add_argument("--no-swanlab", dest="use_swanlab", action="store_false")
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-run-name")
    parser.add_argument("--swanlab-workspace")
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline", "disabled"))
    parser.set_defaults(use_swanlab=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key in ("max_steps", "output"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.use_swanlab is not None:
        config["use_swanlab"] = args.use_swanlab
    for field, key in (("swanlab_project", "swanlab_project"),
                       ("swanlab_run_name", "swanlab_run_name"),
                       ("swanlab_workspace", "swanlab_workspace"),
                       ("swanlab_mode", "swanlab_mode")):
        value = getattr(args, field)
        if value is not None:
            config[key] = value
    print(run(config, args.resume))


if __name__ == "__main__":
    main()
