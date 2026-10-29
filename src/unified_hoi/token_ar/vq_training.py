"""Independent H/O/C update budgets, exact full batches and resumable EMA state."""
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..runtime import move_batch, save_checkpoint, seed_everything, write_json
from . import MODALITIES
from .body import digest_file
from .codecs import CodecBundle, CodecConfig, codec_payload
from .data import MotionDataset


def run_independent(config, resume=None):
    from .train_codecs import _MUTABLE, finish_swanlab, init_swanlab, log_swanlab, validate
    budgets = config["modality_steps"]
    if set(budgets) != set(MODALITIES) or any(type(v) is not int or v < 1 for v in budgets.values()):
        raise ValueError("modality_steps must supply positive integer budgets for human/object/contact")
    maximum = min(int(config.get("max_steps", max(budgets.values()))), max(budgets.values()))
    size = int(config.get("batch_size", 128))
    if min(maximum, size, config.get("validate_every", 1000), config.get("validation_batches", 16)) < 1:
        raise ValueError("Training and validation budgets must be positive")
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Choose a new VQ output directory or --resume")
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    support = digest_file(config["contact_support"])
    kwargs = dict(manifest=config["manifest"], dataset=config["dataset"], support_hash=support,
                  window=config.get("window", 120), include_geometry=False)
    train = MotionDataset(split="train", stride=config.get("stride", 60), **kwargs)
    val = MotionDataset(split="val", **kwargs)
    if len(train) < size or not len(val):
        raise ValueError("Need at least batch_size training windows and a nonempty val split")
    cfg = CodecConfig(dataset=config["dataset"], **config.get("codec", {}))
    model = CodecBundle(cfg)
    model.fit_normalization(train)
    device = torch.device(config.get("device", "cuda:0"))
    model.to(device)
    optimizers = {k: torch.optim.AdamW([p for p in v.parameters() if p.requires_grad],
                                      lr=config.get("learning_rate", 2e-4),
                                      weight_decay=config.get("weight_decay", 0.)) for k, v in model.models.items()}
    metadata = dict(config=config, support_sha256=support, manifest_sha256=digest_file(config["manifest"]),
                    trainer="independent_vq_v2", contact_threshold=.5, selection="final_checkpoint",
                    modality_budgets=budgets)
    step, epoch, skip = 0, 0, 0
    updates = dict.fromkeys(MODALITIES, 0)
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        for key in ("trainer", "support_sha256", "manifest_sha256", "modality_budgets"):
            if saved.get(key) != metadata[key]:
                raise ValueError(f"Resume changed {key}")
        previous = {k: v for k, v in saved["config"].items() if k not in _MUTABLE}
        current = {k: v for k, v in config.items() if k not in _MUTABLE}
        if previous != current or CodecConfig(**saved["codec_config"]) != cfg:
            raise ValueError("Resume changed tokenizer/training semantics")
        model.load_state_dict(saved["model"])
        for k, optimizer in optimizers.items():
            optimizer.load_state_dict(saved["optimizers"][k])
        step, epoch, skip = saved["step"], saved["epoch"], saved["next_batch"]
        updates = saved["modality_updates"]
        if updates != {k: min(step, v) for k, v in budgets.items()}:
            raise ValueError("Invalid modality update counters")
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if maximum <= step:
        raise ValueError("Requested run has no remaining updates")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    validation = DataLoader(val, batch_size=size, shuffle=False,
                            generator=torch.Generator().manual_seed(0))
    swan = init_swanlab(config, output)
    try:
        while step < maximum:
            generator = torch.Generator().manual_seed(config.get("seed", 42) + epoch)
            order = torch.randperm(len(train), generator=generator).tolist()
            # Drop the random tail so EVERY update has exactly the requested batch size.
            batches = [order[i:i + size] for i in range(0, len(order) - size + 1, size)]
            loader = DataLoader(train, batch_sampler=batches[skip:], generator=generator)
            for index, batch in enumerate(loader, start=skip):
                batch = move_batch(batch, device)
                stats = {}
                model.eval()
                for key in MODALITIES:
                    module, optimizer = model.models[key], optimizers[key]
                    if updates[key] >= budgets[key]:
                        module.requires_grad_(False)
                        continue
                    module.train()
                    optimizer.zero_grad(set_to_none=True)
                    loss, detail = module.loss(batch[key], batch["valid_frames"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite {key} VQ loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
                    optimizer.step()
                    updates[key] += 1
                    stats[key + "_loss"] = float(loss.detach())
                    stats.update({key + "_" + k: float(v.detach()) for k, v in detail.items()})
                    if cfg.quantizer == "gradient" and config.get("reset_unused_every", 100):
                        if updates[key] % config.get("reset_unused_every", 100) == 0:
                            unused = module.usage == 0
                            module.reset_unused(batch[key], batch["valid_frames"])
                            for field in ("exp_avg", "exp_avg_sq"):
                                if field in optimizer.state[module.codebook.weight]:
                                    optimizer.state[module.codebook.weight][field][unused] = 0
                step += 1
                if step == 1 or step % config.get("log_every", 20) == 0:
                    print(json.dumps(dict(step=step, modality_updates=updates, **stats)), flush=True)
                    log_swanlab(swan, {"train/" + k: v for k, v in stats.items()}, step)
                boundary = any(step == budget for budget in budgets.values())
                if step % config.get("validate_every", 1000) == 0 or step == maximum or boundary:
                    score = validate(model, validation, device, config.get("validation_batches", 16))
                    payload = codec_payload(model, dict(metadata, optimizers={k: v.state_dict() for k, v in optimizers.items()},
                        step=step, epoch=epoch, next_batch=index + 1, modality_updates=dict(updates),
                        codec_config=asdict(cfg), validation_loss=score,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
                    save_checkpoint(output / "last.pt", payload)
                    print(json.dumps(dict(step=step, modality_updates=updates, validation_loss=score,
                                          selection="last.pt")), flush=True)
                    log_swanlab(swan, {"val/diagnostic_loss": score}, step)
                if step >= maximum:
                    break
            epoch, skip = epoch + 1, 0
    finally:
        finish_swanlab(swan)
    return dict(output=str(output), step=step, modality_updates=updates, checkpoint=str(output / "last.pt"))
