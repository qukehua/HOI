"""Restartable, isolated native-parameter training and per-mode model selection."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from ..mesh_geometry import load_mesh_asset
from ..runtime import device_for, move_batch, seed_everything, rng_state, restore_rng, save_checkpoint, write_json
from ..smpl_mesh import DEFAULT_MODEL_PATH
from . import REPRESENTATION
from .body import BodyGeometry, digest_file
from .contact import load_codec
from .data import ParametricDataset
from .diffusion import controls_for, ParametricDiffusion
from .evaluate import reconstruction
from .model import ParametricConfig, ParametricDenoiser, add_geometry_batch
from .evaluation_protocol import frame_plan, window_selection, VERSION


@torch.no_grad()
def prepare_batch(model, batch, geometry, steps):
    for gender, expected in zip(batch["gender"].tolist(), batch["body_model_sha256"]):
        if geometry.layers["male" if gender else "female"].asset_hash != expected:
            raise ValueError("Native data were prepared with different body assets")
    batch = add_geometry_batch(batch, geometry)
    batch["contact"] = model.codec.encode(batch["surface_contact"])
    batch["diffusion_steps"] = steps
    return batch


def load_run(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    if cfg.get("representation") != REPRESENTATION:
        raise ValueError("This entry point needs a native-parameter checkpoint")
    model = ParametricDenoiser(ParametricConfig(**cfg["model"]))
    model.load_state_dict(saved.get("ema", saved["model"]))
    model = model.to(device).eval()
    geometry = BodyGeometry(cfg["dataset"], cfg["contact_support"], cfg.get("body_models", DEFAULT_MODEL_PATH)).to(device)
    expected = saved["geometry_provenance"]
    if geometry.support_hash != expected["support_sha256"] or any(
            layer.asset_hash != expected["models"][name] for name, layer in geometry.layers.items()):
        raise ValueError("Checkpoint requires the exact native body assets and contact support")
    if (saved["codec_metadata"]["dataset"] != cfg["dataset"] or
            saved["codec_metadata"]["support_sha256"] != geometry.support_hash or
            model.config.dataset != cfg["dataset"]):
        raise ValueError("Cross-dataset contact codec")
    return model, ParametricDiffusion(cfg.get("diffusion_steps", 1000)), geometry, saved


def run_training(config, resume=None):
    from ..train import make_loader, update_ema, init_swanlab, log_swanlab, finish_swanlab
    config = deepcopy(config)
    if config.get("representation") != REPRESENTATION or config.get("dataset") not in ("behave", "omomo"):
        raise ValueError("Set representation: parametric_v1 and one dataset")
    dataset_name = config["dataset"]
    for key, default in (("max_steps", 200000), ("validate_every", 1000), ("validation_batches", 4),
                         ("validation_steps", 20), ("save_every", 1000), ("log_every", 20), ("geometry_frames", 16)):
        if config.get(key, default) < 1:
            raise ValueError(f"{key} must be positive")
    if config.get("best_mode", "010") not in ("010", "100"):
        raise ValueError("best_mode must be 010 or 100")
    model_config = ParametricConfig(**(config.get("model", {}) | {"dataset": dataset_name}))
    if config.get("model", {}).get("dataset", dataset_name) != dataset_name:
        raise ValueError("Model/data dataset mismatch")
    config["model"] = model_config.asdict()
    if config.get("holdout_signatures"):
        raise ValueError("Native full-modality training uses allowed_patterns; legacy signature filters are unsupported")
    output = Path(config["output"])
    if not resume and any((output / n).exists() for n in ("last.pt", "best.pt")):
        raise FileExistsError("Choose a fresh native training run or --resume")
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = device_for(config.get("device", "auto"))
    geometry = BodyGeometry(dataset_name, config["contact_support"], config.get("body_models", DEFAULT_MODEL_PATH))
    codec, codec_metadata = load_codec(config["contact_codec"], dataset_name, geometry.support_hash)
    codec_metadata = {k: v for k, v in codec_metadata.items() if k != "model"}
    codec_metadata["file_sha256"] = digest_file(config["contact_codec"])
    train = ParametricDataset(config["manifest"], "train", dataset_name, geometry.support_hash,
                               config.get("window", 120), config.get("stride", 60),
                               config.get("text_condition", True), model_config.text_dim)
    validation = ParametricDataset(config["manifest"], "val", dataset_name, geometry.support_hash,
                                     config.get("window", 120),
                                     text_condition=config.get("validation_text_condition", config.get("text_condition", True)),
                                    text_dim=model_config.text_dim)
    if not len(train) or not len(validation):
        raise ValueError("Native training requires train and validation splits")
    # Spread the fixed validation budget across the split, not only its first clip.
    from torch.utils.data import Subset
    # Missing fields preserve the validation semantics of existing runs on resume.
    val_plan = frame_plan(validation.records, config.get("validation_fps", 0.))
    candidates = [i for i, (ri, start, end) in enumerate(validation.windows)
                  if len(window_selection(val_plan[ri], start, end)[0])]
    if not candidates:
        raise ValueError("No validation frames at the requested evaluation fps")
    val_indices = [candidates[i] for i in np.linspace(0, len(candidates) - 1,
                   min(config.get("validation_batches", 4), len(candidates))).round().astype(int)]
    validation_subset = Subset(validation, val_indices)
    model = ParametricDenoiser(model_config)
    model.codec.load_state_dict(codec.state_dict())
    model.contact_threshold.fill_(codec_metadata["threshold"])
    if not resume:
        model.normalizer.fit(train.normalization_batches(), codec)
    model, geometry = model.to(device), geometry.to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config.get("learning_rate", 1e-4), weight_decay=config.get("weight_decay", .01))
    diffusion = ParametricDiffusion(config.get("diffusion_steps", 1000))
    provenance = {"support_sha256": geometry.support_hash,
                  "models": {name: layer.asset_hash for name, layer in geometry.layers.items()}}
    step, first_epoch, next_batch = 0, 0, 0
    best = {"010": float("inf"), "100": float("inf")}
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        mutable = {"max_steps", "output", "device", "log_every", "save_every", "progress", "use_swanlab",
                   "swanlab_run_name", "swanlab_mode", "swanlab_workspace", "cpu_threads"}
        if ({k: v for k, v in config.items() if k not in mutable}
                != {k: v for k, v in saved["config"].items() if k not in mutable}):
            raise ValueError("Native resume training semantics/config changed")
        if saved["geometry_provenance"] != provenance or saved["codec_metadata"] != codec_metadata:
            raise ValueError("Body/contact assets changed since checkpoint")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        step, first_epoch, next_batch = saved["step"], saved["epoch"], saved["next_batch"]
        best = saved["best"]
        restore_rng(saved["rng"])
        if step >= config.get("max_steps", 200000):
            return str(Path(resume).resolve())
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    info = {"representation": REPRESENTATION, "dataset": dataset_name,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_contact_parameters": sum(p.numel() for p in model.codec.parameters()),
            "body_assets_in_checkpoint": False, "geometry_provenance": provenance,
            "codec_metadata": codec_metadata, "train_windows": len(train), "validation_windows": len(validation),
            "device": str(device), "best_selection": "mode010 OBJ_V2V and mode100 MPJPE separately",
            "validation_window_indices": val_indices,
             "validation_protocol": {"version": VERSION, "fps": config.get("validation_fps", 0.),
                 "text_conditioned": validation.text_condition, "aggregation": "frame_weighted_mean"},
            "omomo_hands": "neutral convention; no annotated hand motion"}
    write_json(output / "run_info.json", info)
    swan = init_swanlab(config, output, info)

    def payload(epoch, cursor):
        return {"config": config, "model": model.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": rng_state(), "step": step,
                "epoch": epoch, "next_batch": cursor, "best": best.copy(),
                "geometry_provenance": provenance, "codec_metadata": codec_metadata}

    try:
        epoch = first_epoch
        while step < config.get("max_steps", 200000):
            for index, raw in enumerate(make_loader(train, config, epoch)):
                if epoch == first_epoch and index < next_batch:
                    continue
                model.train()
                batch = prepare_batch(model, move_batch(raw, device), geometry, diffusion.steps)
                batch["text_features"] = batch["text_features"] * (torch.rand(len(batch["human"]), 1, device=device)
                                                                   >= config.get("text_dropout", .1))
                masks = controls_for(batch, dataset_name, allowed=config.get("allowed_patterns"))
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = diffusion.training_loss(model, batch, masks, geometry, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite native training loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.))
                if not torch.isfinite(grad):
                    raise FloatingPointError("Nonfinite native training gradient")
                optimizer.step()
                update_ema(ema, model, config.get("ema_decay", .999))
                step += 1
                if step == 1 or step % config.get("log_every", 20) == 0:
                    row = {"step": step, **metrics, "gradient_norm": float(grad)}
                    print(json.dumps(row), flush=True)
                    with (output / "train.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    log_swanlab(swan, {"train/" + k: v for k, v in metrics.items()}, step)
                if step % config.get("validate_every", 1000) == 0:
                    original_rng = rng_state()
                    scores = {"010": [], "100": []}
                    for vi, val in enumerate(make_loader(validation_subset, config | {"batch_size": 1}, 0, shuffle=False)):
                        val = prepare_batch(ema, move_batch(val, device), geometry, diffusion.steps)
                        entry = validation.records[int(val["record_index"][0])]
                        vertices, _ = load_mesh_asset(str(Path(entry["object_mesh_path"]).resolve()))
                        ri, start = int(val["record_index"][0]), int(val["window_start"][0])
                        selected, _ = window_selection(val_plan[ri], start, start + int(val["valid_frames"].sum()))
                        for mode in scores:
                            masks = controls_for(val, dataset_name, mode)
                            result = diffusion.sample(ema, val, masks, min(diffusion.steps, config.get("validation_steps", 20)),
                                                      config.get("seed", 42) + 100000 + vi)
                            report = reconstruction(result, val, geometry, mode, vertices * entry["object_mesh_scale"],
                                                    frame_indices=selected)
                            metric = report["metrics"]["OBJ_V2V" if mode == "010" else "MPJPE"]
                            scores[mode].append((metric, report["frames"]))
                    restore_rng(original_rng)
                    improved_modes = []
                    for mode, values in scores.items():
                        score = float(np.average([v for v, _ in values], weights=[n for _, n in values]))
                        log_swanlab(swan, {"val/" + mode: score}, step)
                        with (output / "validation.jsonl").open("a", encoding="utf-8") as f:
                            f.write(json.dumps({"step": step, "mode": mode, "metric": "OBJ_V2V" if mode == "010" else "MPJPE",
                                                "score": score, "windows": len(values)}) + "\n")
                        if score < best[mode]:
                            best[mode] = score
                            improved_modes.append(mode)
                    for mode in improved_modes:
                        save_checkpoint(output / f"best_{mode}.pt", payload(epoch, index + 1))
                        if mode == config.get("best_mode", "010"):
                            save_checkpoint(output / "best.pt", payload(epoch, index + 1))
                    print(json.dumps({"step": step, "best_per_mode": best}), flush=True)
                if step % config.get("save_every", 1000) == 0 or step == config.get("max_steps", 200000):
                    save_checkpoint(output / "last.pt", payload(epoch, index + 1))
                if step >= config.get("max_steps", 200000):
                    break
            epoch += 1
            next_batch = 0
    finally:
        finish_swanlab(swan)
    return str(output / "last.pt")
