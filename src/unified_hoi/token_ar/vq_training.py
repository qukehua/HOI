"""Independent H/O/C update budgets, exact full batches and resumable EMA state."""
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..runtime import move_batch, save_checkpoint, seed_everything, write_json
from . import MODALITIES
from .body import BodyGeometry, digest_file
from .codecs import CodecBundle, CodecConfig, codec_payload
from .data import MotionDataset


def _cpu_module_state(module, optimizer):
    blob = {"model": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
            "optimizer": optimizer.state_dict()}
    for state in blob["optimizer"]["state"].values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.detach().cpu().clone()
    return blob


def _restore_module_state(module, optimizer, blob, device):
    module.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _selection_score(report):
    required = ("human_joint_cm", "object_trans_cm", "object_rot_deg", "contact_f1")
    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError(f"Geometry selection needs {missing}")
    return (report["human_joint_cm"] + report["object_trans_cm"] + report["object_rot_deg"]
            + 100. * (1. - report["contact_f1"]))


def run_independent(config, resume=None):
    from .train_codecs import _MUTABLE, finish_swanlab, init_swanlab, log_swanlab, validate_report
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
    if cfg.canonical_scene:
        # A list is not registered as a submodule, so the body model stays out of the checkpoint.
        model.scene_geometry = [BodyGeometry(config["dataset"], config["contact_support"],
                                             config.get("body_models", "data/smplx_models")).to(device)]
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
        expected = {k: min(step, v) for k, v in budgets.items()}
        if saved.get("contact_frozen"):
            expected["contact"] = updates["contact"]
            if not 0 <= updates["contact"] <= min(step, budgets["contact"]):
                raise ValueError("Invalid frozen contact update counter")
        if updates != expected:
            raise ValueError("Invalid modality update counters")
        contact_frozen = bool(saved.get("contact_frozen", False))
        best_contact = float(saved.get("best_contact_loss", float("inf")))
        contact_stalls = int(saved.get("contact_stalls", 0))
        best_selection = float(saved.get("best_selection", float("inf")))
        contact_snapshot = saved.get("contact_snapshot")
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        contact_frozen, contact_stalls = False, 0
        best_contact, best_selection = float("inf"), float("inf")
        contact_snapshot = None
    contact_patience = int(config.get("contact_patience", 0))
    if contact_patience < 0:
        raise ValueError("contact_patience must be nonnegative")
    select_geometry = bool(config.get("select_by_geometry", False))
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
                view, _ = model.canonical_view(batch)
                geometry = model.scene_geometry[0] if getattr(model, "scene_geometry", None) else None
                stats = {}
                model.eval()
                for key in MODALITIES:
                    module, optimizer = model.models[key], optimizers[key]
                    if (key == "contact" and contact_frozen) or updates[key] >= budgets[key]:
                        module.requires_grad_(False)
                        continue
                    module.train()
                    optimizer.zero_grad(set_to_none=True)
                    loss, detail = module.loss(view[key], view["valid_frames"],
                                               gender=view["gender"] if key == "human" else None,
                                               geometry=geometry if key == "human" else None)
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
                    report = validate_report(model, validation, device, config.get("validation_batches", 16))
                    if contact_patience and not contact_frozen:
                        contact_loss = report["contact_reconstruction"]
                        if contact_loss < best_contact:
                            best_contact = contact_loss
                            contact_stalls = 0
                            contact_snapshot = _cpu_module_state(model.models["contact"], optimizers["contact"])
                        else:
                            contact_stalls += 1
                            if contact_stalls >= contact_patience:
                                if contact_snapshot is None:
                                    raise RuntimeError("Contact early stop has no saved weights to restore")
                                _restore_module_state(model.models["contact"], optimizers["contact"],
                                                      contact_snapshot, device)
                                contact_frozen = True
                                report = validate_report(model, validation, device, config.get("validation_batches", 16))
                    score = report["loss"]
                    chosen = "last.pt"
                    selection_score = None
                    if select_geometry:
                        selection_score = _selection_score(report)
                        if selection_score < best_selection:
                            best_selection = selection_score
                            chosen = "best.pt"
                    payload = codec_payload(model, dict(metadata, optimizers={k: v.state_dict() for k, v in optimizers.items()},
                        step=step, epoch=epoch, next_batch=index + 1, modality_updates=dict(updates),
                        codec_config=asdict(cfg), validation_loss=score,
                        contact_frozen=contact_frozen, best_contact_loss=best_contact,
                        contact_stalls=contact_stalls, best_selection=best_selection,
                        contact_snapshot=contact_snapshot, selection_score=selection_score,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
                    save_checkpoint(output / "last.pt", payload)
                    if chosen == "best.pt":
                        save_checkpoint(output / "best.pt", payload)
                    row = dict(step=step, modality_updates=updates, validation_loss=score, selection=chosen,
                               contact_frozen=contact_frozen, human_joint_cm=report.get("human_joint_cm"),
                               object_trans_cm=report.get("object_trans_cm"), object_rot_deg=report.get("object_rot_deg"),
                               contact_f1=report.get("contact_f1"), contact_reconstruction=report.get("contact_reconstruction"),
                               selection_score=selection_score, best_selection=best_selection)
                    print(json.dumps(row), flush=True)
                    log_swanlab(swan, {"val/" + k: v for k, v in row.items() if isinstance(v, (int, float))}, step)
                if step >= maximum:
                    break
            epoch, skip = epoch + 1, 0
    finally:
        finish_swanlab(swan)
    return dict(output=str(output), step=step, modality_updates=updates, checkpoint=str(output / "last.pt"))
