"""One masked multi-task AR model; full/partial backbone tuning without per-task LoRA."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader
import yaml

from ..runtime import move_batch, write_json
from . import ARCHITECTURE, MODES
from .body import digest_file
from .checkpoint import read_metadata
from .codecs import load_codecs
from .data import MotionDataset
from .model import build_model
from .sequence import SequenceBuilder, build_training_batch


@torch.no_grad()
def validate(model, codecs, builder, dataset, config, accelerator):
    model.eval()
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 1), shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(config.get("seed", 42)))
    # Identical fixed validation windows on all ranks keep FSDP forward collectives aligned.
    sums = {mode: [0., 0] for mode in MODES}
    for index, batch in enumerate(loader):
        if index >= config.get("validation_batches", 8):
            break
        batch = move_batch(batch, accelerator.device)
        if not config.get("validation_text_condition", True):
            batch["text"] = [""] * len(batch["text"])
        for mode in MODES:
            packed = build_training_batch(batch, codecs, builder, [mode] * len(batch["human"]))
            with accelerator.autocast():
                result = model(**packed, object_bps=batch["object_bps"])
            sums[mode][0] += float(result["loss_sum"])
            sums[mode][1] += int(result["token_count"])
    if any(count == 0 for _, count in sums.values()):
        raise ValueError("Empty validation set")
    scores = {mode: loss / count for mode, (loss, count) in sums.items()}
    return sum(scores.values()) / len(scores), scores


def run(config, resume=None):
    from accelerate import Accelerator
    from accelerate.utils import GradientAccumulationPlugin, set_seed
    if config.get("architecture") != ARCHITECTURE:
        raise ValueError("Use a token_ar training configuration")
    modes = config.get("allowed_modes", list(MODES))
    if not modes or any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError("Invalid H/O/C masks")
    if min(config["max_steps"], config.get("batch_size", 1), config.get("validation_batches", 8)) < 1:
        raise ValueError("Training and validation budgets must be positive")
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Choose a new AR output directory or explicitly resume")
    # FSDP no_sync retains full, unsharded gradients across microbatches. Synchronizing
    # each microbatch keeps them sharded while still accumulating before optimizer.step.
    accumulation = GradientAccumulationPlugin(num_steps=config.get("gradient_accumulation_steps", 8),
                                               sync_each_batch=config.get("sync_each_batch", False))
    accelerator = Accelerator(gradient_accumulation_plugin=accumulation,
                              mixed_precision=config.get("mixed_precision", "bf16"))
    torch.set_num_threads(config.get("cpu_threads", 4))
    set_seed(config.get("seed", 42))
    support_hash = digest_file(config["contact_support"])
    codecs, codec_saved = load_codecs(config["codecs"], config["dataset"], support_hash, accelerator.device)
    manifest_hash = digest_file(config["manifest"])
    if codec_saved["manifest_sha256"] != manifest_hash:
        raise ValueError("VQ training manifest differs from AR training manifest; verify splits explicitly")
    kwargs = dict(manifest=config["manifest"], dataset=config["dataset"], support_hash=support_hash,
                  window=config.get("window", 120))
    train = MotionDataset(split="train", stride=config.get("stride", 60), **kwargs)
    val = MotionDataset(split="val", **kwargs)
    if not len(train) or not len(val):
        raise ValueError("Need nonempty train and validation splits")
    old = read_metadata(resume) if resume else None
    if old:
        from transformers import AutoTokenizer
        model, _ = build_model(config, codecs.config, old["backbone_config"])
        tokenizer = AutoTokenizer.from_pretrained(Path(resume) / "tokenizer", local_files_only=True)
    else:
        model, tokenizer = build_model(config, codecs.config)
    model.configure_training(config.get("train_scope", "full"), config.get("last_layers", 4),
                             config.get("gradient_checkpointing", True))
    builder = SequenceBuilder(tokenizer, model.vocab, config.get("max_text_tokens", 128),
                              config.get("max_sequence_tokens", 1024))
    metadata = {"architecture": ARCHITECTURE, "config": config, "codec_config": asdict(codecs.config),
                "codec_sha256": digest_file(config["codecs"]), "support_sha256": support_hash,
                "manifest_sha256": manifest_hash, "backbone_config": model.backbone.config.to_dict(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "selection_metric": "mean_of_seven_teacher_forced_validation_token_cross_entropies",
                "num_processes": accelerator.num_processes}
    if old:
        for key in ("codec_sha256", "manifest_sha256", "support_sha256", "num_processes"):
            if old[key] != metadata[key]:
                raise ValueError(f"Resume changed {key}")
        allowed_changes = {"max_steps", "output"}
        if ({k: v for k, v in old["config"].items() if k not in allowed_changes}
                != {k: v for k, v in config.items() if k not in allowed_changes}):
            raise ValueError("Resume changed AR training semantics")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=config.get("learning_rate", 2e-5), weight_decay=config.get("weight_decay", .01))
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    loader = DataLoader(train, batch_size=config.get("batch_size", 1), shuffle=True,
                        num_workers=0, generator=generator)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if old:
        accelerator.load_state(str(resume))
        step, epoch, skip, best = old["step"], old["epoch"], old["next_batch"], old["best_val"]
    else:
        step, epoch, skip, best = 0, 0, 0, float("inf")
        set_seed(config.get("seed", 42), device_specific=True)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "config.json", config)
        write_json(output / "run_info.json", metadata)
    accelerator.wait_for_everyone()
    optimizer.zero_grad(set_to_none=True)
    while step < config["max_steps"]:
        generator.manual_seed(config.get("seed", 42) + epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        active = accelerator.skip_first_batches(loader, skip) if skip else loader
        for index, batch in enumerate(active, start=skip):
            model.train()
            with accelerator.accumulate(model):
                batch = move_batch(batch, accelerator.device)
                selected = [modes[int(torch.randint(len(modes), ()))] for _ in batch["human"]]
                packed = build_training_batch(batch, codecs, builder, selected, config.get("text_dropout", .1))
                with accelerator.autocast():
                    result = model(**packed, object_bps=batch["object_bps"])
                global_count = accelerator.reduce(result["token_count"].detach(), reduction="sum")
                loss = result["loss_sum"] * accelerator.num_processes / global_count
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite AR loss")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            step += 1
            if step % config.get("log_every", 20) == 0 or step == 1:
                logged_loss = accelerator.reduce(result["loss_sum"].detach(), reduction="sum") / global_count
                accelerator.print(json.dumps({"step": step, "token_ce": float(logged_loss)}), flush=True)
            if step % config.get("validate_every", 1000) == 0 or step == config["max_steps"]:
                score, details = validate(model, codecs, builder, val, config, accelerator)
                improved = score < best
                best = min(best, score)
                destination = output / f"checkpoint-{step:07d}"
                if destination.exists():
                    raise FileExistsError(destination)
                # All ranks must finish the existence check before any rank creates the directory.
                accelerator.wait_for_everyone()
                accelerator.save_state(str(destination), safe_serialization=True)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(destination / "tokenizer")
                    shutil.copyfile(config["codecs"], destination / "codecs.pt")
                    write_json(destination / "metadata.json", {**metadata, "step": step, "epoch": epoch,
                                                               "next_batch": index + 1, "best_val": best,
                                                               "validation": details})
                    (destination / "COMPLETE").write_text(ARCHITECTURE, encoding="utf-8")
                    write_json(output / "last.json", {"checkpoint": destination.name, "step": step})
                    if improved:
                        write_json(output / "best.json", {"checkpoint": destination.name, "step": step, "score": score})
                accelerator.wait_for_everyone()
                accelerator.print(json.dumps({"step": step, "val_token_ce": score, "modes": details}), flush=True)
            if step >= config["max_steps"]:
                break
        epoch, skip = epoch + 1, 0
    accelerator.end_training()
    return {"output": str(output), "step": step}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Complete token_ar checkpoint directory")
    parser.add_argument("--backbone")
    parser.add_argument("--output")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key in ("backbone", "output", "max_steps"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    print(run(config, args.resume))


if __name__ == "__main__":
    main()
