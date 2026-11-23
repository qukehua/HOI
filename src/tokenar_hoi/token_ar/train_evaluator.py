"""Train a native motion/text evaluator on train only; save the final checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from ..runtime import move_batch, save_checkpoint, seed_everything, write_json
from .body import digest_file
from .motion_text import EvaluatorConfig, EvaluatorDataset, MotionTextEvaluator, VIEWS, evaluator_payload
from .embedding_metrics import embedding_scores


@torch.no_grad()
def validate(model, dataset, device, batch_size):
    model.eval()
    texts, sources, captions = [], [], []
    motion = {view: [] for view in VIEWS}
    loader = DataLoader(dataset, batch_size=batch_size, generator=torch.Generator().manual_seed(0))
    for batch in loader:
        batch = move_batch(batch, device)
        texts.append(model.encode_text(batch["text_embedding"]).cpu())
        sources.extend(batch["source_id"])
        captions.extend(batch["caption_key"])
        for view in VIEWS:
            motion[view].append(model.encode_motion(batch, view).cpu())
    text = torch.cat(texts).numpy()
    return {view: embedding_scores(torch.cat(rows).numpy(), torch.cat(rows).numpy(), text,
                                   source_ids=sources, captions=captions)["ground_truth"]
            for view, rows in motion.items()}


def run(config, resume=None):
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("cpu_threads", 4))
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Choose a new evaluator output directory or --resume")
    maximum, size = int(config.get("max_steps", 20000)), int(config.get("batch_size", 64))
    if maximum < 1 or size < 2 or config.get("validate_every", 1000) < 1:
        raise ValueError("Evaluator requires positive steps and batch_size >= 2")
    support = digest_file(config["contact_support"])
    kwargs = dict(manifest=config["manifest"], dataset=config["dataset"], support_hash=support,
                  window=config.get("window", 120))
    train = EvaluatorDataset(split="train", stride=config.get("stride", 60), **kwargs)
    val = EvaluatorDataset(split="val", **kwargs)
    print(json.dumps(dict(captioned_train_windows=len(train), captioned_val_windows=len(val),
                          excluded_unlabelled_train_sequences=len(train.excluded_unlabelled),
                          excluded_unlabelled_val_sequences=len(val.excluded_unlabelled))), flush=True)
    if len(train) < size or len(val) < 2:
        raise ValueError("Need a full train batch and at least two validation windows")
    model = MotionTextEvaluator(EvaluatorConfig(dataset=config["dataset"], **config.get("model", {})))
    model.fit_normalization(train)
    device = torch.device(config.get("device", "cuda:0"))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-4),
                                  weight_decay=config.get("weight_decay", .01))
    metadata = dict(config=config, training_split="train", support_sha256=support,
                    manifest_sha256=digest_file(config["manifest"]),
                    training_sources=sorted({r.get("source_sequence_id", r["sequence_id"]) for r in train.records}),
                    training_text_sha256=train.feature_hashes, validation_text_sha256=val.feature_hashes,
                    excluded_unlabelled_train=train.excluded_unlabelled,
                    excluded_unlabelled_val=val.excluded_unlabelled,
                    selection="final_checkpoint", feature_protocol="native_params_clip_text_l2_v1")
    step = 0
    if resume:
        old = torch.load(resume, map_location="cpu", weights_only=True)
        mutable = {"max_steps", "output", "device", "cpu_threads", "log_every", "validate_every"}
        if ({k: v for k, v in old["config"].items() if k not in mutable}
                != {k: v for k, v in config.items() if k not in mutable}):
            raise ValueError("Evaluator resume configuration changed")
        for key in ("manifest_sha256", "support_sha256", "training_text_sha256", "validation_text_sha256"):
            if old[key] != metadata[key]:
                raise ValueError(f"Evaluator resume changed {key}")
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        step = old["step"]
        torch.set_rng_state(old["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(old["cuda_rng"])
    if maximum <= step:
        raise ValueError("No remaining evaluator training steps")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    while step < maximum:
        generator = torch.Generator().manual_seed(config.get("seed", 42) + step)
        indices = torch.randperm(len(train), generator=generator)[:size].tolist()
        batch = next(iter(DataLoader(train, batch_sampler=[indices], generator=generator)))
        batch = move_batch(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluator contrastive loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        step += 1
        if step == 1 or step % config.get("log_every", 20) == 0:
            print(json.dumps(dict(step=step, contrastive_loss=float(loss.detach()))), flush=True)
        if step == maximum or step % config.get("validate_every", 1000) == 0:
            scores = validate(model, val, device, size)
            payload = evaluator_payload(model, dict(metadata, step=step, optimizer=optimizer.state_dict(),
                validation=scores, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
            save_checkpoint(output / "last.pt", payload)
            write_json(output / "validation.json", dict(step=step, ground_truth_retrieval=scores))
            print(json.dumps(dict(step=step, ground_truth_retrieval=scores)), flush=True)
    return dict(output=str(output), step=step, checkpoint=str(output / "last.pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key in ("max_steps", "output"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    print(run(config, args.resume))


if __name__ == "__main__":
    main()
