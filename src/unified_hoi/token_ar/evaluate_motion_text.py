"""Evaluate native-rate VQ reconstructions in a frozen, independently trained embedding space."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..runtime import move_batch, write_json
from . import MODALITIES
from .body import digest_file
from .codecs import canonicalize
from .embedding_metrics import embedding_scores
from .evaluation_protocol import fingerprint
from .motion_text import EvaluatorDataset, VIEWS, load_evaluator


def figure_metrics(scores, view):
    """One row of the MotionGPT table: FID, R-precision Top-1/2/3, Diversity."""
    if view not in VIEWS:
        raise ValueError(f"FID view must be one of {VIEWS}")
    row = scores[view]["reconstructed"]
    return {"view": view, "FID": row["FID"],
            "R_Precision": {k: row["R_Precision"][k] for k in ("top1", "top2", "top3")},
            "Diversity": row["Diversity"],
            "Diversity_real": scores[view]["ground_truth"]["Diversity"]}


def figure_table(scores, view="joint"):
    table = {name: figure_metrics(scores, name) for name in VIEWS}
    chosen = dict(table[view])
    chosen["views"] = table
    return chosen


@torch.no_grad()
def reconstruct_batch(codecs, batch, device):
    batch = move_batch(batch, device)
    n = int(batch["valid_frames"].sum())
    canonical, anchor = codecs.canonical_view(batch)
    tokens = codecs.encode_view(canonical)
    values = {k: codecs.models[k].decode(tokens[k][0], n) for k in MODALITIES}
    if anchor is None:
        values["human"], values["object"] = canonicalize(values["human"], values["object"], codecs.config.dataset)
    else:
        values["human"], values["object"] = codecs.restore(values["human"], values["object"], anchor)
    reconstructed = {k: values[k][None] for k in MODALITIES}
    reconstructed.update(valid_frames=torch.ones(1, n, dtype=torch.bool, device=device), fps=batch["fps"])
    return batch, reconstructed


@torch.no_grad()
def score_motion_pairs(evaluator, pairs, device, *, seed=42, candidates=32, repeats=20, diversity_pairs=300):
    gt, pred = {v: [] for v in VIEWS}, {v: [] for v in VIEWS}
    texts, sources, captions = [], [], []
    for ground, predicted in pairs:
        ground = move_batch(ground, device)
        predicted = move_batch(predicted, device)
        for view in VIEWS:
            gt[view].append(evaluator.encode_motion(ground, view).cpu())
            pred[view].append(evaluator.encode_motion(predicted, view).cpu())
        texts.append(evaluator.encode_text(ground["text_embedding"]).cpu())
        sources.extend(ground["source_id"])
        captions.extend(ground["caption_key"])
    if len(texts) < 2:
        raise ValueError("FID requires at least two windows")
    text = torch.cat(texts).numpy()
    return {view: embedding_scores(torch.cat(gt[view]).numpy(), torch.cat(pred[view]).numpy(), text,
                                   source_ids=sources, captions=captions, seed=seed, candidates=candidates,
                                   repeats=repeats, pairs=diversity_pairs) for view in VIEWS}


@torch.no_grad()
def evaluate_motion_text(codecs, saved, checkpoint, output, *, evaluator, manifest, contact_support,
                         split="val", window=120, limit=None, device="cpu", seed=42,
                         candidates=32, repeats=20, diversity_pairs=300):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    support = digest_file(contact_support)
    if saved.get("support_sha256") != support:
        raise ValueError("VQ checkpoint/contact support mismatch")
    dataset = EvaluatorDataset(manifest, split, codecs.config.dataset, support, window)
    total = len(dataset) if limit is None else min(limit, len(dataset))
    if total < 2:
        raise ValueError("Motion-text evaluation requires at least two windows")
    evaluator_model, evaluator_saved = load_evaluator(evaluator, codecs.config.dataset, support, dataset.records, device)
    pairs, windows = [], []
    loader = DataLoader(dataset, batch_sampler=[[i] for i in range(total)],
                        generator=torch.Generator().manual_seed(0))
    for index, batch in enumerate(loader):
        pairs.append(reconstruct_batch(codecs, batch, device))
        record, start, end = dataset.windows[index]
        windows.append([dataset.records[record]["sequence_id"], start, end])
        if (index + 1) % 10 == 0 or index + 1 == total:
            print(f"Motion-text VQ evaluation {index + 1}/{total}", flush=True)
    scores = score_motion_pairs(evaluator_model, pairs, device, seed=seed, candidates=candidates,
                                repeats=repeats, diversity_pairs=diversity_pairs)
    summary = dict(kind="vq_motion_text_reconstruction", dataset=codecs.config.dataset, split=split,
                   checkpoint=str(checkpoint), checkpoint_sha256=digest_file(checkpoint),
                   checkpoint_step=saved.get("step"), modality_updates=saved.get("modality_updates"),
                   evaluator=str(evaluator), evaluator_sha256=digest_file(evaluator),
                   evaluator_step=evaluator_saved["step"], evaluator_training_split="train",
                   manifest_sha256=digest_file(manifest), support_sha256=support,
                   text_cache_sha256=fingerprint(dataset.feature_hashes),
                   captioned_sequences=len(dataset.records),
                   excluded_unlabelled_sequences=dataset.excluded_unlabelled,
                   excluded_unlabelled_windows=dataset.excluded_windows,
                   window=window, evaluated_windows=total, evaluated_native_frames=sum(w[2] - w[1] for w in windows),
                   evaluation_windows_sha256=fingerprint(windows), limit_windows=limit,
                   protocol="native_motion_text_v1", embedding_metrics=scores,
                   notes=["Frozen evaluator trained independently on native train data, never VQ reconstructions.",
                          "Custom BEHAVE/OMOMO metric space, not numerically comparable to Uni-HOI or HumanML3D.",
                          "Full native-rate windows, not the 1fps geometry evaluation frames.",
                          "Only captioned sequences are included; unlabelled sequence/window exclusions are explicit.",
                          "human/object/contact use their own motion encoders; joint uses all three reconstructed streams.",
                          "FID is sequence embedding distribution distance; Diversity should match GT, not be maximized.",
                          "R-Precision is paired text-to-motion top-k; semantic variant handles repeated captions/sources.",
                          "Check GT retrieval against random chance before trusting reconstruction scores.",
                          "Matching Score uses mean paired Euclidean distance of unit-normalized embeddings (lower is better)."])
    write_json(output / "summary.json", summary)
    return summary
