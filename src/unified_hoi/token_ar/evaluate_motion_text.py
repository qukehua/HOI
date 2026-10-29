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
    gt, pred = {v: [] for v in VIEWS}, {v: [] for v in VIEWS}
    texts, sources, captions, windows = [], [], [], []
    loader = DataLoader(dataset, batch_sampler=[[i] for i in range(total)],
                        generator=torch.Generator().manual_seed(0))
    for index, batch in enumerate(loader):
        batch = move_batch(batch, device)
        n = int(batch["valid_frames"].sum())
        tokens = codecs.encode(batch)
        values = {k: codecs.models[k].decode(tokens[k][0], n) for k in MODALITIES}
        values["human"], values["object"] = canonicalize(values["human"], values["object"], codecs.config.dataset)
        reconstructed = {k: v[None] for k, v in values.items()}
        reconstructed.update(valid_frames=torch.ones(1, n, dtype=torch.bool, device=device), fps=batch["fps"])
        for view in VIEWS:
            gt[view].append(evaluator_model.encode_motion(batch, view).cpu())
            pred[view].append(evaluator_model.encode_motion(reconstructed, view).cpu())
        texts.append(evaluator_model.encode_text(batch["text_embedding"]).cpu())
        sources.extend(batch["source_id"])
        captions.extend(batch["caption_key"])
        record, start, end = dataset.windows[index]
        windows.append([dataset.records[record]["sequence_id"], start, end])
        if (index + 1) % 10 == 0 or index + 1 == total:
            print(f"Motion-text VQ evaluation {index + 1}/{total}", flush=True)
    text = torch.cat(texts).numpy()
    scores = {v: embedding_scores(torch.cat(gt[v]).numpy(), torch.cat(pred[v]).numpy(), text,
                                  source_ids=sources, captions=captions, seed=seed,
                                  candidates=candidates, repeats=repeats, pairs=diversity_pairs) for v in VIEWS}
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
