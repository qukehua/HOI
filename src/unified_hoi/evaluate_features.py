"""Dataset-level Uni-HOI table-1 metrics from externally encoded HOI/text features.

No fallback to raw joint coordinates, model hidden states, or text-only CLIP.
The evaluator checkpoint and its exact preprocessing must be supplied externally;
the paper does not identify enough details to certify numerical comparability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .runtime import write_json


def _features(x, name):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite nonempty [N,D] features")
    return x


def frechet_distance(generated, reference):
    """Gaussian Frechet distance with unbiased covariance and a PSD matrix root."""
    generated, reference = _features(generated, "generated"), _features(reference, "reference")
    if generated.shape[1] != reference.shape[1] or min(len(generated), len(reference)) < 2:
        raise ValueError("FID requires >=2 samples per distribution with matching dimensions")
    diff = generated.mean(0) - reference.mean(0)
    cg, cr = np.atleast_2d(np.cov(generated, rowvar=False)), np.atleast_2d(np.cov(reference, rowvar=False))
    eig, vec = np.linalg.eigh((cg + cg.T) / 2)
    root = (vec * np.sqrt(np.maximum(eig, 0))) @ vec.T
    middle = root @ cr @ root
    trace_root = np.sqrt(np.maximum(np.linalg.eigvalsh((middle + middle.T) / 2), 0)).sum()
    result = float(diff @ diff + np.trace(cg) + np.trace(cr) - 2 * trace_root)
    return max(0., result)


def retrieval_precision(motion, text, *, batch_size, seed):
    """Text-to-HOI top-k retrieval against equal-sized, shuffled candidate batches.

    The final incomplete group is excluded (and counted), so top-3 is never
    inflated by a smaller candidate pool. Ties use a conservative worst rank.
    Feature normalization is owned by the trained evaluator, not guessed here.
    """
    motion, text = _features(motion, "motion"), _features(text, "text")
    if motion.shape != text.shape or batch_size < 3 or len(motion) < batch_size:
        raise ValueError("R-precision needs paired features and >= one candidate batch of size >=3")
    order = np.random.default_rng(seed).permutation(len(motion))
    usable = len(order) // batch_size * batch_size
    hits = np.zeros(3, dtype=np.int64)
    for ids in order[:usable].reshape(-1, batch_size):
        distance = cdist(text[ids], motion[ids], metric="euclidean")
        rank = (distance <= np.diag(distance)[:, None]).sum(-1)
        hits += np.array([(rank <= k).sum() for k in (1, 2, 3)])
    return {f"R_precision_top{k}": float(hits[k - 1] / usable) for k in (1, 2, 3)}, {
        "retrieval_candidate_batch_size": batch_size, "retrieval_evaluated_samples": usable,
        "retrieval_dropped_tail_samples": len(order) - usable,
    }


def diversity(features, *, pairs, seed):
    """Mean Euclidean distance between two independently sampled index sets.

    Each set is sampled without replacement; an index can occur in both sets,
    as in the usual T2M evaluator. The requested sample budget is not silently
    reduced for a small dataset.
    """
    features = _features(features, "features")
    if pairs < 1 or len(features) < pairs:
        raise ValueError("Diversity requires at least the requested number of pairs")
    rng = np.random.default_rng(seed)
    a, b = rng.choice(len(features), pairs, replace=False), rng.choice(len(features), pairs, replace=False)
    return float(np.linalg.norm(features[a] - features[b], axis=-1).mean())


def evaluate_features(generated, reference, text, *, retrieval_batch_size, diversity_pairs, seed=42):
    generated, reference, text = (_features(x, name) for x, name in
                                   ((generated, "generated"), (reference, "reference"), (text, "text")))
    if generated.shape != reference.shape or generated.shape != text.shape:
        raise ValueError("generated/reference/text embeddings must be aligned [N,D] in the same trained space")
    retrieval, details = retrieval_precision(generated, text, batch_size=retrieval_batch_size, seed=seed)
    real_retrieval, _ = retrieval_precision(reference, text, batch_size=retrieval_batch_size, seed=seed)
    div = diversity(generated, pairs=diversity_pairs, seed=seed)
    real_div = diversity(reference, pairs=diversity_pairs, seed=seed)
    return {"metrics": {"FID": frechet_distance(generated, reference), **retrieval, "Diversity": div},
            "ground_truth": {**real_retrieval, "Diversity": real_div},
            "details": {**details, "samples": len(generated), "feature_dim": generated.shape[1],
                        "diversity_pairs": diversity_pairs, "seed": seed,
                        "diversity_absolute_gap_to_gt": abs(div - real_div),
                        "Diversity_direction": "closer_to_ground_truth_not_unbounded_higher",
                        "feature_normalization": "none_added_by_metric_code"}}


def evaluate_archive(path, *, retrieval_batch_size, diversity_pairs, seed=42, benchmark=None):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        required = {"generated", "reference", "text", "sample_ids", "dataset", "provenance_json"}
        if not required <= set(z.files):
            raise ValueError(f"feature archive missing {sorted(required - set(z.files))}")
        data = {k: z[k] for k in required}
    provenance = json.loads(str(data["provenance_json"].item()))
    for key in ("evaluator_name", "checkpoint_sha256", "preprocessing", "source", "generated_samples_source"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError(f"feature provenance must specify {key}")
    digest = provenance["checkpoint_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError("checkpoint_sha256 must identify the real trained evaluator checkpoint")
    n = len(data["generated"])
    ids, datasets = data["sample_ids"].astype(str), data["dataset"].astype(str)
    if ids.shape != (n,) or datasets.shape != (n,) or len(set(ids)) != n:
        raise ValueError("sample IDs must be unique and datasets must align with all feature rows")
    if set(datasets) - {"behave", "omomo"}:
        raise ValueError("this protocol expects BEHAVE/OMOMO features")
    if any(len(data[k]) != n for k in ("reference", "text")):
        raise ValueError("feature row counts disagree")
    if benchmark is not None:
        index = Path(benchmark) / "feature_inputs.jsonl"
        actual_hash = hashlib.sha256(index.read_bytes()).hexdigest()
        if provenance.get("feature_inputs_sha256") != actual_hash:
            raise ValueError("features must identify the exact benchmark feature_inputs.jsonl SHA256")
        rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = {r["sample_id"]: r["dataset"] for r in rows}
        if len(expected) != len(rows) or dict(zip(ids, datasets)) != expected:
            raise ValueError("feature IDs/datasets do not cover the benchmark text-to-HOI outputs exactly")
    results = {}
    for dataset in sorted(set(datasets)):
        mask = datasets == dataset
        results[dataset] = evaluate_features(data["generated"][mask], data["reference"][mask], data["text"][mask],
                                             retrieval_batch_size=retrieval_batch_size,
                                             diversity_pairs=diversity_pairs, seed=seed)
    return {"profile": "uni-hoi-table1", "metrics_by_dataset": results, "evaluator_provenance": provenance,
            "feature_archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "benchmark_identity_verified": benchmark is not None,
            "directly_comparable_to_paper_table": False,
            "limitation": "Uni-HOI evaluator weights, input preprocessing, retrieval batch size and repeated-trial protocol have not been verified. Matching formulas alone is insufficient."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, help="Validate IDs and source hash against generated text-to-HOI samples")
    parser.add_argument("--retrieval-batch-size", type=int, required=True,
                        help="Use the reference evaluator's setting; Uni-HOI does not state it")
    parser.add_argument("--diversity-pairs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Choose a new feature evaluation output")
    result = evaluate_archive(args.features, retrieval_batch_size=args.retrieval_batch_size,
                              diversity_pairs=args.diversity_pairs, seed=args.seed, benchmark=args.benchmark)
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
