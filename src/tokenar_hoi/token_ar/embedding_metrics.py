"""Explicit, reproducible native motion/text embedding metrics (sequence-level)."""
import numpy as np


def _features(values):
    values = np.asarray(values, np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Embedding metrics require at least two finite sequences")
    return values


def frechet_distance(real, predicted):
    real, predicted = _features(real), _features(predicted)
    if real.shape[1] != predicted.shape[1]:
        raise ValueError("FID embedding dimensions differ")
    a, b = np.atleast_2d(np.cov(real, rowvar=False)), np.atleast_2d(np.cov(predicted, rowvar=False))
    eigenvalues, vectors = np.linalg.eigh((a + a.T) / 2)
    sqrt_a = (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.T
    middle = sqrt_a @ b @ sqrt_a
    trace_sqrt = np.sqrt(np.maximum(np.linalg.eigvalsh((middle + middle.T) / 2), 0)).sum()
    return float(max(0., np.square(real.mean(0) - predicted.mean(0)).sum() + np.trace(a) + np.trace(b) - 2 * trace_sqrt))


def embedding_scores(real, predicted, text, *, source_ids, captions, seed=42, candidates=32, repeats=20, pairs=300):
    real, predicted, text = map(_features, (real, predicted, text))
    if real.shape != predicted.shape or text.shape != real.shape:
        raise ValueError("Need paired GT/reconstructed/text embeddings of the same shape")
    n = len(real)
    if len(source_ids) != n or len(captions) != n or min(candidates, repeats, pairs) < 1:
        raise ValueError("Invalid retrieval protocol")
    pool = min(candidates, n)
    rng = np.random.default_rng(seed)
    first = rng.integers(n, size=pairs)
    second = (first + rng.integers(1, n, size=pairs)) % n
    result = {}
    # Identical pools for GT and reconstruction, fixed size per query (no short-tail groups).
    pools = np.empty((repeats, n, pool), np.int64)
    for repeat in range(repeats):
        for i in range(n):
            negatives = rng.choice(n - 1, pool - 1, replace=False)
            negatives += negatives >= i
            pools[repeat, i] = np.concatenate(([i], negatives))
    for name, motion in (("ground_truth", real), ("reconstructed", predicted)):
        strict, semantic = [], []
        tie_rng = np.random.default_rng(seed + 1)
        for indices in pools:
            distance = np.linalg.norm(text[:, None] - motion[indices], axis=-1)
            # Random tie breaking avoids favoring the positive at slot 0 for identical features.
            tie_order = np.argsort(tie_rng.random(distance.shape), axis=-1)
            rank = np.take_along_axis(tie_order, np.argsort(np.take_along_axis(distance, tie_order, 1),
                                                         axis=-1, kind="stable"), 1)
            strict.append([float(np.any(rank[:, :min(k, pool)] == 0, axis=1).mean()) for k in (1, 2, 3)])
            matches = np.array([[source_ids[i] == source_ids[j] or captions[i] == captions[j] for j in row]
                                for i, row in enumerate(indices)])
            ranked = np.take_along_axis(matches, rank, 1)
            semantic.append([float(ranked[:, :min(k, pool)].any(1).mean()) for k in (1, 2, 3)])
        result[name] = dict(FID=0. if name == "ground_truth" else frechet_distance(real, predicted),
                            Diversity=float(np.linalg.norm(motion[first] - motion[second], axis=-1).mean()),
                            Matching_Score=float(np.linalg.norm(text - motion, axis=-1).mean()),
                            R_Precision={f"top{k}": float(np.mean(strict, axis=0)[k - 1]) for k in (1, 2, 3)},
                            R_Precision_std={f"top{k}": float(np.std(strict, axis=0)[k - 1]) for k in (1, 2, 3)},
                            semantic_R_Precision={f"top{k}": float(np.mean(semantic, axis=0)[k - 1]) for k in (1, 2, 3)})
    result["protocol"] = dict(sequences=n, effective_candidates=pool, requested_candidates=candidates,
                               repeats=repeats, diversity_pairs=pairs, seed=seed,
                               ranking="text-to-motion; paired index; randomized ties; fixed candidate count per query",
                               random_chance={f"top{k}": min(k, pool) / pool for k in (1, 2, 3)},
                               semantic_positive="same source sequence OR exact normalized caption")
    return result
