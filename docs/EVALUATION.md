# Uni-HOI evaluation protocol

The default CLI/benchmark profile is now `uni-hoi`, following the user's PDF
**Uni-HOI, arXiv:2604.27491v2 (11 September 2026)**, section 4.1 and tables 1-3.
Metrics depend on the **task**, not only on the dataset.

| Paper task | Dataset in the paper | Project mode | Primary metrics |
| --- | --- | --- | --- |
| Object-to-human, optionally with text | FullBodyManipulation / OMOMO, table 2 | `101` | HandJPE (cm), MPJPE (cm), C_prec, C_rec, C_acc, c% |
| Human-to-object | BEHAVE / GRAB, table 3 | `011` | E_ch (m), E_v2v (m); see naming ambiguity below |
| Text-to-HOI | BEHAVE and OMOMO, table 1 | `111` plus actual text | FID, R-precision Top-1/2/3, Diversity |

Sources:
- [Uni-HOI section 4.1 and tables](https://arxiv.org/html/2604.27491v2).
- [OMOMO metric implementation](https://github.com/lijiaman/omomo_release/blob/e9d8c52f41866ed0c07ac574cb8f7f9424b6d1e7/evaluation_metrics.py#L278).
- [OMOMO SMPL-H 24-joint decoding and sample selection](https://github.com/lijiaman/omomo_release/blob/e9d8c52f41866ed0c07ac574cb8f7f9424b6d1e7/trainer_full_body_manip_diffusion.py).
- [Object Pop-up evaluator](https://github.com/ptrvilya/object-popup/blob/b4eeedc449466d665ab192c0e7a15fd2a4393ca6/popup/core/evaluator.py).

## What is implemented, and what is comparable

`paper_metrics.py` implements the geometry formulas and adapters.
`evaluate_features.py` implements the dataset-level embedding metrics.
The benchmark separates datasets, generation directions and text settings.
The existing control diagnostics remain supplementary; anchor error is not
renamed MPJPE, and predicted contact bits are not treated as detected contacts.

**Matching metric definitions is not a complete reproduction of Uni-HOI's
tables.** Its exact evaluator checkpoint/preprocessing, test sequence IDs,
fps, windowing, repetitions, retrieval batch size and diversity sampling budget
have not been verified. Our data still uses 10fps windows, filtered rigid OMOMO
objects and existing test splits. OMOMO source fps remains an explicit
unverified 30fps assumption. Reports therefore set
`directly_comparable_to_paper_table: false`. Do not copy our scores directly
into a claimed reproduction of its tables.

No model was trained for this metric change. Tests verify computations and
pipeline behavior, not model quality or physical execution.

## OMOMO: object-to-human

- Decode 24 evaluation joints from the predicted pelvis position and global
  rotations using the **same subject's raw rest offsets**. The first 22 are body
  joints; the final two are middle-finger bases (SMPL-H joints 28 and 43), attached
  to wrists 20 and 21. This follows OMOMO's SMPL-decoded evaluation route, not the
  network's independently predicted xyz channels. Their inconsistency remains
  visible in the supplementary FK diagnostic.
- **HandJPE_cm:** mean Euclidean position error of the two middle-finger bases in
  world coordinates, multiplied by 100. No root alignment.
- **MPJPE_cm:** subtract each skeleton's own pelvis at each frame; average
  Euclidean error across all 24 joints, including the root, then multiply by 100.
  This is translation alignment only, not PA-MPJPE/Procrustes.
- **Detected contact:** at least one of the two hand joints is at distance
  **strictly less than 0.05m** from an original object mesh vertex. Use the full
  original vertex set, not the 1024 area-sampled training points. Ground truth
  contacts are recalculated from reference joints. Prediction contacts are
  recalculated from generated joints. The diffusion contact head is not used.
- Contact precision/recall/accuracy are calculated from a **per-frame OR over
  the two hands**, not 22 per-joint bits. Following OMOMO, an empty precision or
  recall denominator produces 0; TP/FP/TN/FN are saved to make this explicit.
- **c_percent:** fraction of frames with detected predicted contact, in [0,1],
  matching Uni-HOI table 2 (e.g. 0.61). Multiply by 100 only for a displayed
  percentage. This is not contact accuracy.
- Reference object geometry uses original **per-frame OMOMO scales**; the small
  fixed-scale approximation in training is not silently applied to the evaluator.
  Supplied condition values and generated object poses must match the reference.

Only evaluation reads the raw 24-joint offsets and original scales. They are
not new inputs to the denoiser, and training archives/checkpoints are unchanged.
Original meshes and the trusted test joblib must also be copied to Linux:

```text
OMOMO/data/test_diffusion_manip_seq_joints24.p
OMOMO/data/captured_objects/*_cleaned_simplified.obj
```

Paths can be overridden with `--omomo-raw`, `--omomo-objects` and
`--omomo-source-fps`. Raw/canonical timestamps, skeleton offsets and object
transforms are cross-checked; mismatches fail instead of guessing hand locations.

OMOMO's released quant-evaluation selects one of 20 samples by minimum MPJPE.
Our default is **one sample**. `--samples-per-window 20 --modes 101` supports that
selection, using the **same selected sample for every metric**. Uni-HOI does not
state that it uses this exact sampling budget. Report K; never compare best-of-20
against single-sample results without disclosure.

## BEHAVE: human-to-object

Use the official template mesh with the same vertex-mean centering as our
BEHAVE annotation conversion. Paths default to `data/raw/behave/objects`;
override with `--behave-objects`.

- **E_v2v_m:** transform all corresponding template vertices using predicted and
  reference poses; average their Euclidean distances over vertices and frames.
  It is not translation error, a squared loss, or a nearest-neighbor metric.
- **E_ch_m:** independently sample 10,000 area-uniform points on each predicted
  and reference mesh **per frame**; sum the two mean nearest-neighbor Euclidean
  distances. Distances are unsquared and the two directions are summed, not
  averaged. This follows the cited Object Pop-up implementation. A fixed metric
  seed makes the random sampling reproducible. Smaller `--chamfer-samples` is
  allowed for diagnostics but changes the estimator.
- Independent sampling means even GT-vs-GT sampled Chamfer has a small nonzero
  floor. Corresponding-vertex V2V is zero for identical poses.
- No alignment of prediction to reference is applied. Rotation errors remain
  visible even if translations match.
- **Naming ambiguity:** Uni-HOI section 4.1 defines E_ch as bidirectional Chamfer,
  while table 3 labels its column E_c. We do not assume E_c means center error.
  Primary output is the explicitly defined `E_ch_m`; a separate
  `details.object_centroid_error_m` computes the 3D vertex-centroid displacement.
  The cited Object Pop-up source also has a scalar `vertices.mean()` center
  calculation; that implementation issue is not copied into the 3D diagnostic.

OMOMO human-to-object can also be evaluated explicitly with `--modes 011`,
but it is an extension to the paper's table 3 setting. BEHAVE object-to-human
does not have a verified 24-joint adapter and is rejected by this profile.

## Text-to-HOI: dataset-level feature metrics

The formulas are implemented; **the matching trained HOI/text evaluator is
missing**. No public Uni-HOI evaluator/checkpoint was located in the paper or
the source search for this change. Motion-generation weights, the model's hidden
states, raw skeleton coordinates, or a text-only CLIP encoder are not replacements.

Generate with `--modes 111 --text-conditioned`. Every evaluated window must have
a real cached text condition; otherwise the run fails. The benchmark writes
predictions and `feature_inputs.jsonl`. Its per-window FID/R-precision/Diversity
are `null`, with the missing dependency stated. FID is never computed per clip.

An external, trained and frozen HOI/text evaluator must encode the generated
motion, aligned reference motion and text into its shared evaluation space.
Save a non-pickled NPZ with:

| Key | Shape/type | Contract |
| --- | --- | --- |
| generated, reference, text | each [N,D] | Aligned embeddings from the same trained evaluation space |
| sample_ids | [N] strings | Unique; match benchmark feature_inputs exactly |
| dataset | [N] strings | `behave` or `omomo`; metrics computed separately |
| provenance_json | scalar JSON string | Evaluator provenance listed below |

Required provenance fields: `evaluator_name`, `checkpoint_sha256`,
`preprocessing`, `source`, `generated_samples_source`; when binding to a
benchmark also provide `feature_inputs_sha256` (SHA256 of its exact index file).
Provenance is recorded, not accepted as proof of paper equivalence.

```bash
python -m unified_hoi.evaluate_features \
  --features outputs/hoi_evaluator_features.npz \
  --benchmark runs/eval_text_hoi \
  --retrieval-batch-size 32 --diversity-pairs 300 --seed 42 \
  --output runs/eval_text_hoi/feature_metrics.json
```

Here 32/300 are **example settings, not stated Uni-HOI settings**; use the
reference evaluator's actual protocol. The CLI requires them explicitly.

- FID uses sample means, unbiased covariances and a numerically stable PSD
  matrix square root. At least two samples are required.
- R-precision performs text-to-HOI retrieval with Euclidean distance in equal
  candidate batches. A seeded permutation fixes candidate groups; incomplete
  final groups are dropped and counted. Ties receive the worst tied rank.
  Feature normalization is not guessed.
- Diversity is mean feature distance between two independently sampled sets
  of indices, each without replacement. The requested pair budget cannot be
  silently reduced. Report real-data Diversity and the gap: **closer to real
  is better**, not arbitrarily larger.
- A `--benchmark` check binds sample IDs/datasets and the exact index hash to
  a generation run. It does not certify the semantics of a third-party encoder.
- Run independent seeds/repetitions with new output paths for uncertainty;
  no fabricated confidence intervals are emitted.

## Entry points and aggregation

```bash
# Evaluate each dataset with the checkpoint trained on that dataset.
python scripts/benchmark.py \
  --checkpoint runs/omomo_seed42/last.pt \
  --manifest data/processed/omomo_combined.jsonl \
  --modes 101 --output runs/eval_omomo --save-predictions

python scripts/benchmark.py \
  --checkpoint runs/behave_seed42/last.pt \
  --manifest data/processed/behave_with_text.jsonl \
  --modes 011 --output runs/eval_behave --save-predictions

# Existing mixed-control experiments remain available.
python scripts/benchmark.py \
  --profile diagnostics --modes mixed human_object \
  --checkpoint runs/omomo_seed42/last.pt \
  --manifest data/processed/omomo_combined.jsonl \
  --output runs/eval_controls
```

Primary `summary.json.metrics` groups by dataset/task/text setting. It provides
macro means over evaluated windows and frame-weighted means, plus the supported
window count. Contact frame-weighted means are weighted averages of per-window
ratios, **not pooled precision/recall**; raw confusion counts remain in
`per_clip.jsonl`. The paper does not fully specify the aggregation protocol.
Do not mix the two aggregation choices in one comparison.

The default single-archive CLI adds `paper_evaluation` to the existing flat
diagnostics. It infers a task only from exact saved control masks; mixed
constraints do not become a paper task. `--profile diagnostics` runs the old
diagnostic evaluator only. Ground truth read for task scoring is not fed into
generation. Text provenance is saved by current sampling; older archives without
it are not silently labeled text-conditioned.

## Supplementary control diagnostics

`evaluate_batch(states, controls, batch)` returns flat JSON-compatible scalar
metrics. It never restores anchors before measuring predictions. It ignores
padding, including nonfinite values confined to padded frames, and does not
read unobserved reference motion or reference contact labels. Counts accompany
metrics with restricted support; an empty support produces `null`, not success
or zero error.

### Diagnostic metrics

- Raw observed-feature MAE is reported by modality and in aggregate. The
  aggregate mixes metre-valued positions, rotation encodings and contact values;
  it is an implementation diagnostic, not a physically meaningful distance.
- Position error in metres uses fully observed xyz triples. Component MAE also
  covers partly observed xyz, such as ground-plane paths. Angular error in
  degrees uses fully observed rotation-6D entries. Degenerate prediction or
  target rotation encodings receive a 180-degree penalty rather than inheriting
  the geometry helper's fallback rotation; prediction degeneracy rates are also
  reported. Partial rotation features are evaluated only by raw feature MAE.
- Contact geometry is evaluated **only** at explicitly observed contact bits.
  Positive distance is the distance to the nearest sampled object-surface point.
  Positive violation means distance >= threshold; negative violation means
  distance < threshold. Default threshold is 0.05 metres, overridable with
  `batch['contact_threshold_m']`. Point sampling density affects these metrics.
- `contact_anchor_satisfaction_rate` pools explicit positive and negative
  constraints. `contact_all_anchors_satisfied_rate` requires every explicit
  contact constraint within an example to succeed, averaging only examples that
  have constraints. `contact_positive_and_negative_all_satisfied_rate` further
  restricts to examples containing both signs. Predicted contact-value accuracy
  is reported separately and does not determine geometric evaluation support.
- Slip in m/s is joint velocity in object-local coordinates, at adjacent valid
  frames where both contact bits are explicitly positive. Moving rigidly with
  the object has zero slip. Unknown contact frames and invalid frames are not
  bridged. A failed positive contact can still have low slip, so read slip
  alongside positive distance and violation rate.
- FK error compares predicted nonroot joint xyz with FK from predicted global
  rotations, canonical rest offsets and predicted root xyz. The root is excluded
  because its error is zero by construction. The default is the 22-joint SMPL
  parent tree; an alternative topologically ordered `batch['parents']` is allowed.
- Floor metrics report the fraction, mean depth and maximum depth below a Y-up
  plane, separately for human joint centres and sampled object points. Default
  plane is Y=0, overridable with `batch['floor_y_m']`. They do not measure full
  human-mesh penetration or object-object collisions.
- Speed, acceleration and jerk means describe joint and object-translation
  trajectories. Derivatives use actual timestamps and midpoint time intervals.
  FPS is used only when explicitly supplied and timestamps are absent. Without
  timing information these metrics and slip are null. Fewer valid temporal
  samples also produce null. **Smooth motion alone is not evidence of realism.**

All results are labelled `kinematic_proxies_only_not_physical_validation`.
There is no force, balance, actuator, grasp-stability or task-completion test.
Hard inpainting can make anchor errors zero by construction, so those numbers
alone cannot establish learned controllability or interaction quality. This
module does not calculate distributional realism or model composition scores;
those require dataset-level experiments and held-out manifests.

### Diagnostic CLI

```text
python -m unified_hoi.evaluate --profile diagnostics --prediction output.npz --reference canonical.npz --output metrics.json
```

The prediction contains one unbatched `human`, `object`, `contact` sequence plus
explicit `fps` and `timestamps`. The canonical reference supplies object points,
rest offsets, and values for controls explicitly marked `from_reference: true`.
By default the CLI reads the actual `human_observed`/`human_mask` (and object and
contact equivalents) saved during sampling. It never redraws a random mask.
If none are present, the evaluation uses empty controls; if only some are present,
it raises an error. Supply `--controls spec.json` to explicitly override saved
controls, with `from_reference: true` required to read reference anchor values.
The CLI reads a reference segment matching the prediction length; use
`--reference-start N` for a later segment. Control frame indices are relative
to that segment. Prediction and reference frame intervals must agree; a constant
timestamp-origin offset is allowed, resampling or time warping is not guessed.
The CLI emits strict JSON to stdout and optionally to `--output`.
