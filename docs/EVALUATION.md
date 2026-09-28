# Evaluation scope

`evaluate_batch(states, controls, batch)` returns flat JSON-compatible scalar
metrics. It never restores anchors before measuring predictions. It ignores
padding, including nonfinite values confined to padded frames, and does not
read unobserved reference motion or reference contact labels. Counts accompany
metrics with restricted support; an empty support produces `null`, not success
or zero error.

## Metrics

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

## CLI

```text
python -m unified_hoi.evaluate --prediction output.npz --reference canonical.npz --output metrics.json
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
