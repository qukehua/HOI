# Control interface and evaluation protocol

All public positions use world coordinates, metres and Y-up. Human and object
rotation features are the first two **rows** of a rotation matrix. Controls are
hard feature anchors; preserving them does not imply valid forward kinematics,
constant bone lengths, plausible forces or an executable interaction.

## Tensor contract

| Field | Shape | Meaning |
| --- | --- | --- |
| `human` | `[B,T,J,9]` | world xyz then global rotation 6D; default J=22 |
| `object` | `[B,T,9]` | world xyz then rotation 6D of a known rigid object |
| `contact` | `[B,T,J]` | binary human-joint/object contact |
| `valid_frames` | `[B,T]` | bool, True for actual sequence frames |

`ControlBatch(values, masks, valid_frames, signatures=None)` has identical value
and mask shapes for each modality. `masks=True` means **observed**, never
"generate". Contact `(value=0, mask=True)` explicitly means no contact;
`(value=1, mask=True)` means contact; `mask=False` means unknown regardless of
the placeholder. Factories fill unknown values with zero, preventing hidden
ground truth from being accidentally passed to the model. Padding has no
observations. `validate()` checks these contracts and returns the batch;
`to(device)` preserves bool masks; `apply(states)` returns a new dictionary with
observed features restored by `torch.where`, preserving all unknown features.

`signatures` is a tuple with one actual observed-modality signature per example:
`none`, `H`, `O`, `C`, `H+O`, `H+C`, `O+C`, or `H+O+C`. Even one observed feature
activates its modality. This deliberately does not label a partly observed
modality as fully known.

## Sampling

`sample_controls(states, valid_frames, mode=None, generator=None,
holdout_signatures=(), allowed_patterns=None, max_attempts=128)` samples each
example independently. A supplied `torch.Generator` makes sampling repeatable.

Seven TriDi-style modes use **generated bits** in H/O/C order:

| Mode | Fully observed | Generated |
| --- | --- | --- |
| `100` | O,C | H |
| `010` | H,C | O |
| `001` | H,O | C |
| `110` | C | H,O |
| `101` | O | H,C |
| `011` | H | O,C |
| `111` | none | H,O,C |

`000` is rejected because there is no generation target. Partial modes are
`human_keyframes`, `object_waypoints`, `contact_intervals`, `root_path`
(root joint 0, world xyz), `body_parts` (joint xyz), `rotation_keyframes`
(joint features 3:9), and `mixed` (human keyframes + object waypoints + contact
intervals). `human_object`, `human_contact`, and `object_contact` combine just
the corresponding two sparse subpatterns, supporting meaningful pairwise
composition holdouts while leaving generation targets in both entities.
The default pool contains all seven full-modality modes and all
partial modes. `allowed_patterns` **replaces** this pool with a whitelist and
can contain either kind; an explicit `mode` must be in the whitelist if given.

Held-out signatures are checked against the actual masks from **every** source,
including full-modality modes. Rejection sampling raises a clear error after
`max_attempts`; it never bypasses a holdout or changes an explicit mode. This
is a coarse modality-composition protocol, not yet a fine-grained contact-part,
object-category or semantic-action composition benchmark.
Sampling also rejects masks that observe every valid feature, which can occur
for a mixed pattern on extremely short sequences: there must be a prediction
target. Explicit JSON compilation may fully anchor an example.

## Explicit JSON controls

`compile_controls(spec, reference, valid_frames)` supports B=1. A reference
provides shape, dtype and device; its values are read **only** when an entry
explicitly sets `from_reference: true`. Explicit values remain world-space.

```json
{
  "entries": [
    {"modality": "human", "frames": [0, 10], "joints": [20],
     "features": [0, 1, 2], "values": [[0.2, 1.0, 0.4], [0.3, 1.1, 0.6]]},
    {"modality": "object", "frames": [20], "features": [0, 1, 2],
     "values": [0.3, 0.8, 0.6]},
    {"modality": "contact", "start": 10, "end": 20,
     "joints": [20], "values": 1},
    {"modality": "contact", "frames": [0], "joints": [20], "values": 0}
  ]
}
```

`start`/`end` are inclusive; use them or `frames`, not both. Indices must be
unique, in bounds and valid frames. Human/contact joints default to all joints.
Human/object features default to all nine. Contact is scalar per joint, so omit
`features` or use `[0]`. Values broadcast to `[frames,joints,features]` for
human, `[frames,features]` for object and `[frames,joints]` for contact. Human
entries with one joint also accept `[frames,features]`. Conflicting overlapping
anchors raise an error; equal repeated anchors are accepted within 1e-6.
Missing values never silently enable reference conditioning.

`check_anchor_feasibility(controls, object_points)` checks a contact only when
the joint xyz, full object pose and contact bit are all observed. It reports
sampled-surface distance mismatches and degenerate rotation anchors. Geometry
sampling density affects these distances. It neither certifies physical
feasibility nor proves that an actual mesh has no feasible solution; its report
always contains `establishes_physical_feasibility: false`.

## Composition generalization without leakage

1. Split original source sequences/sessions before windows and controls are
   sampled. Overlapping windows and augmentations of a source stay together.
   This is a dataset-manifest responsibility, not handled by the mask sampler.
2. For the coarse composition test, exclude `H+O+C` from training, validation
   used for selection and all conditioning augmentation paths. Train on the
   seven generation modes and allowed single/pair observed signatures; evaluate
   the `mixed` pattern only on held-out source sequences. Use the same checkpoint
   selection procedure across methods. The test results must not tune losses.
3. Generate evaluation constraints from each test sequence using fixed saved
   seeds/specifications; thus feasible reference-derived tasks have matched
   conditions across methods. Independently generated human-first inputs form
   a separate robustness set, with no automatic claim that they are feasible.
4. Report unseen condition composition, unseen object identity, and both unseen
   separately. A modality signature alone cannot establish novelty of object
   identity or contact strategy. Any fine-grained pattern holdout needs a
   richer manifest and a check across all sampler branches before claiming it.
5. Measure all-constraint satisfaction in addition to per-type errors. Compare
   the same model trained on all combinations as a control for composition
   difficulty, and an equal-capacity random-mask baseline. Report physical
   execution separately from exact feature restoration, since hard inpainting
   makes the latter true by construction.

The public TriDi implementation uses per-modality diffusion times and `1` for
generated modalities, whereas Kimodo's `motion_mask` uses True for observed
features and its `pad_mask` uses True for valid frames. Kimodo internally
expresses joint positions relative to a smoothed root; this package's public
controls do not reuse those relative values without a coordinate conversion.
