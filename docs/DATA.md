# Data conversion and the sequence contract

The converter consumes licensed official **raw sequence annotations**, not rendered
videos, TriDi frame datasets, or OMOMO's already-windowed motion vectors. No source
dataset or SMPL model is included. No semantic text is invented from filenames.

## Archive format

One NPZ stores one continuous sequence (or a continuous part of a sequence):

| Field | Shape / convention |
| --- | --- |
| `human` | float32 `[T,22,9]`: world joint position in metres, then **global** rotation6d |
| `object` | float32 `[T,9]`: template-origin translation, then rotation6d |
| `contact` | float32 `[T,22]`, binary joint-to-sampled-object-surface distance proxy |
| `object_points` | float32 `[K,3]`, fixed object-local points in metres, default K=1024 |
| `rest_offsets` | float32 `[22,3]`, parent-relative offsets in the SMPL22 rest frame |
| `timestamps` | float64 `[T]`, strictly increasing actual seconds |
| `fps` | positive scalar; every adjacent timestamp must agree, allowing millisecond rounding |
| `sequence_id`, `dataset`, `subject_id`, `object_id`, `text` | scalar strings |

Rotation6d is the first **two matrix rows**, concatenated, matching PyTorch3D.
Column vectors transform as `v_world = R @ v_local + translation`. All world
quantities use a right-handed Y-up frame. A single fixed rotation transforms the
entire sequence; there is no per-frame centering or heading alignment. World-frame
rotations are left-multiplied by this rotation. Object-local points and local bone
offsets stay in their original local frames, so the geometry equation remains valid.

The object translation is not necessarily its center of mass. Do not replace
`obj_trans` with `obj_com_pos` without changing the object template origin as well.
Contact is a geometric supervision proxy (default 5 cm), not measured hand contact,
penetration-free interaction, force closure, or physical feasibility. SMPL22 does
not include fingers. Increasing the surface sample count reduces sampling error,
but does not turn this proxy into mesh or force ground truth.

Extra provenance fields include `source_to_world`, `source_sequence_id`, `world_up`,
`units`, `contact_definition`, and `timestamps_origin`.

An optional `floor_y` is a calibrated world-space floor height in metres. The
dataset does not infer a physical floor from the coordinate origin. Without this
field, floor regularization is disabled and evaluation's zero-plane diagnostic is
marked `floor_calibrated=false`. Timing-dependent diagnostics also carry
`timing_verified_by_metadata`; explicitly assumed OMOMO rates remain false.

## BEHAVE 30 fps

Install the optional preprocessing dependencies and obtain licensed SMPL-H model
files from their official provider. The body-model directory is passed to
`smplx.build_layer(..., model_type="smplh", use_pca=False)`.

```bash
python -m unified_hoi.preprocess behave --help
python -m unified_hoi.preprocess behave \
  --input data/raw/behave \
  --objects data/raw/behave/objects \
  --smpl-models data/smplx_models \
  --split-file data/raw/behave/split.json \
  --val-from-train 0.1 --split-seed 0 \
  --target-fps 10 --points 1024 --output data/processed/behave
```

Each sequence uses `info.json`, `smpl_fit_all.npz` (`poses`, `betas`, `trans`,
`frame_times`) and `object_fit_all.npz` (`angles`, `trans`, `frame_times`). Human and
object samples are matched by their actual timestamps. The template mesh is first
centered using its **vertex mean**, as in official TriDi's BEHAVE30 preprocessing,
before applying object angles/translation. The source camera frame has Y pointing
down; a fixed 180-degree X rotation converts to Y-up. Original timestamps survive.

Missing intervals are split into separate continuous records before downsampling.
The output remembers the common source sequence ID for all resulting parts.
`--target-fps` must divide the actual source fps. **1 fps annotations must not be
relabeled or upsampled to 30 fps.** A claimed source rate with no matching intervals
is rejected. Sparse BEHAVE `person/fit02/person_fit.pkl` directories are not accepted
by the full-rate converter. Time-varying body-shape coefficients are explicitly
rejected rather than silently choosing a different skeleton each frame.

Source references: `external/tridi/tridi/preprocessing/preprocess_behave_30fps.py`
(timestamp intersection, SMPL-H body parameters, object-template centering) and
`external/tridi/docs/data.md` (official dataset layout).

## OMOMO

```powershell
python -m unified_hoi.preprocess omomo --help
python -m unified_hoi.preprocess omomo `
  --input D:/code/HOI/OMOMO/data/train_diffusion_manip_seq_joints24.p `
  --objects D:/code/HOI/OMOMO/data/captured_objects `
  --source-fps 30 --target-fps 10 `
  --fps-provenance "Assumed 30 fps from official OMOMO visualization; raw release metadata has no rate" `
  --output data/processed/omomo_train
```

**30 fps in this example is an explicit, unverified release-rate assumption.**
Replace it when the actual annotation rate is confirmed. Capturing at 120 fps does
not establish the rate of the released annotation arrays.
The official sequence loader does not carry a universal fps field; TriDi's transfer
script assumes 120 fps, while OMOMO's visualization writes 30 fps videos. Neither
is a sufficient reason to silently assign a rate to an arbitrary release. The CLI
requires the rate explicitly and checks a record's fps when present. The archive,
manifest, and conversion report preserve `fps_provenance` and whether metadata
verified it. Reconstructed timestamps depend on this assumption; time-dependent
speed and physics metrics should not be reported as validated until it is resolved.
Use `--limit 10` for an initial conversion check.

Raw sequence fields are `seq_name`, `trans`, `root_orient`, `pose_body`,
`rest_offsets`, `trans2joint`, `obj_trans`, `obj_rot`, and `obj_scale`. These give
22-joint forward kinematics directly, without needing a proprietary body model:

```text
human_root_joint = trans - trans2joint
object_world_vertices = obj_scale * (obj_rot @ rest_vertices) + obj_trans
```

The source frame is Z-up. A fixed -90-degree X rotation converts world quantities
to Y-up. The first 22 rest offsets define the common SMPL22 skeleton. Real release
annotations have small frame-to-frame Procrustes scale variation. By default, a
sequence is accepted when `(scale.max - scale.min) / median(scale) <= 0.01`; its
median scale is baked into fixed local object points. **This is an approximation
that changes annotated geometry, not an error-free recovery of a rigid object.**
`--scale-relative-tolerance` changes this policy; `0` requires exactly constant
scale. Larger variations are excluded with the measured relative range in the report.

For every accepted sequence, the archive and manifest record median scale,
relative range, and `scale_max_surface_displacement_m`: the maximum absolute scale
deviation times the maximum template-vertex radius. The CLI computes this bound
over the full template mesh, so it bounds the change at every mesh surface point
and frame. Direct record conversions without a mesh radius instead explicitly
label the bound `sampled_surface`. The report includes the maximum over accepted
sequences. Articulated `mop` / `vacuum` top-bottom geometry is always excluded;
this version models one rigid body and never merges independently moving parts.

The official split is subjects `sub1` through `sub15` for training and `sub16`,
`sub17` for test. `--split-file` can supply an explicit sequence-level train/val/test
mapping instead. `--split` is an explicit override for all input sequences.
`--val-from-train 0.2 --split-seed 0` assigns approximately 20% of training
sequences to validation using a stable hash of dataset, sequence ID, and seed.
This is done before windowing and never reassigns test sequences. The result is
stable when input order or `--limit` changes; tiny subsets need not have exactly
the requested fraction. Do not switch seeds or holdout fractions across train/val
conversions of the same dataset.

Do not pass `*_window_*` files: their `motion[T,276]` concatenates 24 joint
positions, 24 frame displacements, and 22 global rotations after window-specific
canonicalization. It is not the raw pose and does not contain actual velocities in
metres/second. Source: `external/omomo/manip/data/hand_foot_dataset.py`, methods
`apply_transformation_to_obj_geometry`, `load_object_geometry`,
`cal_normalize_data_input`, and `process_window_data`.

## Why not use TriDi's preprocessed HDF5 for temporal training?

TriDi intentionally trains on static frames. Its preprocessing recenters **each
frame** on the pelvis and filters non-contact frames, then renumbers frame indices.
The HDF5 includes inverse-transform fields, but the allocated `orig_t_stamp` dataset
is not written by `DatasetSample.dump_hdf5`. The optional 1 fps/10 fps merging can
also produce mixed temporal rates in a filename labelled 10 fps. Reconstructing a
continuous world-space motion by concatenating those indices is invalid. Convert
from raw annotations and retain approach/release/non-contact frames instead.

## Manifests, split integrity, and text

Each conversion writes `manifest.jsonl` and `conversion_report.json`, including all
rejected sequence IDs and reasons. Existing output manifests require explicit
`--overwrite`. A manifest line includes relative `path`, `dataset`, `sequence_id`,
`source_sequence_id`, `subject_id`, `object_id`, `split`, `fps`, `frames`, and `text`.
Relative paths are resolved against the manifest directory, not the working directory.
To combine manifests, update relative paths or write absolute paths in a new manifest.

`HOIDataset(manifest, split, window, stride, text_dim=512, text_condition=True)` selects sequences by split
before taking any windows. It rejects a common source sequence assigned to multiple
splits, mixed object-point counts, and mixed frame rates. All parts/overlapping
windows from one source sequence must stay in one split. Fit normalization and text
encoders using the training split only. Do not create a random window-level split.

Short final windows repeat the last pose and have `valid_frames=False` for padding;
their padded timestamps also repeat. Use `valid_frames` to exclude padding from all
losses and metrics. No motion is interpolated. The returned dictionary batches with
PyTorch's default collator.

Training passes the config's `text_condition` switch to both train and validation
datasets. With `false`, cached embeddings are not opened and every sample returns
zero text features and `text_available=False`; the same manifest can therefore
serve both text-conditioned and no-text runs. Explicit `true` rejects a training
split with no cache paths. Omitted switches retain the previous use-if-present
behavior. Strict resume rejects changes to the switch.

Text defaults to `""`. Supply verified annotations through `--text-file`, a JSON
mapping from source sequence IDs to text. Optional manifest `text_features_path`
points to a finite `.npy` embedding of shape `[text_dim]` from the selected frozen
text encoder. Without a cache, the dataset returns zeros and `text_available=False`,
even if verified text exists. This prevents random or fabricated embeddings from
masquerading as text conditioning.

Conversion fixtures and temporal/split checks are covered by `tests/test_data.py`.
Actual licensed-dataset geometry and body-model validation require the corresponding
assets; passing synthetic fixtures does not establish real-data correctness.

## Local data validation performed on 2026-09-28

The provided OMOMO release contains 5,280 training and 602 test raw sequences.
Neither raw dictionary supplies frame-rate or timestamp metadata. A smoke conversion
of the first 20 training sequences and first 10 test sequences succeeded, explicitly
assuming 30 fps and downsampling to 10 fps. Stable training holdout with fraction
0.2 and seed 0 produced 18 train and 2 validation records; all 10 test records
remained test. No sequence in these small subsets was rejected. Maximum mesh
surface displacement from median-scale approximation was 0.818 mm for the training
subset and 1.190 mm for the test subset (rounded upward). Converted global rotations
and rest offsets reproduced converted joint positions within 0.00000036 metres.
Default PyTorch DataLoader collation was checked on real records.

These subsets only exercise an early portion of the release; they are not an
evaluation benchmark and do not validate all object categories. Full-conversion
reports must be inspected for articulated objects and excessive scale variation.
The complete release does contain both, unlike these early smoke subsets.

The downloaded BEHAVE release contains 299 sequence directories. Three inspected
human parameter archives have `[T,156]` poses, constant `[T,10]` betas, and actual
`frame_times` at approximately 30 fps. SMPL-H assets are a separate download.
The official ready-to-load male/female models are retained in `data/smplx_models/smplh/`.
The redundant `smplx.zip` was verified against both extracted files and archived
outside `data/`; see `docs/data_cleanup_report.json` for its recoverable location.
Both pass CPU loading and a forward pass with `num_betas=10, use_pca=False`;
see `data/smplx_models/smplh_download_report.json` for hashes and checks.
Full BEHAVE conversion has now completed on the local CPU: 293 sequences,
147,520 frames at 10 fps, and 1,024 sampled points per object. The split is
194 train / 17 validation / 82 test, using the official train/test mapping and
a source-level 0.1 validation holdout with seed 0. Six source sequences are absent
from the official mapping and are explicitly rejected; there are no other
conversion errors. All 293 outputs were checked against original timestamps and
object transforms. Maximum FK error is approximately 7.46e-7 metres. Five records
have no positive joint-to-surface contact proxy labels; no contacts were invented.
These are historical full-source conversion results. The current
`data/processed/behave/sequences/` contains 1,454 text-aligned clips, with reports
in `data/processed/behave/preparation_report.json` and `verification_report.json`.
Full source archives can be rebuilt into a separate `data/processed/behave_source/`
using the retained raw parameters, object meshes, and SMPL-H models.

BEHAVE and OMOMO are trained and validated separately. Use
`data/processed/behave_with_text.jsonl` for BEHAVE-only runs and
`data/processed/omomo_with_text.jsonl` for OMOMO-only runs. These manifests support
both settings of `text_condition`. Do not merge them into
a joint training manifest. Linux/CUDA validation and formal training remain
pending; the OMOMO source-fps assumption still applies even though both datasets
are stored at 10 fps.
