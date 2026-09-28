# Official upstream integrations

This project uses real upstream code, with an explicit boundary between reused
components and the new HOI model. The official repositories and checkpoints are
not interchangeable with the new model's checkpoint.

## Pinned source

| Repository | Commit | Source-code license |
| --- | --- | --- |
| [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) | `58e781898b3d7e328a676a75d3e338c45dce3ad9` | Apache-2.0 |
| [ptrvilya/tridi](https://github.com/ptrvilya/tridi) | `afa9631dc2b3a250588ab64026eeaa37f18f0d38` | MIT |
| [lijiaman/omomo_release](https://github.com/lijiaman/omomo_release) | `e9d8c52f41866ed0c07ac574cb8f7f9424b6d1e7` | MIT |

Run `python scripts/fetch_upstreams.py --verify-only` to check existing source
checkouts without using the network. Omit `--verify-only` to clone missing
repositories at these exact commits. Existing repositories with a different
commit, remote, or local changes are reported and left unchanged; the script
does not reset, clean, or overwrite them. `--only kimodo` / `--only tridi` select
one repository. This script never downloads model weights or datasets.

The OMOMO source checkout supplies the official sequence-loader reference and its
shipped `omomo_text_anno.zip`; motion annotations remain in the user's `OMOMO/data`.
The optional source text ZIP contains 4912 verified sequence-level sentences. Code
licenses do not replace the dataset/asset terms.

The manifest is in `src/unified_hoi/integrations/upstream.py`. Complete source
licenses are retained in `src/unified_hoi/integrations/licenses/`. Body assets,
datasets, pretrained checkpoints, and embedded third-party code can have their
own licenses. In particular, Kimodo's README lists **NVIDIA R&D Model License**
for `Kimodo-SMPLX-RP-v1`, and **NVIDIA Open Model License** for its SOMA/G1 models.
Apache-2.0 describes Kimodo source code, not every model asset. TriDi's embedded
SMPL-X conversion directory retains separate SMPL-X and GRAB notices.

## Code that is actually reused

### TriDi primitives

`integrations/_tridi_primitives.py` contains the exact function/class bodies of
`get_timestep_embedding` and `Projection` from
[`tridi/model/denoising/transformer_uni_3.py`](https://github.com/ptrvilya/tridi/blob/afa9631dc2b3a250588ab64026eeaa37f18f0d38/tridi/model/denoising/transformer_uni_3.py).
That source file has Git blob ID `9115e4abe94956d0f6b8708ca6d1a734fbd4d2ad`.
Only extraction and provenance comments are new. An AST comparison against the
pinned checkout is part of `fetch_upstreams.py --verify-only`; formatting and
line endings do not affect this check.

The new model can import the vendored `Projection` directly, or use
`integrations.upstream.get_projection_class()` and `get_timestep_embedding()`.
These primitives require PyTorch and NumPy but do not import TriDi's dataset,
mesh, PointNeXt, SMPL, diffusion-package, or external asset stack. Importing the
provenance module itself needs only the Python standard library.

Other inspected official TriDi code, **not claimed as a directly reused trainer**:

- `tridi/model/tridi.py:TriDiModel` constructs seven H/O/contact diffusion-time
  configurations and provides training and conditional sampling.
- `tridi/model/denoising/transformer_uni_3.py:TransformertUni3WayModel` tokenizes
  static SMPL+H shape, orientation, pose, translation, object pose, and contact.
- `tridi/model/conditioning/contact_ae_clip.py` defines the contact encoders and
  decoder. Its trained contact latent is not a vector of 22 joint contact labels.
- `tridi/core/trainer.py` and `main.py` provide the official training workflow;
  that workflow requires the original representation and assets.

### Official Kimodo inference

`integrations/kimodo.py:generate_human` calls the actual official API:

1. `kimodo.load_model(..., return_resolved_name=True)` from
   `kimodo/model/load_model.py` loads the upstream configuration and weights.
2. `kimodo.constraints.load_constraints_lst(path, model.skeleton)` reads the
   official JSON constraint format. It supports `Root2DConstraintSet`,
   `FullBodyConstraintSet`, and `EndEffectorConstraintSet` / hand/foot variants.
3. `Kimodo.__call__(prompt, num_frames, num_denoising_steps=..., constraint_lst=...)`
   runs the upstream two-stage denoiser. This is the actual callable API; the
   upstream class overrides `__call__`, rather than exposing a generation
   `forward` method.
4. `kimodo.exports.motion_io.save_kimodo_npz` saves the official raw output.

The adapter also saves a `.meta.json` sidecar with the source commit, resolved
model name, skeleton/joint names, FPS, prompt, seed, constraints path, and
coordinate convention. Text encoding is explicitly local. `probe` and `convert`
never load checkpoints. `generate` can download missing upstream model and
LLM2Vec assets; it is never run automatically by the training/data scripts.

The pinned public Kimodo tree contains inference, interactive-demo and benchmark
code, model/representation classes, and loading utilities. Its README explicitly
lists those releases. We did not find a motion-model training entry point,
dataset training pipeline, or training loss implementation in this tree. A
module's `.train()` switch is not an official training pipeline. The new HOI
training loop must therefore be described as this project's implementation.

## Running the adapter

Install the project's package so that `unified_hoi` is importable, then use a
separate environment with the official Kimodo dependencies when doing inference.
The source tree's `pyproject.toml`, `setup.py`, and installation documentation
describe those dependencies. Full installation builds the optional C++ motion
correction component using CMake. For inference without correction, upstream
`setup.py` supports `SKIP_MOTION_CORRECTION_IN_SETUP=1`; the adapter defaults to
`post_processing=False`. Choose a suitable PyTorch/CUDA installation separately.
Dependencies and large checkpoints are not installed by this adapter.

```text
python -m unified_hoi.integrations.kimodo probe

python -m unified_hoi.integrations.kimodo generate --prompt "A person picks up a box" --duration 4 --model Kimodo-SMPLX-RP-v1 --device cuda:0 --output outputs/kimodo/human.npz

python -m unified_hoi.integrations.kimodo convert --input outputs/kimodo/human.npz --output outputs/kimodo/human_reference.npz --target-fps 10
```

Add `--constraints path/to/kimodo_constraints.json` to `generate` to use the
official constraint API. That file is **Kimodo's schema**, not the new HOI
constraint schema. The adapter generates one clip per call, to make shape and
provenance unambiguous. CPU text encoding is the default; local LLM2Vec still
requires substantial CPU RAM. Actual pretrained generation must be reported as
unverified until model assets and dependencies are installed and the command
has run successfully.

For an NPZ created by upstream's CLI without this adapter's sidecar, conversion
requires explicit provenance:

```text
python -m unified_hoi.integrations.kimodo convert --input official.npz --output human_reference.npz --skeleton smplx22 --fps 30
```

Pass `--fps` using the source model's actual FPS; it cannot override a different
rate in the source sidecar. `--target-fps` performs actual integer-stride
downsampling, e.g. 30 to 10 retains frames 0,3,6,... and their source timestamps.
It rejects upsampling and noninteger ratios; it never just relabels time.
The bridge rejects SOMA/G1 and multi-sample batches; retargeting them is a separate
operation, not an implicit renaming of joints.

## Exact representation bridge

The current project convention is Y-up, meters, 22 SMPL-X body joints:

- `human[T,22,9]`: **world** joint positions followed by global rotation 6D,
  with the **first two matrix rows** flattened.
- `object[T,9]`: world position and the same rotation convention.
- `contact[T,22]`: human-object contact per joint.

For `smplx22`, the bridge reads `posed_joints[T,22,3]` and
`global_rot_mats[T,22,3,3]` from the official NPZ. It validates finite values,
rotation orthonormality and handedness, then uses
`global_rot_mats[..., :2, :].reshape(T,22,6)`.
**Kimodo's internal `matrix_to_cont6d` uses the first two columns**, so copying
its internal 6D features would silently change the convention.

The saved v2 reference contains `human`, `human_mask` (all observed features),
`fps`, `source_fps`, `timestamps`, and `rest_offsets[22,3]`. Static skeleton offsets
are recovered as `R_parent.T @ (p_child-p_parent)` over ALL original frames,
before downsampling. Maximum local-offset, bone-length and accumulated FK errors
must each be at most 1e-4 metres. An inconsistent export is rejected rather than
silently assigned the skeleton of a different dataset subject. The HOI sampling
CLI uses these offsets as its body-shape condition when a prior is supplied.

The reference omits object and contact data. This is an input for human-conditioned
HOI generation, **not paired interaction training data**. Users can subsequently
select a subset of observed human features using the project's condition masks.
The world pelvis position is retained. It is not SMPL's `transl`: upstream's
`exports/smplx.py:get_amass_parameters` explicitly subtracts a neutral pelvis
offset to compute that quantity. The bridge deliberately does not apply the
AMASS export's Z-up coordinate conversion.

## Checkpoint compatibility

**Kimodo checkpoints cannot be loaded unchanged into a model with appended
object/contact channels.** Its `TwostageDenoiser` derives root/body input and
output dimensions from `motion_rep.motion_rep_dim`, and
`TransformerEncoderBlock` builds dimension-specific input/output projections.
Its checkpoint loading uses strict `load_state_dict`. Normalization statistics,
the skeleton, feature ordering and two-stage root conversion also matter.
Using `strict=False` does not resolve same-key tensor-size mismatches or make new
output heads trained. Partial initialization needs an explicit conversion with
reported copied/skipped keys and subsequent HOI training.

**TriDi checkpoints are not compatible with the new temporal 22-joint model.**
The official default human vector is 325 dimensions (`10 + 52*6 + 3`, SMPL+H),
object pose is 9 dimensions, and contact latent is 128 dimensions.
`prepare_sbj` explicitly splits `[10,6,51*6,3]`; its inputs are `[B,D]` static
poses. The new model uses `[B,T,...]`, has no equivalent 30-joint finger pose or
10-dimensional shape output, and exposes explicit contact labels rather than
the pretrained contact autoencoder latent. No silent zero-padding, guessed
contact latent, or falsely labeled pretrained HOI checkpoint is provided.
