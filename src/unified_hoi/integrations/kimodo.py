"""Real official Kimodo inference and a strict SMPL-X 22-joint output bridge.

The adapter does not load or download models until ``generate_human`` is called.
It does not reinterpret human-only pretrained weights as an HOI checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import sys

from .upstream import UPSTREAMS, checkout_report


DEFAULT_MODEL = "Kimodo-SMPLX-RP-v1"


def default_checkout() -> Path:
    return Path(__file__).resolve().parents[3] / "external" / "kimodo"


@contextmanager
def _official_imports(checkout: Path, text_encoder_device: str):
    """Prefer the requested checkout and restore process settings after inference."""
    if not (checkout / "kimodo" / "model" / "kimodo_model.py").is_file():
        raise FileNotFoundError(f"Not an official Kimodo checkout: {checkout}")
    loaded = sys.modules.get("kimodo")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(checkout):
        raise RuntimeError("A different Kimodo installation is already imported; use a fresh Python process.")
    previous = {key: os.environ.get(key) for key in ("TEXT_ENCODER_MODE", "TEXT_ENCODER_DEVICE")}
    # Keep text encoding local; do not send prompts to an external encoder service.
    os.environ["TEXT_ENCODER_MODE"] = "local"
    os.environ["TEXT_ENCODER_DEVICE"] = text_encoder_device
    sys.path.insert(0, str(checkout))
    try:
        yield
    finally:
        sys.path.remove(str(checkout))
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _check_output_paths(paths, overwrite: bool):
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; choose another output or pass --overwrite.")
        path.parent.mkdir(parents=True, exist_ok=True)


def generate_human(
    prompt: str,
    output_path: str | Path,
    *,
    duration: float = 4.0,
    model_name: str = DEFAULT_MODEL,
    checkout: str | Path | None = None,
    constraints_path: str | Path | None = None,
    device: str | None = None,
    text_encoder_device: str = "cpu",
    diffusion_steps: int = 50,
    seed: int = 1234,
    post_processing: bool = False,
    overwrite: bool = False,
) -> dict:
    """Generate ONE human clip through official load_model and Kimodo.__call__.

    The named model checkpoint and local LLM2Vec assets may be downloaded by
    upstream Hugging Face loaders on first use. Dependencies must be installed
    in the calling environment. ``constraints_path`` is official Kimodo JSON,
    not this project's HOI constraint schema.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if diffusion_steps < 1:
        raise ValueError("diffusion_steps must be positive")
    checkout = Path(checkout or default_checkout()).resolve()
    output_path = Path(output_path).resolve()
    if output_path.suffix != ".npz":
        raise ValueError("output_path must end in .npz")
    metadata_path = output_path.with_suffix(".meta.json")
    _check_output_paths([output_path, metadata_path], overwrite)
    if constraints_path is not None:
        constraints_path = Path(constraints_path).resolve(strict=True)
    provenance = checkout_report(checkout, "kimodo")
    if not provenance.get("pinned") or provenance.get("dirty"):
        raise RuntimeError(f"Use the clean pinned Kimodo checkout: {provenance}")
    with _official_imports(checkout, text_encoder_device):
        try:
            torch = importlib.import_module("torch")
            kimodo = importlib.import_module("kimodo")
            constraints_api = importlib.import_module("kimodo.constraints")
            motion_io = importlib.import_module("kimodo.exports.motion_io")
            seed_everything = importlib.import_module("kimodo.tools").seed_everything
        except ImportError as exc:
            raise RuntimeError(
                "Official Kimodo dependencies are unavailable. Follow docs/UPSTREAM.md "
                "in a dedicated inference environment; no substitute motion was generated."
            ) from exc
        selected_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model, resolved_name = kimodo.load_model(
            model_name,
            device=selected_device,
            return_resolved_name=True,
            text_encoder_fp32=text_encoder_device == "cpu",
        )
        fps = float(model.fps)
        frames = int(round(duration * fps))
        if frames < 1:
            raise ValueError("duration is shorter than one model frame")
        constraints = [] if constraints_path is None else constraints_api.load_constraints_lst(
            str(constraints_path), model.skeleton, device=selected_device
        )
        seed_everything(seed)
        with torch.no_grad():
            # Scalar prompt/length + omitted num_samples produces unbatched T,J output.
            motion = model(
                prompt,
                frames,
                num_denoising_steps=diffusion_steps,
                constraint_lst=constraints,
                post_processing=post_processing,
                return_numpy=True,
            )
        motion_io.save_kimodo_npz(str(output_path), motion)
        skeleton = model.output_skeleton
        metadata = {
            "schema": "official-kimodo-human-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": resolved_name,
            "upstream_commit": provenance["actual_commit"],
            "fps": fps,
            "num_frames": frames,
            "skeleton": skeleton.name,
            "joint_names": list(skeleton.bone_order_names),
            "coordinates": "Y-up, +Z forward, meters; root_positions is world pelvis",
            "prompt": prompt,
            "seed": seed,
            "diffusion_steps": diffusion_steps,
            "post_processing": post_processing,
            "constraints_path": None if constraints_path is None else str(constraints_path),
            "code_license": UPSTREAMS["kimodo"]["code_license"],
            "checkpoint_license_note": "Model-specific; SMPLX-RP-v1 uses NVIDIA R&D Model License.",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"motion": str(output_path), "metadata": str(metadata_path)}


def convert_human_reference(
    input_path: str | Path,
    output_path: str | Path,
    *,
    fps: float | None = None,
    target_fps: float | None = None,
    skeleton: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Convert verified SMPL-X22 raw output to a timed human reference.

    Rotations are converted from full global matrices to the FIRST TWO ROWS.
    Kimodo's internal cont6d uses columns and must not be copied directly.
    Missing object/contact arrays are deliberately omitted; this is a reference,
    not a fabricated paired HOI training sample.

    ``fps`` declares the SOURCE rate when its sidecar is missing; it cannot
    override a different recorded rate. ``target_fps`` only permits integer
    stride downsampling, preserving frame zero and original sample times.
    A static skeleton is inferred from all source frames, before subsampling.
    Local offsets and FK reconstruction must agree within 1e-4 meters.
    """
    import numpy as np
    from ..geometry import SMPL_PARENTS

    input_path = Path(input_path).resolve(strict=True)
    output_path = Path(output_path).resolve()
    if output_path.suffix != ".npz":
        raise ValueError("output_path must end in .npz")
    if output_path == input_path:
        raise ValueError("Reference output must differ from the original Kimodo NPZ")
    source_meta = input_path.with_suffix(".meta.json")
    metadata = json.loads(source_meta.read_text(encoding="utf-8")) if source_meta.is_file() else {}
    declared_skeleton = skeleton or metadata.get("skeleton")
    if declared_skeleton != "smplx22":
        raise ValueError(
            f"Expected explicit smplx22 provenance, got {declared_skeleton!r}. "
            "SOMA/G1 require retargeting and are not supported by this bridge."
        )
    if metadata.get("skeleton") not in (None, "smplx22"):
        raise ValueError("Explicit skeleton conflicts with source metadata")
    source_fps = fps if fps is not None else metadata.get("fps")
    if source_fps is None or not math.isfinite(float(source_fps)) or float(source_fps) <= 0:
        raise ValueError("Provide a positive fps or the generator's .meta.json sidecar")
    source_fps = float(source_fps)
    if fps is not None and metadata.get("fps") is not None:
        recorded_fps = float(metadata["fps"])
        if not math.isfinite(recorded_fps) or recorded_fps <= 0:
            raise ValueError("Source metadata must contain a positive fps")
        if not math.isclose(source_fps, recorded_fps, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("fps conflicts with source metadata; use target_fps to downsample, not relabel time")
        source_fps = recorded_fps
    stride = 1
    if target_fps is not None:
        target_fps = float(target_fps)
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("target_fps must be finite and positive")
        if target_fps > source_fps:
            raise ValueError("target_fps cannot exceed source fps; upsampling is not supported")
        ratio = source_fps / target_fps
        if not math.isfinite(ratio):
            raise ValueError("target_fps must yield a finite integer stride")
        stride = round(ratio)
        if stride < 1 or not math.isclose(ratio, stride, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("target_fps must divide source fps by an integer stride")
    output_fps = source_fps / stride
    with np.load(input_path, allow_pickle=False) as arrays:
        positions = np.asarray(arrays["posed_joints"], dtype=np.float32)
        rotations = np.asarray(arrays["global_rot_mats"], dtype=np.float32)
    if positions.ndim == 4 and positions.shape[0] == 1:
        positions = positions[0]
    if rotations.ndim == 5 and rotations.shape[0] == 1:
        rotations = rotations[0]
    if positions.ndim != 3 or positions.shape[1:] != (22, 3) or positions.shape[0] < 1:
        raise ValueError(f"Expected one nonempty clip of shape [T,22,3], got {positions.shape}")
    if rotations.shape != (*positions.shape[:2], 3, 3):
        raise ValueError(f"Rotation shape does not match positions: {rotations.shape}")
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError("Kimodo output contains nonfinite positions or rotations")
    gram = rotations @ np.swapaxes(rotations, -1, -2)
    if not np.allclose(gram, np.eye(3), atol=2e-3) or not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-3):
        raise ValueError("global_rot_mats are not valid right-handed rotation matrices")
    # Kimodo's default inverse() returns FK-derived world joints. Each child's
    # rest offset is expressed in its PARENT's frame, not in the child's frame.
    source_frames = len(positions)
    positions64 = positions.astype(np.float64)
    rotations64 = rotations.astype(np.float64)
    parents = np.asarray(SMPL_PARENTS[1:])
    world_offsets = positions64[:, 1:] - positions64[:, parents]
    local_offsets = np.einsum("tjki,tjk->tji", rotations64[:, parents], world_offsets)
    rest_offsets = np.zeros((22, 3), dtype=np.float64)
    rest_offsets[1:] = local_offsets.mean(axis=0)
    max_offset_error = float(np.linalg.norm(local_offsets - rest_offsets[None, 1:], axis=-1).max())
    bone_lengths = np.linalg.norm(world_offsets, axis=-1)
    max_bone_length_error = float(np.abs(bone_lengths - bone_lengths.mean(axis=0)).max())
    reconstructed = np.zeros_like(positions64)
    reconstructed[:, 0] = positions64[:, 0]
    for joint, parent in enumerate(SMPL_PARENTS[1:], start=1):
        reconstructed[:, joint] = reconstructed[:, parent] + np.einsum(
            "tik,k->ti", rotations64[:, parent], rest_offsets[joint]
        )
    max_fk_error = float(np.linalg.norm(reconstructed - positions64, axis=-1).max())
    skeleton_tolerance = 1e-4  # meters; also bounds accumulated FK error along each chain.
    if max(max_offset_error, max_bone_length_error, max_fk_error) > skeleton_tolerance:
        raise ValueError(
            "Source motion has inconsistent static skeleton offsets: "
            f"offset error={max_offset_error:.6g} m, bone-length error={max_bone_length_error:.6g} m, "
            f"FK error={max_fk_error:.6g} m; tolerance={skeleton_tolerance:g} m"
        )
    source_indices = np.fromiter(range(0, source_frames, stride), dtype=np.int64)
    timestamps = source_indices.astype(np.float64) / source_fps
    positions = positions[source_indices]
    rotations = rotations[source_indices]
    human = np.concatenate([positions, rotations[..., :2, :].reshape(*positions.shape[:2], 6)], axis=-1)
    reference_meta_path = output_path.with_suffix(".meta.json")
    _check_output_paths([output_path, reference_meta_path], overwrite)
    np.savez_compressed(
        output_path,
        human=human,
        human_mask=np.ones_like(human, dtype=np.bool_),
        fps=np.asarray(output_fps, dtype=np.float64),
        source_fps=np.asarray(source_fps, dtype=np.float64),
        timestamps=timestamps,
        rest_offsets=rest_offsets.astype(np.float32),
    )
    reference_meta = {
        "schema": "unified-hoi-human-reference-v2",
        "source": str(input_path),
        "source_metadata": metadata,
        "fps": output_fps,
        "source_fps": source_fps,
        "source_num_frames": source_frames,
        "num_frames": len(human),
        "downsample_stride": stride,
        "timestamps": "seconds from source frame zero; retained source_index / source_fps",
        "skeleton": "smplx22",
        "rest_offsets": "mean parent-frame offsets inferred from all source frames; root offset is zero",
        "skeleton_tolerance_m": skeleton_tolerance,
        "max_offset_error_m": max_offset_error,
        "max_bone_length_error_m": max_bone_length_error,
        "max_fk_error_m": max_fk_error,
        "human_layout": "world joint xyz + global rotation matrix first two rows",
        "coordinates": "Y-up, meters; no coordinate transform applied",
        "root_semantics": "joint 0 is world pelvis position, not SMPL transl",
        "not_a_paired_training_sample": True,
    }
    reference_meta_path.write_text(json.dumps(reference_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"reference": str(output_path), "metadata": str(reference_meta_path), "frames": len(human)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Call official Kimodo; may download model assets")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--duration", type=float, default=4.0)
    generate.add_argument("--model", default=DEFAULT_MODEL)
    generate.add_argument("--checkout", type=Path, default=default_checkout())
    generate.add_argument("--constraints", type=Path)
    generate.add_argument("--device")
    generate.add_argument("--text-encoder-device", default="cpu")
    generate.add_argument("--diffusion-steps", type=int, default=50)
    generate.add_argument("--seed", type=int, default=1234)
    generate.add_argument("--post-processing", action="store_true")
    generate.add_argument("--overwrite", action="store_true")
    convert = sub.add_parser("convert", help="Convert existing SMPL-X22 NPZ; no weights required")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--fps", type=float, help="Source rate if missing; must match any existing sidecar")
    convert.add_argument("--target-fps", type=float, help="Output rate; integer stride downsampling only")
    convert.add_argument("--skeleton", choices=["smplx22"])
    convert.add_argument("--overwrite", action="store_true")
    probe = sub.add_parser("probe", help="Inspect pinned source; no imports/downloads")
    probe.add_argument("--checkout", type=Path, default=default_checkout())
    args = parser.parse_args(argv)
    if args.command == "probe":
        result = checkout_report(args.checkout, "kimodo")
    elif args.command == "convert":
        result = convert_human_reference(args.input, args.output, fps=args.fps, target_fps=args.target_fps,
                                         skeleton=args.skeleton, overwrite=args.overwrite)
    else:
        result = generate_human(
            args.prompt, args.output, duration=args.duration, model_name=args.model,
            checkout=args.checkout, constraints_path=args.constraints, device=args.device,
            text_encoder_device=args.text_encoder_device, diffusion_steps=args.diffusion_steps,
            seed=args.seed, post_processing=args.post_processing, overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
