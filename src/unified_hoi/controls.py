"""Observed-value controls for a joint human/object/contact sequence model.

Masks always use True=observed. A contact value of zero is an explicit negative
contact constraint, not an unknown value. Public positions are world-space,
Y-up metres; six-dimensional rotations store the first two matrix rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


MODALITIES = ("human", "object", "contact")
GENERATED_MODES = ("100", "010", "001", "110", "101", "011", "111")
PARTIAL_PATTERNS = (
    "human_keyframes", "object_waypoints", "contact_intervals", "mixed",
    "root_path", "body_parts", "rotation_keyframes",
    "human_object", "human_contact", "object_contact",
)
_SYMBOLS = ("H", "O", "C")


def _validate_states(states: Mapping[str, Tensor], valid_frames: Tensor) -> None:
    if valid_frames.dtype != torch.bool or valid_frames.ndim != 2:
        raise ValueError("valid_frames must be a bool tensor [B, T] (True=valid)")
    if any(size < 1 for size in valid_frames.shape):
        raise ValueError("control batches require nonempty batch and time dimensions")
    if not all(name in states for name in MODALITIES):
        raise ValueError(f"states must contain {MODALITIES}")
    human, obj, contact = (states[name] for name in MODALITIES)
    if human.ndim != 4 or human.shape[-1] != 9 or human.shape[2] < 1:
        raise ValueError("human must have shape [B, T, J, 9] with J >= 1")
    if obj.shape != (*human.shape[:2], 9):
        raise ValueError("object must have shape [B, T, 9]")
    if contact.shape != human.shape[:3]:
        raise ValueError("contact must have shape [B, T, J]")
    if tuple(human.shape[:2]) != tuple(valid_frames.shape):
        raise ValueError("state batch/time dimensions must match valid_frames")
    if not human.is_floating_point() or not obj.is_floating_point():
        raise ValueError("human and object values must be floating point")
    if any(states[name].device != valid_frames.device for name in MODALITIES):
        raise ValueError("states and valid_frames must be on the same device")


def canonical_signature(signature: str) -> str:
    """Canonical observed-modality signature; 'none' means no observations."""
    if not isinstance(signature, str):
        raise ValueError("signatures must be strings such as 'H+O' or 'none'")
    if signature.strip().lower() in ("", "none"):
        return "none"
    parts = [part.strip().upper() for part in signature.split("+")]
    if len(parts) != len(set(parts)) or not set(parts) <= set(_SYMBOLS):
        raise ValueError(f"invalid observed-modality signature: {signature!r}")
    return "+".join(symbol for symbol in _SYMBOLS if symbol in parts)


def control_signatures(masks: Mapping[str, Tensor]) -> tuple[str, ...]:
    """Compute signatures from actual masks, including partial observations."""
    batch_size = masks["human"].shape[0]
    active = [masks[name].reshape(batch_size, -1).any(dim=1) for name in MODALITIES]
    return tuple(
        "+".join(symbol for symbol, present in zip(_SYMBOLS, active) if present[b].item()) or "none"
        for b in range(batch_size)
    )


@dataclass
class ControlBatch:
    values: dict[str, Tensor]
    masks: dict[str, Tensor]
    valid_frames: Tensor
    signatures: tuple[str, ...] | None = None

    def validate(self) -> "ControlBatch":
        _validate_states(self.values, self.valid_frames)
        if set(self.masks) != set(MODALITIES) or set(self.values) != set(MODALITIES):
            raise ValueError("control values and masks must have exactly three modality keys")
        for name in MODALITIES:
            value, mask = self.values[name], self.masks[name]
            if mask.dtype != torch.bool or mask.shape != value.shape:
                raise ValueError(f"{name} mask must be bool and match its value shape")
            if mask.device != value.device:
                raise ValueError(f"{name} mask and values must share a device")
            valid = self.valid_frames.reshape(*self.valid_frames.shape, *([1] * (mask.ndim - 2)))
            if (mask & ~valid).any():
                raise ValueError(f"{name} has observations on invalid/padded frames")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} control values must be finite; zero-fill unknown values")
        observed_contacts = self.values["contact"][self.masks["contact"]]
        if not ((observed_contacts == 0) | (observed_contacts == 1)).all():
            raise ValueError("observed contact values must be exactly 0 or 1")
        actual = control_signatures(self.masks)
        if self.signatures is not None:
            supplied = tuple(canonical_signature(s) for s in self.signatures)
            if supplied != actual:
                raise ValueError("supplied signatures disagree with the actual observation masks")
        self.signatures = actual
        return self

    @classmethod
    def empty_like(cls, states: Mapping[str, Tensor], valid_frames: Tensor) -> "ControlBatch":
        _validate_states(states, valid_frames)
        return cls(
            values={name: torch.zeros_like(states[name]) for name in MODALITIES},
            masks={name: torch.zeros_like(states[name], dtype=torch.bool) for name in MODALITIES},
            valid_frames=valid_frames.clone(),
        ).validate()

    def to(self, device: torch.device | str) -> "ControlBatch":
        return ControlBatch(
            {name: value.to(device) for name, value in self.values.items()},
            {name: mask.to(device) for name, mask in self.masks.items()},
            self.valid_frames.to(device),
            self.signatures,
        ).validate()

    def apply(self, states: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Replace only observed features, preserving other fields and padding."""
        self.validate()
        _validate_states(states, self.valid_frames)
        result = dict(states)
        for name in MODALITIES:
            if states[name].shape != self.values[name].shape:
                raise ValueError(f"{name} target shape does not match controls")
            result[name] = torch.where(self.masks[name], self.values[name], states[name])
        return result


def _randint(high: int, generator: torch.Generator | None) -> int:
    device = generator.device if generator is not None else torch.device("cpu")
    return int(torch.randint(high, (1,), generator=generator, device=device).item())


def _choose(items: list[int], generator: torch.Generator | None, maximum: int = 3) -> list[int]:
    count = 1 + _randint(min(maximum, len(items)), generator)
    available = list(items)
    result = []
    for _ in range(count):
        result.append(available.pop(_randint(len(available), generator)))
    return result


def _sample_pattern(masks: dict[str, Tensor], b: int, frames: list[int], pattern: str,
                    generator: torch.Generator | None) -> None:
    njoints = masks["human"].shape[2]
    if pattern in GENERATED_MODES:
        for bit, name in zip(pattern, MODALITIES):
            if bit == "0":
                masks[name][b, frames] = True
    elif pattern == "human_keyframes":
        masks["human"][b, _choose(frames, generator)] = True
    elif pattern == "object_waypoints":
        masks["object"][b, _choose(frames, generator)] = True
    elif pattern == "contact_intervals":
        left, right = sorted((_randint(len(frames), generator), _randint(len(frames), generator)))
        joints = _choose(list(range(njoints)), generator, maximum=2)
        for frame in frames[left:right + 1]:
            masks["contact"][b, frame, joints] = True
    elif pattern == "root_path":
        masks["human"][b, frames, 0, :3] = True
    elif pattern == "body_parts":
        joints = _choose(list(range(njoints)), generator, maximum=2)
        for frame in _choose(frames, generator):
            masks["human"][b, frame, joints, :3] = True
    elif pattern == "rotation_keyframes":
        joints = _choose(list(range(njoints)), generator, maximum=2)
        for frame in _choose(frames, generator):
            masks["human"][b, frame, joints, 3:] = True
    elif pattern in ("mixed", "human_object", "human_contact", "object_contact"):
        combinations = {
            "mixed": ("human_keyframes", "object_waypoints", "contact_intervals"),
            "human_object": ("human_keyframes", "object_waypoints"),
            "human_contact": ("human_keyframes", "contact_intervals"),
            "object_contact": ("object_waypoints", "contact_intervals"),
        }
        for subpattern in combinations[pattern]:
            _sample_pattern(masks, b, frames, subpattern, generator)
    else:
        raise ValueError(f"unknown control pattern: {pattern!r}")


def sample_controls(
    states: Mapping[str, Tensor],
    valid_frames: Tensor,
    mode: str | None = None,
    generator: torch.Generator | None = None,
    holdout_signatures: Sequence[str] = (),
    allowed_patterns: Sequence[str] | None = None,
    *,
    max_attempts: int = 128,
) -> ControlBatch:
    """Sample per-example masks, rejecting held-out *actual* signatures.

    A generated-bit mode orders H/O/C and uses 1=generate, 0=fully observed.
    With mode=None the default pool is all seven modes plus partial patterns.
    allowed_patterns replaces that pool with an explicit whitelist (it may
    contain bit modes). An explicit mode must also belong to the whitelist if
    one is supplied. Rejection never silently falls back to a different mode.
    """
    controls = ControlBatch.empty_like(states, valid_frames)
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    all_patterns = GENERATED_MODES + PARTIAL_PATTERNS
    pool = tuple(all_patterns if allowed_patterns is None else allowed_patterns)
    if not pool or any(pattern not in all_patterns for pattern in pool):
        raise ValueError("allowed_patterns must be a nonempty list of supported patterns")
    if mode is not None:
        if mode not in all_patterns or (allowed_patterns is not None and mode not in pool):
            raise ValueError(f"mode {mode!r} is unsupported or excluded by allowed_patterns")
        pool = (mode,)
    excluded = {canonical_signature(signature) for signature in holdout_signatures}
    for b in range(valid_frames.shape[0]):
        frames = valid_frames[b].nonzero(as_tuple=False).flatten().tolist()
        if not frames:
            raise ValueError(f"sample {b} has no valid frames")
        for _ in range(max_attempts):
            for name in MODALITIES:
                controls.masks[name][b].zero_()
            pattern = pool[_randint(len(pool), generator)]
            _sample_pattern(controls.masks, b, frames, pattern, generator)
            actual = control_signatures({name: mask[b:b + 1] for name, mask in controls.masks.items()})[0]
            has_target = any(not mask[b, frames].all().item() for mask in controls.masks.values())
            if actual not in excluded and has_target:
                break
        else:
            raise ValueError(
                f"cannot sample controls for sample {b} after {max_attempts} attempts; "
                f"allowed patterns={pool}, held-out signatures={sorted(excluded)}; "
                "a valid sample must retain at least one unobserved prediction target"
            )
    controls.values = {
        name: torch.where(controls.masks[name], states[name], torch.zeros_like(states[name]))
        for name in MODALITIES
    }
    controls.signatures = None
    return controls.validate()


def _indices(value: Any, limit: int, label: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a nonempty list of indices")
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < limit for i in value):
        raise ValueError(f"{label} indices must be integers in [0, {limit})")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicate indices")
    return list(value)


def compile_controls(
    spec: Mapping[str, Any], reference: Mapping[str, Tensor], valid_frames: Tensor,
) -> ControlBatch:
    """Compile explicit JSON-style entries into hard feature anchors for B=1.

    Reference *values* are read only for entries declaring from_reference=true.
    Overlapping equal anchors are accepted; contradictory ones raise an error.
    """
    controls = ControlBatch.empty_like(reference, valid_frames)
    if valid_frames.shape[0] != 1:
        raise ValueError("compile_controls expects a single example (B=1)")
    if not isinstance(spec, Mapping) or not isinstance(spec.get("entries", []), list):
        raise ValueError("control specification must contain an entries list")
    length, njoints = valid_frames.shape[1], reference["human"].shape[2]
    for number, entry in enumerate(spec.get("entries", [])):
        if not isinstance(entry, Mapping):
            raise ValueError(f"entry {number} must be an object")
        name = entry.get("modality")
        if name not in MODALITIES:
            raise ValueError(f"entry {number}: invalid modality {name!r}")
        if "frames" in entry:
            if "start" in entry or "end" in entry:
                raise ValueError("specify frames or inclusive start/end, not both")
            frames = _indices(entry["frames"], length, "frames")
        else:
            start, end = entry.get("start"), entry.get("end")
            if any(isinstance(i, bool) or not isinstance(i, int) for i in (start, end)):
                raise ValueError("entries require frames or integer start/end")
            if start > end or start < 0 or end >= length:
                raise ValueError("invalid inclusive start/end interval")
            frames = list(range(start, end + 1))
        if not valid_frames[0, frames].all():
            raise ValueError(f"entry {number} anchors an invalid/padded frame")
        if name == "object":
            if "joints" in entry:
                raise ValueError("object entries do not have joints")
            axes = [frames, _indices(entry.get("features", list(range(9))), 9, "features")]
        else:
            joints = _indices(entry.get("joints", list(range(njoints))), njoints, "joints")
            axes = [frames, joints]
            if name == "human":
                axes.append(_indices(entry.get("features", list(range(9))), 9, "features"))
            elif "features" in entry and entry["features"] != [0]:
                raise ValueError("contact is scalar per joint; omit features or use [0]")
        grid = torch.meshgrid(
            *(torch.tensor(axis, dtype=torch.long, device=valid_frames.device) for axis in axes),
            indexing="ij",
        )
        selector = (0, *grid)
        shape = tuple(len(axis) for axis in axes)
        from_reference = entry.get("from_reference", False)
        if not isinstance(from_reference, bool):
            raise ValueError("from_reference must be a JSON boolean")
        if from_reference and "values" in entry:
            raise ValueError("provide values or from_reference=true, not both")
        if from_reference:
            values = reference[name][selector]
        elif "values" in entry:
            try:
                # Validate contact numbers before casting: a bool reference must
                # not silently turn illegal 0.5 or 2 values into valid True.
                value_dtype = torch.float64 if name == "contact" else reference[name].dtype
                values = torch.as_tensor(entry["values"], device=valid_frames.device, dtype=value_dtype)
                if name == "human" and len(axes[1]) == 1 and values.shape == (len(frames), len(axes[2])):
                    values = values.unsqueeze(1)
                values = torch.broadcast_to(values, shape)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(f"entry {number} values must broadcast to selection shape {shape}") from error
        else:
            raise ValueError(f"entry {number} requires explicit values or from_reference=true")
        if not torch.isfinite(values).all():
            raise ValueError(f"entry {number} has nonfinite anchor values")
        if name == "contact" and not ((values == 0) | (values == 1)).all():
            raise ValueError("contact anchors must be exactly 0 or 1")
        values = values.to(reference[name].dtype)
        previous = controls.values[name][selector]
        already = controls.masks[name][selector]
        equal = torch.isclose(previous, values, rtol=0.0, atol=1e-6)
        if (already & ~equal).any():
            raise ValueError(f"entry {number} conflicts with an existing {name} anchor")
        controls.values[name][selector] = values
        controls.masks[name][selector] = True
    controls.signatures = None
    return controls.validate()


def check_anchor_feasibility(
    controls: ControlBatch, object_points: Tensor, *, contact_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Report contradictions against supplied sampled geometry, not physics.

    Only observed joint xyz + full object pose + observed contact are checked.
    Distances to a point cloud depend on sampling resolution: a reported mismatch
    is a diagnostic, not proof that the true mesh or physical task is infeasible.
    """
    controls.validate()
    batch_size = controls.valid_frames.shape[0]
    if (object_points.ndim != 3 or object_points.shape[0] != batch_size
            or object_points.shape[-1] != 3 or object_points.shape[1] == 0):
        raise ValueError("object_points must have shape [B, K, 3] with K > 0")
    if contact_tolerance <= 0 or not torch.isfinite(torch.tensor(contact_tolerance)):
        raise ValueError("contact_tolerance must be finite and positive")
    if object_points.device != controls.valid_frames.device or not torch.isfinite(object_points).all():
        raise ValueError("object_points must be finite and on the controls device")
    report: dict[str, Any] = {
        "checked_anchors": 0, "conflicts": [], "status": "unchecked",
        "establishes_physical_feasibility": False,
        "limitations": "Point-cloud geometry check only; no mesh, FK, force, balance or reachability guarantee.",
    }
    for b in range(batch_size):
        for frame in controls.valid_frames[b].nonzero(as_tuple=False).flatten().tolist():
            if not controls.masks["object"][b, frame].all():
                continue
            obj = controls.values["object"][b, frame]
            first, second = obj[3:6], obj[6:9]
            first_norm = torch.linalg.vector_norm(first)
            if first_norm <= 1e-8:
                report["conflicts"].append({"kind": "degenerate_object_rotation", "batch": b, "frame": frame})
                continue
            row0 = first / first_norm
            orthogonal = second - torch.dot(row0, second) * row0
            second_norm = torch.linalg.vector_norm(orthogonal)
            if second_norm <= 1e-8:
                report["conflicts"].append({"kind": "degenerate_object_rotation", "batch": b, "frame": frame})
                continue
            row1 = orthogonal / second_norm
            rotation = torch.stack((row0, row1, torch.linalg.cross(row0, row1)))
            points = object_points[b].to(obj.dtype) @ rotation.T + obj[:3]
            for joint in range(controls.values["human"].shape[2]):
                if not (controls.masks["human"][b, frame, joint, :3].all()
                        and controls.masks["contact"][b, frame, joint]):
                    continue
                position = controls.values["human"][b, frame, joint, :3]
                distance = float(torch.linalg.vector_norm(points - position, dim=-1).min().item())
                expected = bool(controls.values["contact"][b, frame, joint].item())
                report["checked_anchors"] += 1
                if expected != (distance < contact_tolerance):
                    report["conflicts"].append({
                        "kind": "contact_geometry_mismatch", "batch": b, "frame": frame,
                        "joint": joint, "contact": int(expected), "distance_m": distance,
                    })
    if report["conflicts"]:
        report["status"] = "geometry_mismatch"
    elif report["checked_anchors"]:
        report["status"] = "no_checked_conflict"
    return report
