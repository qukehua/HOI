import pytest
import torch

from unified_hoi.controls import (
    GENERATED_MODES, PARTIAL_PATTERNS, ControlBatch, canonical_signature,
    check_anchor_feasibility, compile_controls, sample_controls,
)


def example(batch=1, length=6, joints=4):
    human = torch.arange(batch * length * joints * 9, dtype=torch.float32).reshape(batch, length, joints, 9) / 10
    obj = torch.zeros(batch, length, 9)
    obj[..., 3] = 1
    obj[..., 7] = 1
    contact = torch.zeros(batch, length, joints)
    contact[..., 1] = 1
    valid = torch.ones(batch, length, dtype=torch.bool)
    valid[:, -1] = False
    return {"human": human, "object": obj, "contact": contact}, valid


@pytest.mark.parametrize("mode", GENERATED_MODES)
def test_seven_modes_have_generated_bit_semantics(mode):
    states, valid = example(batch=2)
    controls = sample_controls(states, valid, mode=mode)
    for name, bit in zip(("human", "object", "contact"), mode):
        mask = controls.masks[name]
        assert not mask[:, -1].any()
        assert bool(mask[:, :-1].all()) == (bit == "0")
        assert not controls.values[name][~mask].any()
        if bit == "0":
            assert torch.equal(controls.values[name][mask], states[name][mask])


def test_no_contact_unknown_and_contact_are_three_distinct_cases():
    states, valid = example()
    controls = compile_controls({"entries": [
        {"modality": "contact", "frames": [1], "joints": [0, 1], "values": [0, 1]},
    ]}, states, valid)
    assert controls.values["contact"][0, 1].tolist() == [0, 1, 0, 0]
    assert controls.masks["contact"][0, 1].tolist() == [True, True, False, False]
    target = {name: torch.full_like(value, 0.75) for name, value in states.items()}
    output = controls.apply(target)
    assert output["contact"][0, 1].tolist() == [0, 1, 0.75, 0.75]
    assert target["contact"][0, 1, 0] == 0.75


@pytest.mark.parametrize("invalid", [0.5, 2, -1])
def test_bool_reference_does_not_hide_invalid_numeric_contact_constraints(invalid):
    states, valid = example()
    states["contact"] = states["contact"].bool()
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        compile_controls({"entries": [{"modality": "contact", "frames": [0], "values": invalid}]}, states, valid)


def test_partial_sampling_must_leave_a_prediction_target_on_tiny_sequences():
    states, valid = example(length=2, joints=2)
    states = {name: value[:, :1, :1] if name != "object" else value[:, :1] for name, value in states.items()}
    valid = valid[:, :1]
    with pytest.raises(ValueError, match="prediction target"):
        sample_controls(states, valid, mode="mixed", max_attempts=2)


def test_compile_world_xyz_rotation_root_path_and_inclusive_interval():
    states, valid = example()
    controls = compile_controls({"entries": [
        {"modality": "human", "start": 0, "end": 2, "joints": [0], "features": [0, 2], "values": [3, 4]},
        {"modality": "human", "frames": [3], "joints": [2], "features": [3, 4, 5, 6, 7, 8], "values": [1, 0, 0, 0, 1, 0]},
        {"modality": "object", "frames": [4], "features": [0, 1, 2], "values": [8, 9, 10]},
    ]}, states, valid)
    assert controls.signatures == ("H+O",)
    assert controls.values["human"][0, :3, 0, 0].tolist() == [3, 3, 3]
    assert not controls.masks["human"][0, :3, 0, 1].any()
    assert controls.values["object"][0, 4, :3].tolist() == [8, 9, 10]


def test_explicit_values_do_not_depend_on_reference_values():
    states, valid = example()
    altered = {key: value + 100 for key, value in states.items()}
    spec = {"entries": [{"modality": "human", "frames": [1], "joints": [2], "features": [0, 1, 2], "values": [4, 5, 6]}]}
    first = compile_controls(spec, states, valid)
    second = compile_controls(spec, altered, valid)
    assert all(torch.equal(first.values[key], second.values[key]) for key in states)
    with pytest.raises(ValueError, match="explicit values"):
        compile_controls({"entries": [{"modality": "object", "frames": [0]}]}, states, valid)


def test_from_reference_is_explicit_and_unknown_features_are_zero():
    states, valid = example()
    controls = compile_controls({"entries": [{"modality": "human", "frames": [1, 2], "joints": [2], "from_reference": True}]}, states, valid)
    assert torch.equal(controls.values["human"][0, 1:3, 2], states["human"][0, 1:3, 2])
    assert not controls.values["human"][~controls.masks["human"]].any()


@pytest.mark.parametrize("modality,selection,value", [
    ("human", {"joints": [0], "features": [0]}, 2),
    ("object", {"features": [0]}, 2),
    ("contact", {"joints": [0]}, 1),
])
def test_conflicting_duplicate_anchors_rejected(modality, selection, value):
    states, valid = example()
    first = {"modality": modality, "frames": [0], "values": 0, **selection}
    second = {**first, "values": value}
    with pytest.raises(ValueError, match="conflicts"):
        compile_controls({"entries": [first, second]}, states, valid)
    controls = compile_controls({"entries": [first, first]}, states, valid)
    assert controls.masks[modality].any()


@pytest.mark.parametrize("entry", [
    {"modality": "object", "frames": [-1], "values": 0},
    {"modality": "object", "frames": [5], "values": 0},
    {"modality": "object", "frames": [0, 0], "values": 0},
    {"modality": "object", "frames": [0], "values": float("nan")},
    {"modality": "human", "frames": [0], "joints": [4], "values": 0},
    {"modality": "contact", "frames": [0], "values": 0.5},
    {"modality": "object", "start": 3, "end": 2, "values": 0},
    {"modality": "object", "frames": [0], "from_reference": True, "values": 0},
])
def test_invalid_controls_fail_before_inference(entry):
    states, valid = example()
    with pytest.raises(ValueError):
        compile_controls({"entries": [entry]}, states, valid)


@pytest.mark.parametrize("pattern", PARTIAL_PATTERNS)
def test_partial_patterns_are_reproducible_and_do_not_observe_padding(pattern):
    states, valid = example(batch=3)
    a = sample_controls(states, valid, mode=pattern, generator=torch.Generator().manual_seed(8))
    b = sample_controls(states, valid, mode=pattern, generator=torch.Generator().manual_seed(8))
    for key in states:
        assert torch.equal(a.masks[key], b.masks[key])
        assert not a.masks[key][:, -1].any()
    assert all(signature != "none" for signature in a.signatures)


def test_holdout_filters_whole_modality_modes_and_partial_sources():
    states, valid = example(batch=48)
    heldout = ("H+O", "H+O+C")
    controls = sample_controls(states, valid, holdout_signatures=heldout,
                               generator=torch.Generator().manual_seed(2))
    assert not set(controls.signatures).intersection(heldout)
    for mode, signature in (("001", "H+O"), ("mixed", "H+O+C")):
        with pytest.raises(ValueError, match="after 3 attempts"):
            sample_controls(states, valid, mode=mode, holdout_signatures=(signature,), max_attempts=3)


def test_whitelist_and_canonical_signature_cannot_bypass_holdout():
    states, valid = example(batch=2)
    controls = sample_controls(states, valid, allowed_patterns=("111",))
    assert controls.signatures == ("none", "none")
    assert canonical_signature(" O + h ") == "H+O"
    with pytest.raises(ValueError, match="after 2 attempts"):
        sample_controls(states, valid, mode="001", holdout_signatures=("O+H",), max_attempts=2)
    with pytest.raises(ValueError, match="excluded"):
        sample_controls(states, valid, mode="mixed", allowed_patterns=("111",))
    with pytest.raises(ValueError):
        sample_controls(states, valid, mode="000")


def test_validation_to_and_apply_preserve_unknowns_and_extra_fields():
    states, valid = example()
    controls = ControlBatch.empty_like(states, valid).to("cpu")
    states["extra"] = torch.tensor(7)
    result = controls.apply(states)
    assert all(torch.equal(result[key], value) for key, value in states.items())
    controls.masks["contact"][0, -1, 0] = True
    with pytest.raises(ValueError, match="invalid/padded"):
        controls.validate()


def test_geometric_report_uses_observed_anchors_only_and_never_claims_physics():
    states, valid = example()
    base = [
        {"modality": "object", "frames": [0], "values": [0, 0, 0, 1, 0, 0, 0, 1, 0]},
        {"modality": "human", "frames": [0], "joints": [0], "features": [0, 1, 2], "values": [2, 0, 0]},
        {"modality": "contact", "frames": [0], "joints": [0], "values": 1},
    ]
    controls = compile_controls({"entries": base}, states, valid)
    points = torch.tensor([[[0.0, 0.0, 0.0]]])
    report = check_anchor_feasibility(controls, points)
    assert report["checked_anchors"] == 1
    assert report["status"] == "geometry_mismatch"
    assert report["conflicts"][0]["distance_m"] == 2
    assert report["establishes_physical_feasibility"] is False
    without_contact = compile_controls({"entries": base[:2]}, states, valid)
    assert check_anchor_feasibility(without_contact, points)["checked_anchors"] == 0


def test_geometry_respects_object_rotation_and_negative_contact():
    states, valid = example()
    controls = compile_controls({"entries": [
        {"modality": "object", "frames": [0], "values": [3, 0, 0, 0, -1, 0, 1, 0, 0]},
        {"modality": "human", "frames": [0], "joints": [0], "features": [0, 1, 2], "values": [3, 1, 0]},
        {"modality": "contact", "frames": [0], "joints": [0], "values": 0},
    ]}, states, valid)
    report = check_anchor_feasibility(controls, torch.tensor([[[1.0, 0.0, 0.0]]]))
    assert report["conflicts"][0]["distance_m"] == 0
    assert report["conflicts"][0]["contact"] == 0
