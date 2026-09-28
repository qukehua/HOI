import json

import numpy as np
import pytest
import torch

from unified_hoi.controls import ControlBatch, compile_controls
from unified_hoi.evaluate import evaluate_batch, main


def example(batch=1, length=5, joints=2):
    human = torch.zeros(batch, length, joints, 9)
    obj = torch.zeros(batch, length, 9)
    human[..., 3] = obj[..., 3] = 1
    human[..., 7] = obj[..., 7] = 1
    human[:, :, 1:, 0] = 1
    contact = torch.zeros(batch, length, joints)
    valid = torch.ones(batch, length, dtype=torch.bool)
    offsets = torch.zeros(batch, joints, 3)
    offsets[:, 1:, 0] = 1
    states = {"human": human, "object": obj, "contact": contact}
    metadata = {
        "object_points": torch.zeros(batch, 1, 3), "rest_offsets": offsets,
        "timestamps": torch.arange(length, dtype=torch.float64)[None].repeat(batch, 1) * 0.5,
        "valid_frames": valid,
    }
    return states, valid, metadata


def test_empty_contact_support_is_none_even_when_predictions_say_contact():
    states, valid, batch = example()
    states["contact"].fill_(1)
    controls = ControlBatch.empty_like(states, valid)
    controls.signatures = None
    report = evaluate_batch(states, controls, batch)
    assert controls.signatures is None
    assert report["anchor_feature_mae"] is None
    assert report["contact_positive_distance_m"] is None
    assert report["contact_negative_violation_rate"] is None
    assert report["contact_anchor_satisfaction_rate"] is None
    assert report["contact_slip_mean_m_s"] is None
    assert report["contact_anchor_count"] == 0
    assert report["human_fk_error_m"] == 0
    assert report["physical_execution_evaluated"] is False
    assert report["smoothness_is_realism_metric"] is False
    json.dumps(report, allow_nan=False)


def test_position_partial_components_rotation_and_raw_feature_metrics():
    states, valid, batch = example()
    controls = compile_controls({"entries": [
        {"modality": "human", "frames": [0], "joints": [0], "features": [0, 1, 2], "values": [3, 4, 0]},
        {"modality": "object", "frames": [0], "features": [0], "values": 2},
        {"modality": "human", "frames": [0], "joints": [1], "features": [3, 4, 5, 6, 7, 8], "values": [0, -1, 0, 1, 0, 0]},
    ]}, states, valid)
    report = evaluate_batch(states, controls, batch)
    assert report["human_anchor_position_error_m"] == 5
    assert report["human_anchor_rotation_error_deg"] == pytest.approx(90)
    assert report["object_anchor_position_error_m"] is None
    assert report["object_anchor_position_component_mae_m"] == 2
    assert report["object_anchor_feature_mae"] == 2
    assert report["anchor_feature_count"] == 10


def test_degenerate_rotation_cannot_look_like_a_correct_identity_anchor():
    states, valid, batch = example()
    controls = compile_controls({"entries": [
        {"modality": "object", "frames": [0], "features": [3, 4, 5, 6, 7, 8], "from_reference": True},
    ]}, states, valid)
    states["object"][0, 0, 3:] = 0
    report = evaluate_batch(states, controls, batch)
    assert report["object_anchor_rotation_error_deg"] == 180
    assert report["object_degenerate_rotation_rate"] == pytest.approx(0.2)


def test_positive_negative_contact_geometry_and_values_are_separate():
    states, valid, batch = example()
    controls = compile_controls({"entries": [
        {"modality": "contact", "frames": [0], "joints": [0, 1], "values": [1, 0]},
    ]}, states, valid)
    report = evaluate_batch(states, controls, batch)
    assert report["contact_positive_distance_m"] == 0
    assert report["contact_negative_violation_rate"] == 0
    assert report["contact_anchor_satisfaction_rate"] == 1
    assert report["contact_positive_and_negative_all_satisfied_rate"] == 1
    assert report["contact_anchor_value_accuracy"] == 0.5
    states["human"][0, 0, 1, :3] = 0
    report = evaluate_batch(states, controls, batch)
    assert report["contact_negative_violation_rate"] == 1
    assert report["contact_anchor_satisfaction_rate"] == 0.5
    assert report["contact_all_anchors_satisfied_rate"] == 0
    assert report["contact_positive_and_negative_all_satisfied_rate"] == 0


def test_padding_nonfinite_values_are_ignored_in_all_metrics():
    states, valid, batch = example(length=6)
    valid[:, -2:] = False
    controls = ControlBatch.empty_like(states, valid)
    baseline = evaluate_batch(states, controls, batch)
    for value in states.values():
        value[:, -2:] = float("nan")
    batch["timestamps"][:, -2:] = float("nan")
    after = evaluate_batch(states, controls, batch)
    assert after == baseline
    assert after["valid_frame_count"] == 4
    assert after["human_speed_sample_count"] == 6
    assert after["human_acceleration_sample_count"] == 4
    assert after["human_jerk_sample_count"] == 2


def test_object_relative_slip_ignores_rigid_transport_and_uses_actual_dt():
    states, valid, batch = example()
    stamps = torch.tensor([[0.0, 0.1, 0.3, 0.7, 1.0]], dtype=torch.float64)
    batch["timestamps"] = stamps
    translation = stamps.float()[..., None]
    states["object"][..., 0] = stamps
    states["human"][..., 0] += translation
    controls = compile_controls({"entries": [
        {"modality": "contact", "start": 0, "end": 4, "joints": [0], "values": 1},
    ]}, states, valid)
    report = evaluate_batch(states, controls, batch)
    assert report["contact_slip_mean_m_s"] == 0
    assert report["human_speed_mean_m_s"] == pytest.approx(1)
    states["human"][:, :, 0, 0] += 2 * stamps
    report = evaluate_batch(states, controls, batch)
    assert report["contact_slip_mean_m_s"] == pytest.approx(2)
    assert report["contact_slip_pair_count"] == 4


def test_slip_does_not_bridge_unknown_contact_frames_or_invalid_frames():
    states, valid, batch = example(length=5)
    controls = compile_controls({"entries": [
        {"modality": "contact", "frames": [0, 2, 4], "joints": [0], "values": 1},
    ]}, states, valid)
    report = evaluate_batch(states, controls, batch)
    assert report["contact_slip_mean_m_s"] is None
    assert report["contact_slip_pair_count"] == 0
    valid[:, 2] = False
    controls = ControlBatch.empty_like(states, valid)
    report = evaluate_batch(states, controls, batch)
    assert report["human_speed_sample_count"] == 4
    assert report["human_acceleration_mean_m_s2"] is None


def test_fk_and_floor_are_nontrivial_proxies():
    states, valid, batch = example()
    states["human"][..., 1, 0] = 2
    states["human"][..., 0, 1] = -0.1
    states["human"][..., 1, 1] = -0.1
    states["object"][..., 1] = -0.2
    report = evaluate_batch(states, ControlBatch.empty_like(states, valid), batch)
    assert report["human_fk_error_m"] == 1
    assert report["human_floor_below_rate"] == 1
    assert report["human_floor_depth_mean_m"] == pytest.approx(0.1)
    assert report["object_floor_depth_max_m"] == pytest.approx(0.2)


def test_derivatives_use_midpoint_times_and_do_not_claim_realism():
    states, valid, batch = example(length=5)
    t = batch["timestamps"]
    states["object"][..., 0] = 2 * t.square()
    report = evaluate_batch(states, ControlBatch.empty_like(states, valid), batch)
    assert report["object_acceleration_mean_m_s2"] == pytest.approx(4)
    assert report["object_jerk_mean_m_s3"] == pytest.approx(0)
    assert report["smoothness_is_realism_metric"] is False


def test_missing_timing_yields_none_instead_of_guessed_fps():
    states, valid, batch = example()
    batch.pop("timestamps")
    report = evaluate_batch(states, ControlBatch.empty_like(states, valid), batch)
    assert report["timing_source"] == "unavailable"
    assert report["human_speed_mean_m_s"] is None
    batch["fps"] = 10
    states["object"][0, :, 0] = torch.arange(5) * 0.1
    report = evaluate_batch(states, ControlBatch.empty_like(states, valid), batch)
    assert report["timing_source"] == "fps"
    assert report["object_speed_mean_m_s"] == pytest.approx(1)


def test_batch_sequences_without_contact_do_not_inflate_joint_success():
    states, valid, batch = example(batch=2)
    controls = ControlBatch.empty_like(states, valid)
    controls.masks["contact"][0, 0, 1] = True
    controls.values["contact"][0, 0, 1] = 1
    controls.signatures = None
    report = evaluate_batch(states, controls, batch)
    assert report["contact_evaluated_sequence_count"] == 1
    assert report["contact_all_anchors_satisfied_rate"] == 0
    assert report["contact_positive_and_negative_all_satisfied_rate"] is None


@pytest.mark.parametrize("failure", ["timestamp", "nonfinite", "geometry"])
def test_invalid_valid_data_is_rejected(failure):
    states, valid, batch = example()
    controls = ControlBatch.empty_like(states, valid)
    if failure == "timestamp":
        batch["timestamps"][0, 1] = 0
    elif failure == "nonfinite":
        states["human"][0, 0, 0, 0] = float("nan")
    else:
        batch["object_points"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        evaluate_batch(states, controls, batch)


def _archives(tmp_path):
    states, valid, batch = example(length=5, joints=22)
    record = {name: value[0].numpy() for name, value in states.items()}
    record.update(object_points=batch["object_points"][0].numpy(), rest_offsets=batch["rest_offsets"][0].numpy(),
                  fps=np.array(2.0), timestamps=batch["timestamps"][0].numpy(), sequence_id=np.array("seq"),
                  dataset=np.array("test"), subject_id=np.array("subject"), object_id=np.array("box"), text=np.array(""))
    reference = tmp_path / "reference.npz"
    prediction = tmp_path / "prediction.npz"
    constraints = tmp_path / "controls.json"
    np.savez(reference, **record)
    np.savez(prediction, **{key: record[key] for key in (*states.keys(), "fps", "timestamps")})
    constraints.write_text(json.dumps({"entries": [{"modality": "object", "frames": [0], "from_reference": True}]}))
    return record, reference, prediction, constraints


def test_cli_loads_static_geometry_and_emits_strict_json(tmp_path, capsys):
    _, reference, prediction, constraints = _archives(tmp_path)
    output = tmp_path / "result.json"
    main(["--prediction", str(prediction), "--reference", str(reference), "--controls", str(constraints), "--output", str(output)])
    report = json.loads(capsys.readouterr().out)
    assert report["object_anchor_feature_mae"] == 0
    assert report["contact_anchor_satisfaction_rate"] is None
    assert json.loads(output.read_text()) == report


def test_cli_rejects_temporally_misaligned_predictions(tmp_path):
    record, reference, prediction, constraints = _archives(tmp_path)
    record["timestamps"] = record["timestamps"] * 2
    np.savez(prediction, **{key: record[key] for key in ("human", "object", "contact", "fps", "timestamps")})
    with pytest.raises(ValueError, match="timestamps disagree"):
        main(["--prediction", str(prediction), "--reference", str(reference), "--controls", str(constraints)])


def test_cli_defaults_to_saved_actual_anchors_not_reference_contacts(tmp_path, capsys):
    record, reference, prediction, _ = _archives(tmp_path)
    stored = {key: record[key] for key in ("human", "object", "contact", "fps", "timestamps")}
    for name in ("human", "object", "contact"):
        stored[f"{name}_observed"] = np.zeros_like(record[name])
        stored[f"{name}_mask"] = np.zeros_like(record[name], dtype=bool)
    stored["contact_observed"][0, 0] = 1
    stored["contact_mask"][0, 0] = True
    np.savez(prediction, **stored)
    main(["--prediction", str(prediction), "--reference", str(reference)])
    report = json.loads(capsys.readouterr().out)
    assert report["control_source"] == "saved_prediction_anchors"
    assert report["contact_positive_anchor_count"] == 1
    assert report["contact_anchor_count"] == 1
    assert report["contact_positive_distance_m"] == 0


def test_cli_without_any_saved_controls_does_not_assume_reference_is_observed(tmp_path, capsys):
    _, reference, prediction, _ = _archives(tmp_path)
    main(["--prediction", str(prediction), "--reference", str(reference)])
    report = json.loads(capsys.readouterr().out)
    assert report["control_source"] == "empty_no_saved_anchors"
    assert report["anchor_feature_count"] == 0
    assert report["contact_all_anchors_satisfied_rate"] is None


def test_cli_incomplete_saved_controls_raise_instead_of_silent_omission(tmp_path):
    record, reference, prediction, _ = _archives(tmp_path)
    stored = {key: record[key] for key in ("human", "object", "contact", "fps", "timestamps")}
    stored["human_mask"] = np.ones_like(record["human"], dtype=bool)
    np.savez(prediction, **stored)
    with pytest.raises(ValueError, match="incomplete stored controls"):
        main(["--prediction", str(prediction), "--reference", str(reference)])
