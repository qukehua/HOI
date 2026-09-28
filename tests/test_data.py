import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from unified_hoi.data import HOIDataset, read_manifest, validate_record
from unified_hoi.preprocess import (UnsupportedSequence, continuous_records, convert_behave_record,
                                    convert_omomo_record, make_record, _training_holdout)


def record(n=7, fps=10, name="sub1_box_take"):
    position = np.zeros((n, 22, 3))
    position[:, :, 0] = np.arange(n)[:, None] * .1
    return make_record(positions=position, rotations=np.tile(np.eye(3), (n, 22, 1, 1)),
                       object_trans=np.zeros((n, 3)), object_rot=np.tile(np.eye(3), (n, 1, 1)),
                       object_points=np.array([[0., 0, 0], [1., 0, 0]]), rest_offsets=np.zeros((22, 3)),
                       timestamps=np.arange(n) / fps, fps=fps, sequence_id=name, dataset="omomo",
                       subject_id="sub1", object_id="box")


def manifest(tmp_path, rec, **extra):
    path = tmp_path / "sequence.npz"
    np.savez(path, **rec)
    entry = dict(path=path.name, sequence_id=rec["sequence_id"], dataset=rec["dataset"],
                 fps=float(rec["fps"]), split="train", **extra)
    dest = tmp_path / "manifest.jsonl"
    dest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return dest


def test_padding_preserves_time_and_missing_text(tmp_path):
    ds = HOIDataset(manifest(tmp_path, record()), "train", window=4, stride=4)
    assert len(ds) == 2
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch["human"].shape == (2, 4, 22, 9)
    assert batch["valid_frames"].tolist() == [[True] * 4, [True, True, True, False]]
    assert not batch["text_available"].any()
    assert not batch["text_features"].any()
    assert batch["timestamps"][1].tolist() == [.4, .5, .6, .6]


def test_split_leakage_uses_source_sequence(tmp_path):
    entries = [dict(path="unused.npz", sequence_id=f"seq_part{i}", source_sequence_id="seq",
                    dataset="behave", fps=30, split=split) for i, split in enumerate(["train", "test"])]
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(map(json.dumps, entries)))
    with pytest.raises(ValueError, match="leakage"):
        read_manifest(path)


def test_gaps_split_before_downsample_and_no_rate_invention():
    rec = record(n=6)
    rec["timestamps"] = np.array([0., .1, .2, 1., 1.1, 1.2])
    with pytest.raises(ValueError, match="gaps"):
        validate_record(rec)
    parts = list(continuous_records(rec, 5))
    assert len(parts) == 2
    np.testing.assert_allclose(parts[1]["timestamps"], [1., 1.2])
    with pytest.raises(ValueError, match="upsampling"):
        list(continuous_records(record(fps=1), 30))
    low_rate = record(fps=1)
    low_rate["fps"] = 30.
    with pytest.raises(ValueError, match="relabeled"):
        list(continuous_records(low_rate))


def omomo_source(n=3):
    return dict(seq_name="sub1_box_take", trans=np.tile([0., 0., 1.], (n, 1)),
                trans2joint=np.array([0., 0., .1]), rest_offsets=np.zeros((24, 3)),
                root_orient=np.zeros((n, 3)), pose_body=np.zeros((n, 63)),
                obj_trans=np.tile(np.array([0., 0., .9])[None, :, None], (n, 1, 1)),
                obj_rot=np.tile(np.eye(3), (n, 1, 1)), obj_scale=np.full(n, 2.))


def test_omomo_scale_origin_axis_and_contact():
    src = omomo_source()
    src["rest_offsets"][1] = [.2, 0, 0]
    points = np.array([[0., 0., 0.], [1., 0, 0]])
    rec = convert_omomo_record(src, points, source_fps=30)
    validate_record(rec)
    np.testing.assert_allclose(rec["object_points"], points * 2)
    np.testing.assert_allclose(rec["human"][0, 0, :3], [0, .9, 0], atol=1e-6)
    np.testing.assert_allclose(rec["rest_offsets"][1], [.2, 0, 0])
    assert rec["contact"][0, 0] == 1
    # Changing world axes must only LEFT multiply object rotation: local geometry is unchanged.
    matrix = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    np.testing.assert_allclose(rec["object"][0, 3:], matrix[:2].reshape(6), atol=1e-6)
    src["obj_scale"][1] = 3
    with pytest.raises(UnsupportedSequence, match="time-varying"):
        convert_omomo_record(src, points, source_fps=30)
    src["seq_name"] = "sub1_mop_take"
    with pytest.raises(UnsupportedSequence, match="articulated"):
        convert_omomo_record(src, points, source_fps=30)


def test_behave_timestamp_alignment_with_model_fixture():
    class BodyFixture:
        def __call__(self, **kwargs):
            return SimpleNamespace(joints=kwargs["transl"][:, None].repeat(1, 22, 1))
    smpl = dict(frame_times=np.array(["t0.000", "t0.033", "t0.067"]),
                poses=np.zeros((3, 156)), betas=np.zeros((3, 10)),
                trans=np.array([[0., -1, 0], [1., -1, 0], [2., -1, 0]]))
    obj = dict(frame_times=np.array(["t0.033", "t0.067"]), angles=np.zeros((2, 3)),
               trans=np.array([[1., -1, 0], [2., -1, 0]]))
    rec = convert_behave_record(smpl, obj, np.array([[0., 0., 0.]]), sequence_id="Date01_Sub01_box",
                               subject_id="Sub01", object_id="box", gender="male", body_model=BodyFixture())
    validate_record(rec)
    np.testing.assert_allclose(rec["timestamps"], [.033, .067])
    np.testing.assert_allclose(rec["human"][:, 0, :3], [[1, 1, 0], [2, 1, 0]], atol=1e-6)


def test_omomo_scale_jitter_is_bounded_reported_and_optional():
    src = omomo_source()
    src["obj_scale"] = np.array([2., 2.005, 2.01])
    points = np.array([[0., 0., 0.], [1., 0, 0]])
    rec = convert_omomo_record(src, points, source_fps=30, object_radius=2.)
    assert float(rec["source_object_scale"]) == 2.005
    assert float(rec["scale_relative_range"]) == pytest.approx(.01 / 2.005)
    assert float(rec["scale_max_surface_displacement_m"]) == pytest.approx(.01)
    assert rec["scale_displacement_domain"] == "mesh_surface_bound"
    assert not rec["fps_verified_by_metadata"]
    with pytest.raises(UnsupportedSequence, match="time-varying"):
        convert_omomo_record(src, points, source_fps=30, scale_relative_tolerance=0)
    # The reported bound covers world-space annotation perturbation at every frame.
    for scale in src["obj_scale"]:
        assert abs(scale - 2.005) * 2 <= rec["scale_max_surface_displacement_m"] + 1e-12


def test_bad_rotation_is_rejected():
    rec = record()
    rec["object"][0, 3:] = 0
    with pytest.raises(ValueError, match="rotation6d"):
        validate_record(rec)


def test_validation_holdout_never_uses_test_and_is_sequence_stable():
    names = [f"omomo:sub1_box_{i}" for i in range(100)]
    splits = {name: _training_holdout(name, "train", .2, 0) for name in names}
    assert "val" in splits.values() and "train" in splits.values()
    assert {name: _training_holdout(name, "train", .2, 0) for name in names[::-1]} == splits
    assert all(_training_holdout(name, "test", .9, 0) == "test" for name in names)
