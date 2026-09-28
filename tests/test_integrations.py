"""Official-output adapter contracts, without network access or model weights.

The inference test substitutes the expensive model, not a pretrained result.
Its callable signature and output keys are checked against the pinned source.
"""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unified_hoi.integrations import kimodo as bridge
from unified_hoi.integrations.upstream import UPSTREAMS
from unified_hoi.geometry import SMPL_PARENTS, forward_kinematics, rotation6d_to_matrix


def fixture_offsets(joints=22):
    offsets = np.zeros((joints, 3), dtype=np.float32)
    for joint in range(1, joints):
        offsets[joint] = [.03 * (joint % 3 - 1), .05 + .004 * joint, .01]
    return offsets


def official_output(frames=3, joints=22, identity=False):
    # Articulated FK-consistent fixture, including time-varying world root and
    # parent rotations. Identity alone cannot detect row/column mistakes.
    rotations = np.tile(np.eye(3, dtype=np.float32), (frames, joints, 1, 1))
    if not identity:
        angles = np.pi / 2 + .07 * np.arange(frames)[:, None] + .02 * np.arange(joints)[None]
        rotations[..., 0, 0] = rotations[..., 1, 1] = np.cos(angles)
        rotations[..., 0, 1] = -np.sin(angles)
        rotations[..., 1, 0] = np.sin(angles)
    positions = np.zeros((frames, joints, 3), dtype=np.float32)
    positions[:, 0] = np.stack([.02 * np.arange(frames), np.ones(frames), .01 * np.arange(frames)], -1)
    local_rotations = rotations.copy()
    offsets = fixture_offsets(joints)
    for joint, parent in enumerate(SMPL_PARENTS[1:joints], start=1):
        positions[:, joint] = positions[:, parent] + np.einsum("tik,k->ti", rotations[:, parent], offsets[joint])
        local_rotations[:, joint] = np.swapaxes(rotations[:, parent], -1, -2) @ rotations[:, joint]
    return {
        "posed_joints": positions,
        "global_rot_mats": rotations,
        "local_rot_mats": local_rotations,
        "root_positions": positions[:, 0].copy(),
        "smooth_root_pos": np.zeros((frames, 3), dtype=np.float32),
        "foot_contacts": np.zeros((frames, 4), dtype=bool),
        "global_root_heading": np.tile(np.array([1., 0], dtype=np.float32), (frames, 1)),
    }


def write_source(tmp_path, *, skeleton="smplx22", fps=24., identity=False, motion=None, sidecar=True):
    path = tmp_path / "official_motion.npz"
    motion = official_output(identity=identity) if motion is None else motion
    np.savez(path, **motion)
    if sidecar:
        path.with_suffix(".meta.json").write_text(json.dumps({
            "skeleton": skeleton, "fps": fps,
            "upstream_commit": UPSTREAMS["kimodo"]["commit"],
            "model": "kimodo-smplx-rp", "fixture_only": True,
        }), encoding="utf-8")
    return path, motion


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_bridge_preserves_world_positions_fps_and_global_rotation_rows(tmp_path, identity, batched):
    motion = official_output(identity=identity)
    if batched:
        motion = {key: value[None] for key, value in motion.items()}
    source, _ = write_source(tmp_path, motion=motion, fps=24.)
    source_bytes = source.read_bytes()
    result = bridge.convert_human_reference(source, tmp_path / "reference.npz")
    with np.load(result["reference"], allow_pickle=False) as converted:
        assert set(converted.files) == {"human", "human_mask", "fps", "source_fps", "timestamps", "rest_offsets"}
        human = converted["human"]
        assert human.shape == (3, 22, 9)
        assert human.dtype == np.float32
        assert converted["human_mask"].dtype == np.bool_ and converted["human_mask"].all()
        assert converted["fps"].shape == () and float(converted["fps"]) == 24.
        assert converted["source_fps"].shape == () and float(converted["source_fps"]) == 24.
        np.testing.assert_array_equal(converted["timestamps"], np.arange(3) / 24.)
        np.testing.assert_allclose(converted["rest_offsets"], fixture_offsets(), atol=1e-7)
        assert converted["rest_offsets"].shape == (22, 3) and converted["rest_offsets"].dtype == np.float32
        original = official_output(identity=identity)
        np.testing.assert_array_equal(human[..., :3], original["posed_joints"])
        expected = original["global_rot_mats"][..., :2, :].reshape(3, 22, 6)
        np.testing.assert_array_equal(human[..., 3:], expected)
        if not identity:
            column_features = np.concatenate((original["global_rot_mats"][..., 0],
                                              original["global_rot_mats"][..., 1]), -1)
            assert not np.array_equal(human[..., 3:], column_features)
            assert not np.array_equal(human[..., 3:], original["local_rot_mats"][..., :2, :].reshape(3, 22, 6))
    meta = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
    assert meta["source"] == str(source.resolve())
    assert meta["source_metadata"]["upstream_commit"] == UPSTREAMS["kimodo"]["commit"]
    assert meta["not_a_paired_training_sample"] is True
    assert meta["root_semantics"].startswith("joint 0 is world pelvis")
    assert source.read_bytes() == source_bytes


def test_bridge_downsamples_actual_frames_times_and_preserves_source_skeleton(tmp_path):
    source, motion = write_source(tmp_path, fps=30., motion=official_output(frames=8))
    result = bridge.convert_human_reference(source, tmp_path / "reference.npz", fps=30., target_fps=10.)
    with np.load(result["reference"]) as converted:
        assert converted["human"].shape == (3, 22, 9)
        assert float(converted["fps"]) == 10 and float(converted["source_fps"]) == 30
        np.testing.assert_array_equal(converted["human"][..., :3], motion["posed_joints"][[0, 3, 6]])
        np.testing.assert_array_equal(converted["human"][..., 3:], motion["global_rot_mats"][[0, 3, 6], ..., :2, :].reshape(3, 22, 6))
        np.testing.assert_array_equal(converted["timestamps"], np.array([0., .1, .2]))
        np.testing.assert_allclose(converted["rest_offsets"], fixture_offsets(), atol=1e-7)
        # Verify the actual downstream representation reconstructs the anchors.
        human = torch.from_numpy(converted["human"])
        reconstructed = forward_kinematics(rotation6d_to_matrix(human[..., 3:]),
                                           torch.from_numpy(converted["rest_offsets"]), human[:, 0, :3])
        torch.testing.assert_close(reconstructed, human[..., :3], atol=2e-7, rtol=0)
    meta = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
    assert meta["schema"] == "unified-hoi-human-reference-v2"
    assert (meta["source_num_frames"], meta["num_frames"], meta["downsample_stride"]) == (8, 3, 3)
    assert meta["skeleton_tolerance_m"] == 1e-4
    assert max(meta["max_fk_error_m"], meta["max_offset_error_m"], meta["max_bone_length_error_m"]) < 1e-6


@pytest.mark.parametrize("target_fps, error", [
    (60., "upsampling"), (12., "integer stride"), (29.97, "integer stride"),
    (0., "positive"), (-10., "positive"), (float("nan"), "positive"), (float("inf"), "positive"),
])
def test_bridge_rejects_unsupported_target_fps(tmp_path, target_fps, error):
    source, _ = write_source(tmp_path, fps=30.)
    with pytest.raises(ValueError, match=error):
        bridge.convert_human_reference(source, tmp_path / "reference.npz", target_fps=target_fps)
    assert not (tmp_path / "reference.npz").exists()


def test_bridge_source_fps_cannot_relabel_existing_metadata(tmp_path):
    source, _ = write_source(tmp_path, fps=30.)
    with pytest.raises(ValueError, match="conflicts.*target_fps"):
        bridge.convert_human_reference(source, tmp_path / "reference.npz", fps=10.)


@pytest.mark.parametrize("inconsistency", ["offset", "rotation_only"])
def test_bridge_rejects_inconsistent_skeleton_even_in_discarded_frame(tmp_path, inconsistency):
    motion = official_output(frames=8)
    if inconsistency == "offset":
        motion["posed_joints"][1, 20, 0] += .01
    else:
        # Positions and bone lengths remain identical, but the required local
        # offsets change when a parent rotation is incompatible with the joints.
        motion["global_rot_mats"][1, 18] = np.eye(3)
    source, _ = write_source(tmp_path, fps=30., motion=motion)
    with pytest.raises(ValueError, match="inconsistent static skeleton"):
        bridge.convert_human_reference(source, tmp_path / "reference.npz", target_fps=10.)
    assert not (tmp_path / "reference.npz").exists()


def test_convert_cli_routes_target_fps_to_real_conversion(tmp_path, capsys):
    source, _ = write_source(tmp_path, fps=30., motion=official_output(frames=8))
    target = tmp_path / "reference.npz"
    assert bridge.main(["convert", "--input", str(source), "--output", str(target), "--target-fps", "10"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["frames"] == 3 and Path(report["reference"]) == target
    with np.load(target) as converted:
        assert float(converted["fps"]) == 10


@pytest.mark.parametrize("skeleton", ["somaskel77", "somaskel30", "g1skel34", "unknown"])
def test_bridge_rejects_unmapped_skeleton_even_when_array_has_22_joints(tmp_path, skeleton):
    source, _ = write_source(tmp_path, skeleton=skeleton)
    with pytest.raises(ValueError, match="smplx22"):
        bridge.convert_human_reference(source, tmp_path / "reference.npz")
    with pytest.raises(ValueError, match="conflicts"):
        bridge.convert_human_reference(source, tmp_path / "reference.npz", skeleton="smplx22")


@pytest.mark.parametrize("kind", ["zero", "reflection", "scaled", "nan_rotation", "nan_position"])
def test_bridge_rejects_corrupt_rotation_or_position(tmp_path, kind):
    motion = official_output()
    if kind == "zero":
        motion["global_rot_mats"][0, 0] = 0
    elif kind == "reflection":
        motion["global_rot_mats"][0, 0] = np.diag([-1., 1, 1])
    elif kind == "scaled":
        motion["global_rot_mats"][0, 0] *= 2
    elif kind == "nan_rotation":
        motion["global_rot_mats"][0, 0, 0, 0] = np.nan
    else:
        motion["posed_joints"][0, 0, 0] = np.nan
    source, _ = write_source(tmp_path, motion=motion)
    with pytest.raises(ValueError, match="nonfinite|rotation"):
        bridge.convert_human_reference(source, tmp_path / "reference.npz")
    assert not (tmp_path / "reference.npz").exists()


@pytest.mark.parametrize("fps", [0., -1., float("nan"), float("inf")])
def test_bridge_rejects_invalid_fps_without_resampling(tmp_path, fps):
    source, _ = write_source(tmp_path)
    with pytest.raises(ValueError, match="positive fps"):
        bridge.convert_human_reference(source, tmp_path / "reference.npz", fps=fps)


def test_cli_export_without_sidecar_requires_explicit_provenance(tmp_path):
    source, _ = write_source(tmp_path, sidecar=False)
    with pytest.raises(ValueError, match="smplx22"):
        bridge.convert_human_reference(source, tmp_path / "reference.npz")
    with pytest.raises(ValueError, match="positive fps"):
        bridge.convert_human_reference(source, tmp_path / "reference.npz", skeleton="smplx22")
    result = bridge.convert_human_reference(source, tmp_path / "reference.npz", skeleton="smplx22", fps=60.)
    with np.load(result["reference"]) as converted:
        assert converted["human"].shape[0] == 3
        assert float(converted["fps"]) == 60.


def test_bridge_refuses_multisample_batch_and_wrong_joint_count(tmp_path):
    for motion in (
        {key: np.stack([value, value]) for key, value in official_output().items()},
        official_output(joints=21),
    ):
        source, _ = write_source(tmp_path, motion=motion)
        with pytest.raises(ValueError, match="shape"):
            bridge.convert_human_reference(source, tmp_path / "reference.npz")


def test_bridge_protects_original_and_existing_output(tmp_path):
    source, _ = write_source(tmp_path)
    with pytest.raises(ValueError, match="differ"):
        bridge.convert_human_reference(source, source, overwrite=True)
    target = tmp_path / "reference.npz"
    bridge.convert_human_reference(source, target)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        bridge.convert_human_reference(source, target)
    assert target.read_bytes() == original


def test_uppercase_extension_is_rejected_or_saved_to_reported_path(tmp_path):
    source, _ = write_source(tmp_path)
    target = tmp_path / "reference.NPZ"
    try:
        result = bridge.convert_human_reference(source, target)
    except ValueError as error:
        assert ".npz" in str(error)
        return
    assert Path(result["reference"]).is_file(), "NumPy silently appended .npz to the reported filename"
    with pytest.raises(FileExistsError):
        bridge.convert_human_reference(source, target)


def test_pinned_official_generation_signature_and_export_fields():
    checkout = bridge.default_checkout()
    path = checkout / "kimodo/model/kimodo_model.py"
    if not path.is_file():
        pytest.skip("Optional pinned Kimodo checkout has not been fetched")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Kimodo")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    names = {arg.arg for arg in method.args.args + method.args.kwonlyargs}
    assert {"prompts", "num_frames", "num_denoising_steps", "constraint_lst", "post_processing", "return_numpy"} <= names
    doc = ast.get_docstring(method)
    assert all(key in doc for key in official_output())
    loading = ast.parse((checkout / "kimodo/model/load_model.py").read_text(encoding="utf-8"))
    load = next(node for node in loading.body if isinstance(node, ast.FunctionDef) and node.name == "load_model")
    assert {"modelname", "device", "return_resolved_name", "text_encoder_fp32"} <= {arg.arg for arg in load.args.args}
    definitions = ast.parse((checkout / "kimodo/skeleton/definitions.py").read_text(encoding="utf-8"))
    skeleton = next(node for node in definitions.body if isinstance(node, ast.ClassDef) and node.name == "SMPLXSkeleton22")
    hierarchy = next(ast.literal_eval(node.value) for node in skeleton.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "bone_order_names_with_parents" for target in node.targets))
    joint_indices = {name: index for index, (name, _) in enumerate(hierarchy)}
    assert tuple(-1 if parent is None else joint_indices[parent] for _, parent in hierarchy) == SMPL_PARENTS


def test_generation_adapter_calls_official_api_contract_without_weights(tmp_path, monkeypatch):
    checkout = tmp_path / "kimodo_source"
    source = checkout / "kimodo/model/kimodo_model.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Stub marker for the import context; no model code or weights.\n")
    constraints_path = tmp_path / "constraints.json"
    constraints_path.write_text("[]")
    calls = {}
    skeleton = SimpleNamespace(name="smplx22", bone_order_names=[f"joint_{i}" for i in range(22)])
    sentinel = object()

    class ModelDouble:
        fps = 24.
        def __init__(self):
            self.skeleton = skeleton
            self.output_skeleton = skeleton

        def __call__(self, prompts, num_frames, num_denoising_steps, *, constraint_lst,
                     post_processing=False, return_numpy=False):
            assert not torch.is_grad_enabled()
            assert isinstance(prompts, str) and isinstance(num_frames, int)
            assert constraint_lst == [sentinel]
            assert return_numpy is True and post_processing is False
            calls["generate"] = (prompts, num_frames, num_denoising_steps)
            return official_output(frames=num_frames)

    def load_model(modelname, *, device, return_resolved_name, text_encoder_fp32):
        assert return_resolved_name is True and text_encoder_fp32 is True
        calls["load"] = (modelname, device)
        return ModelDouble(), "kimodo-smplx-rp"

    def load_constraints_lst(path, model_skeleton, device=None):
        assert path == str(constraints_path.resolve())
        assert model_skeleton is skeleton and device == "cpu"
        return [sentinel]

    def save_kimodo_npz(path, motion):
        calls["save_shape"] = motion["posed_joints"].shape
        np.savez(path, **motion)

    modules = {
        "torch": torch,
        "kimodo": SimpleNamespace(load_model=load_model),
        "kimodo.constraints": SimpleNamespace(load_constraints_lst=load_constraints_lst),
        "kimodo.exports.motion_io": SimpleNamespace(save_kimodo_npz=save_kimodo_npz),
        "kimodo.tools": SimpleNamespace(seed_everything=lambda seed: calls.update(seed=seed)),
    }
    import_module = bridge.importlib.import_module
    monkeypatch.setattr(bridge.importlib, "import_module", lambda name: modules[name] if name in modules else import_module(name))
    monkeypatch.delitem(sys.modules, "kimodo", raising=False)
    monkeypatch.setattr(bridge, "checkout_report", lambda *_: {
        "pinned": True, "dirty": False, "actual_commit": UPSTREAMS["kimodo"]["commit"],
    })
    monkeypatch.setenv("TEXT_ENCODER_MODE", "prior-mode")
    monkeypatch.setenv("TEXT_ENCODER_DEVICE", "prior-device")
    result = bridge.generate_human("Test fixture prompt", tmp_path / "generated.npz", duration=.25,
                                   checkout=checkout, constraints_path=constraints_path,
                                   device="cpu", diffusion_steps=7, seed=31)
    assert calls["load"] == (bridge.DEFAULT_MODEL, "cpu")
    assert calls["generate"] == ("Test fixture prompt", 6, 7)
    assert calls["save_shape"] == (6, 22, 3) and calls["seed"] == 31
    meta = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
    assert meta["fps"] == 24 and meta["num_frames"] == 6
    assert meta["joint_names"] == skeleton.bone_order_names
    assert meta["upstream_commit"] == UPSTREAMS["kimodo"]["commit"]
    assert bridge.os.environ["TEXT_ENCODER_MODE"] == "prior-mode"
    assert bridge.os.environ["TEXT_ENCODER_DEVICE"] == "prior-device"
    assert str(checkout.resolve()) not in sys.path
    converted = bridge.convert_human_reference(result["motion"], tmp_path / "reference.npz")
    with np.load(converted["reference"]) as reference:
        assert reference["human"].shape == (6, 22, 9)
        assert reference["fps"] == 24
