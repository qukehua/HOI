"""Exact restart checks using tiny synthetic canonical records (not research data)."""

from copy import deepcopy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from unified_hoi.runtime import load_model
from unified_hoi.train import run_training


@pytest.fixture(autouse=True)
def restore_cpu_threads():
    old = torch.get_num_threads()
    yield
    torch.set_num_threads(old)


def write_fixture_manifest(directory: Path, sequences=2):
    directory.mkdir()
    entries = []
    for index in range(sequences):
        frames = 3
        human = np.zeros((frames, 22, 9), dtype=np.float32)
        human[..., 1] = 1.0
        human[..., 0] = np.arange(frames)[:, None] * .02 + index * .1
        human[..., 3:] = [1, 0, 0, 0, 1, 0]
        obj = np.zeros((frames, 9), dtype=np.float32)
        obj[..., :3] = [.4 + index * .1, 1.2, .1]
        obj[..., 3:] = [1, 0, 0, 0, 1, 0]
        contact = np.zeros((frames, 22), dtype=np.float32)
        contact[:, 20] = 1
        sequence = f"synthetic_test_sequence_{index}"
        record = {
            "human": human, "object": obj, "contact": contact,
            "object_points": np.asarray(list(itertools.product([-.1, .1], repeat=3)), dtype=np.float32),
            "rest_offsets": np.zeros((22, 3), dtype=np.float32),
            "fps": np.float32(30), "timestamps": np.arange(frames, dtype=np.float64) / 30,
            "dataset": "test_fixture", "sequence_id": sequence,
            "subject_id": "synthetic", "object_id": "analytic_cube",
            "text": "Synthetic fixture for checking deterministic resume.",
        }
        filename = sequence + ".npz"
        np.savez(directory / filename, **record)
        embedding_name = sequence + "_test_features.npy"
        np.save(directory / embedding_name, np.arange(8, dtype=np.float32) / 8)
        entries.append({
            "path": filename, "dataset": "test_fixture", "sequence_id": sequence,
            "split": "train", "fps": 30, "text_features_path": embedding_name,
        })
    path = directory / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    return path


def config_for(manifest, output, *, balanced=True, max_steps=3):
    return {
        "manifest": str(manifest), "output": str(output), "device": "cpu", "seed": 83,
        "cpu_threads": 1, "workers": 0, "window": 3, "stride": 3, "batch_size": 1,
        "balance_datasets": balanced, "max_steps": max_steps, "diffusion_steps": 8,
        "geometry_weight": .001, "learning_rate": 1e-4, "gradient_clip": 1.,
        "ema_decay": .95, "text_dropout": .3, "save_every": 1, "log_every": 1,
        "validate_every": 10000,
        # Nonzero dropout makes restoration of the global torch RNG consequential.
        "model": {"width": 24, "heads": 4, "layers": 1, "dropout": .2, "text_dim": 8},
    }


def assert_nested_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("sequences", [2, 4])
def test_resume_two_to_three_steps_matches_uninterrupted_training(tmp_path, balanced, sequences):
    """Test both an epoch-boundary restart and skipping within an existing epoch."""
    manifest = write_fixture_manifest(tmp_path / "data", sequences=sequences)
    full_config = config_for(manifest, tmp_path / "full", balanced=balanced)
    uninterrupted_path = run_training(full_config)
    partial_config = {**full_config, "output": str(tmp_path / "partial"), "max_steps": 2}
    partial_path = run_training(partial_config)
    resumed_config = {**full_config, "output": str(tmp_path / "resumed")}
    resumed_path = run_training(resumed_config, resume=partial_path)
    uninterrupted = torch.load(uninterrupted_path, map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
    assert uninterrupted["step"] == resumed["step"] == 3
    for field in ("model", "ema", "optimizer", "rng", "epoch", "next_batch"):
        assert_nested_equal(resumed[field], uninterrupted[field])
    full_metrics = [json.loads(line) for line in (tmp_path / "full" / "train.jsonl").read_text().splitlines()]
    resumed_metrics = [json.loads(line) for line in (tmp_path / "resumed" / "train.jsonl").read_text().splitlines()]
    assert resumed_metrics == [full_metrics[-1]]
    # Inference loading must retain the saved normalizer and choose EMA by default.
    loaded, diffusion, checkpoint = load_model(resumed_path)
    assert not loaded.training
    assert diffusion.steps == 8 and checkpoint["step"] == 3
    assert_nested_equal(loaded.state_dict(), resumed["ema"])


def test_resume_rejects_optimizer_trajectory_changes(tmp_path):
    manifest = write_fixture_manifest(tmp_path / "data")
    config = config_for(manifest, tmp_path / "partial", max_steps=2)
    checkpoint = run_training(config)
    for field, changed_value in (("gradient_clip", .05), ("diffusion_steps", 12), ("batch_size", 2)):
        changed = deepcopy(config)
        changed.update(max_steps=3, output=str(tmp_path / ("changed_" + field)))
        changed[field] = changed_value
        with pytest.raises(ValueError, match=field):
            run_training(changed, resume=checkpoint)
