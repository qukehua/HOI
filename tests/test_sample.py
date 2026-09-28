import numpy as np
import pytest
import torch

from unified_hoi.controls import ControlBatch
from unified_hoi.sample import add_human_prior, prepare_batch


def empty_controls():
    state = {"human": torch.zeros(1, 8, 22, 9), "object": torch.zeros(1, 8, 9),
             "contact": torch.zeros(1, 8, 22)}
    return ControlBatch.empty_like(state, torch.ones(1, 8, dtype=torch.bool))


def test_prior_carries_body_shape_and_does_not_modify_other_conditions(tmp_path):
    controls = empty_controls()
    controls.values["object"][:, 0, :3] = 3
    controls.masks["object"][:, 0, :3] = True
    controls.signatures = None
    human = np.zeros((8, 22, 9), np.float32)
    human[..., 3] = human[..., 7] = 1
    shape = np.arange(66, dtype=np.float32).reshape(22, 3) / 1000
    prior = tmp_path / "prior.npz"
    np.savez(prior, human=human, human_mask=np.ones_like(human, bool), fps=10., rest_offsets=shape)
    result, fps, rest = add_human_prior(controls, prior)
    assert fps == 10
    assert torch.equal(rest[0], torch.from_numpy(shape))
    assert result.signatures == ("H+O",)
    assert (result.values["object"][:, 0, :3] == 3).all()


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), 0., -1.])
def test_prior_rejects_invalid_rate(tmp_path, fps):
    prior = tmp_path / "prior.npz"
    human = np.zeros((8, 22, 9), np.float32)
    np.savez(prior, human=human, human_mask=np.ones_like(human, bool), fps=fps,
             rest_offsets=np.zeros((22, 3), np.float32))
    with pytest.raises(ValueError, match="FPS"):
        add_human_prior(empty_controls(), prior)


def test_scene_requires_no_ground_truth_motion(tmp_path):
    path = tmp_path / "scene.npz"
    np.savez(path, object_points=np.zeros((8, 3), np.float32),
             rest_offsets=np.zeros((22, 3), np.float32), fps=10.)
    batch, stamps, _ = prepare_batch(path, frames=8, scene_only=True)
    assert batch["human"].shape == (1, 8, 22, 9)
    np.testing.assert_allclose(stamps, np.arange(8) / 10.)
    with pytest.raises(ValueError, match="frames"):
        prepare_batch(path, frames=0, scene_only=True)
