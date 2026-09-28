"""Behavioral checks for conditioning, geometry and temporal diffusion.

Tiny analytic fixtures test contracts, not motion quality or pretrained results.
"""

from dataclasses import replace

import pytest
import torch

from unified_hoi.controls import ControlBatch, compile_controls, sample_controls
from unified_hoi.diffusion import HOIDiffusion, normalized_controls, project_relations
from unified_hoi.geometry import (
    axis_angle_to_matrix,
    forward_kinematics,
    human_in_object_frame,
    matrix_to_rotation6d,
    nearest_surface,
    object_to_world,
    rotation6d_to_matrix,
)
from unified_hoi.model import ModelConfig, UnifiedHOIDenoiser
from unified_hoi.normalization import MODALITIES, StateNormalizer


@pytest.fixture(autouse=True)
def small_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def example_batch(batch_size=2, frames=4):
    offsets = torch.zeros(batch_size, 22, 3)
    offsets[:, 1:, 1] = .05
    root = torch.zeros(batch_size, frames, 3)
    root[..., 1] = 1.0
    root[..., 0] = torch.arange(frames) * .03
    angles = torch.zeros(batch_size, frames, 22, 3)
    angles[..., 1] = .2
    matrices = axis_angle_to_matrix(angles)
    xyz = forward_kinematics(matrices, offsets[:, None], root)
    human = torch.cat((xyz, matrix_to_rotation6d(matrices)), -1)
    obj = torch.zeros(batch_size, frames, 9)
    obj[..., :3] = torch.tensor([.4, 1.2, .1])
    obj[..., 3:] = torch.tensor([1., 0, 0, 0, 1, 0])
    contact = torch.zeros(batch_size, frames, 22)
    contact[:, :, 20] = 1.0
    valid = torch.ones(batch_size, frames, dtype=torch.bool)
    if batch_size > 1 and frames > 1:
        valid[-1, -1] = False
    return {
        "human": human, "object": obj, "contact": contact,
        "valid_frames": valid, "rest_offsets": offsets,
        "object_points": torch.tensor([[[-.1, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]]]).expand(batch_size, -1, -1).clone(),
        "fps": torch.full((batch_size,), 30.),
        "text_features": torch.zeros(batch_size, 8),
    }


def small_model():
    torch.manual_seed(12)
    return UnifiedHOIDenoiser(ModelConfig(width=16, heads=2, layers=1, dropout=0., text_dim=8))


def mixed_controls(batch):
    assert batch["human"].shape[0] == 1
    return compile_controls({"entries": [
        {"modality": "human", "frames": [0], "joints": [0], "from_reference": True},
        {"modality": "human", "frames": [1], "joints": [3], "features": [0], "values": .13},
        {"modality": "object", "frames": [0], "features": [0, 1, 2], "from_reference": True},
        {"modality": "contact", "frames": [0, 1], "joints": [0], "values": 0},
        {"modality": "contact", "frames": [0, 1], "joints": [20], "values": 1},
    ]}, batch, batch["valid_frames"])


def assert_anchors_exact(states, controls):
    for key in MODALITIES:
        torch.testing.assert_close(states[key][controls.masks[key]],
                                   controls.values[key][controls.masks[key]], rtol=0, atol=0)


def assert_free_rotations_orthonormal(states, controls):
    for key in ("human", "object"):
        free = ~controls.masks[key][..., 3:].any(-1)
        valid = controls.valid_frames
        if free.ndim == 3:
            valid = valid[..., None].expand_as(free)
        rows = states[key][..., 3:][free & valid].reshape(-1, 2, 3)
        torch.testing.assert_close(rows.norm(dim=-1), torch.ones(rows.shape[:2]), rtol=0, atol=2e-5)
        torch.testing.assert_close((rows[:, 0] * rows[:, 1]).sum(-1), torch.zeros(len(rows)), rtol=0, atol=2e-5)


def test_rotation_rows_and_zero_axis_angle_gradients():
    aa = torch.tensor([[0., 0, torch.pi / 2], [0., 0, 0]], requires_grad=True)
    matrix = axis_angle_to_matrix(aa)
    six = matrix_to_rotation6d(matrix)
    torch.testing.assert_close(six[0], torch.tensor([0., -1, 0, 1, 0, 0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(rotation6d_to_matrix(six), matrix, atol=1e-6, rtol=0)
    (matrix * torch.arange(9).reshape(3, 3)).sum().backward()
    assert torch.isfinite(aa.grad).all()


def test_degenerate_rotation_predictions_still_produce_so3_and_finite_gradients():
    six = torch.tensor([[0., 0, 0, 0, 0, 0], [1., 0, 0, 2, 0, 0],
                        [.3, .6, -.2, .8, .1, -.4]], requires_grad=True)
    rotation = rotation6d_to_matrix(six)
    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), torch.eye(3).expand(3, -1, -1), atol=1e-6, rtol=0)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(3), atol=1e-6, rtol=0)
    (rotation * torch.arange(9).reshape(3, 3)).sum().backward()
    assert torch.isfinite(six.grad).all()


def test_fk_uses_parent_global_rotation_and_world_root():
    global_rot = axis_angle_to_matrix(torch.tensor([[0., 0, torch.pi / 2],
                                                   [torch.pi / 2, 0, 0], [0., 0, 0]]))
    offsets = torch.tensor([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
    root = torch.tensor([3., 4, 5], requires_grad=True)
    joints = forward_kinematics(global_rot, offsets, root, parents=(-1, 0, 1))
    torch.testing.assert_close(joints, torch.tensor([[3., 4, 5], [3, 5, 5], [3, 5, 6]]), atol=1e-6, rtol=0)
    joints.sum().backward()
    torch.testing.assert_close(root.grad, torch.full((3,), 3.))


def test_object_transform_inverse_and_nearest_point_are_consistent():
    points = torch.tensor([[[1., 0, 0], [0., 1, 0]]])
    obj = torch.tensor([[[3., 4, 5, 0, -1, 0, 1, 0, 0]]], requires_grad=True)
    world = object_to_world(points, obj)
    torch.testing.assert_close(world, torch.tensor([[[[3., 5, 5], [2, 4, 5]]]]))
    torch.testing.assert_close(human_in_object_frame(world, obj), points[:, None])
    distance, near = nearest_surface(world + torch.tensor([0., 0, .1]), obj, points)
    torch.testing.assert_close(distance, torch.full((1, 1, 2), .1), atol=1e-6, rtol=0)
    torch.testing.assert_close(near, points[:, None])
    distance.sum().backward()
    assert torch.isfinite(obj.grad).all()


def test_normalization_ignores_padding_and_roundtrips_states():
    batch = example_batch()
    poisoned = {key: value.clone() for key, value in batch.items()}
    for key in MODALITIES:
        poisoned[key][~batch["valid_frames"]] = 1e5
    a, b = StateNormalizer(), StateNormalizer()
    a.fit([batch])
    b.fit([poisoned])
    for key, tensor in a.state_dict().items():
        torch.testing.assert_close(tensor, b.state_dict()[key], rtol=0, atol=0)
    restored = a.decode(a.encode(batch))
    for key in MODALITIES:
        torch.testing.assert_close(restored[key], batch[key], atol=1e-6, rtol=1e-6)


def test_normalized_controls_preserve_negative_contact_and_mask_semantics():
    batch = example_batch(batch_size=1)
    controls = mixed_controls(batch)
    normalizer = StateNormalizer()
    normalizer.fit([batch])
    normalized = normalized_controls(controls, normalizer)
    # Zero means explicit no-contact in physical space, and -1 after normalization.
    assert normalized.values["contact"][0, 0, 0].item() == -1
    assert normalized.values["contact"][0, 0, 20].item() == 1
    states = {key: torch.randn_like(value) for key, value in normalized.values.items()}
    anchored = normalized.apply(states)
    for key in MODALITIES:
        torch.testing.assert_close(anchored[key][~normalized.masks[key]],
                                   states[key][~normalized.masks[key]], rtol=0, atol=0)
    physical = normalizer.decode(anchored)
    for key in MODALITIES:
        torch.testing.assert_close(physical[key][controls.masks[key]],
                                   controls.values[key][controls.masks[key]], atol=1e-6, rtol=1e-6)


def test_training_has_finite_gradients_through_relation_and_coarse_stages():
    batch = example_batch(batch_size=1)
    model = small_model()
    model.normalizer.fit([batch])
    loss, metrics = HOIDiffusion(steps=8).training_loss(
        model, batch, mixed_controls(batch), generator=torch.Generator().manual_seed(3))
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"No gradient through {name}"
        assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient: {name}"
    assert model.relation_net.projection[0].weight.grad.abs().sum() > 0
    assert model.coarse_head.projection[0].weight.grad.abs().sum() > 0


def test_unobserved_control_values_and_reference_states_do_not_leak():
    batch = example_batch(batch_size=1)
    model = small_model().eval()
    controls = mixed_controls(batch)
    cond = normalized_controls(controls, model.normalizer)
    hidden = replace(cond, values={
        key: torch.where(cond.masks[key], cond.values[key], torch.full_like(cond.values[key], 1234.))
        for key in MODALITIES
    })
    noisy = {key: torch.randn_like(value) for key, value in controls.values.items()}
    changed = {**batch, **{key: torch.full_like(batch[key], 987.) for key in MODALITIES}}
    times = torch.tensor([[2, 4, 6]])
    with torch.no_grad():
        a = model(noisy, times, cond, batch)
        b = model(noisy, times, hidden, changed)
    for key in MODALITIES:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_padded_state_values_do_not_change_valid_predictions():
    batch = example_batch()
    model = small_model().eval()
    controls = ControlBatch.empty_like(batch, batch["valid_frames"])
    cond = normalized_controls(controls, model.normalizer)
    a = {key: torch.randn_like(value) for key, value in controls.values.items()}
    b = {key: value.clone() for key, value in a.items()}
    for key in MODALITIES:
        b[key][~batch["valid_frames"]] = 100.
    with torch.no_grad():
        out_a = model(a, torch.full((2, 3), 3), cond, batch)
        out_b = model(b, torch.full((2, 3), 3), cond, batch)
    for key in MODALITIES:
        torch.testing.assert_close(out_a[key][batch["valid_frames"]],
                                   out_b[key][batch["valid_frames"]], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("projection_steps", [0, 2])
def test_sampling_preserves_hard_anchors_and_returns_valid_free_rotations(projection_steps):
    batch = example_batch(batch_size=1)
    controls = mixed_controls(batch)
    model = small_model()
    model.normalizer.fit([batch])
    output = HOIDiffusion(steps=8).sample(model, batch, controls, steps=3, seed=10,
                                         projection_steps=projection_steps)
    assert_anchors_exact(output, controls)
    assert_free_rotations_orthonormal(output, controls)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert ((output["contact"] >= 0) & (output["contact"] <= 1)).all()


def test_sampling_masks_padding_and_does_not_read_unknown_reference_motion():
    batch = example_batch()
    model = small_model()
    controls = sample_controls(batch, batch["valid_frames"], mode="111")
    changed = {**batch, **{key: torch.full_like(batch[key], 456.) for key in MODALITIES}}
    diffusion = HOIDiffusion(steps=6)
    first = diffusion.sample(model, batch, controls, steps=2, seed=9)
    second = diffusion.sample(model, changed, controls, steps=2, seed=9)
    for key in MODALITIES:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        assert not first[key][~batch["valid_frames"]].any()


def test_relation_projection_moves_free_object_towards_anchored_contact():
    batch = example_batch(batch_size=1, frames=1)
    # All body poses are known. Only object translation can resolve the contact.
    controls = compile_controls({"entries": [
        {"modality": "human", "frames": [0], "from_reference": True},
        {"modality": "object", "frames": [0], "features": [3, 4, 5, 6, 7, 8], "from_reference": True},
        {"modality": "contact", "frames": [0], "joints": [20], "values": 1},
    ]}, batch, batch["valid_frames"])
    states = {key: batch[key].clone() for key in MODALITIES}
    before, _ = nearest_surface(states["human"][..., :3], states["object"], batch["object_points"])
    after = project_relations(states, controls, batch, steps=10)
    distance, _ = nearest_surface(after["human"][..., :3], after["object"], batch["object_points"])
    assert distance[0, 0, 20] < before[0, 0, 20]
    assert_anchors_exact(after, controls)
