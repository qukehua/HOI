import torch

from unified_hoi.objectives import geometric_losses


def scene():
    human = torch.zeros(1, 2, 22, 9)
    human[..., 1] = -2  # Camera origin may be above the floor.
    human[..., 3] = human[..., 7] = 1
    obj = torch.zeros(1, 2, 9)
    obj[..., 3] = obj[..., 7] = 1
    states = {"human": human, "object": obj, "contact": torch.zeros(1, 2, 22)}
    batch = {"object_points": torch.zeros(1, 1, 3), "rest_offsets": torch.zeros(1, 22, 3),
             "valid_frames": torch.ones(1, 2, dtype=torch.bool), "fps": torch.tensor([10.])}
    return states, batch


def test_no_floor_force_without_calibrated_ground():
    states, batch = scene()
    assert geometric_losses(states, batch)["floor"] == 0
    batch.update(floor_available=torch.tensor([True]), floor_y=torch.tensor([-1.]))
    assert geometric_losses(states, batch)["floor"] == 1


def test_noncontact_uses_dataset_distance_threshold():
    states, batch = scene()
    states["human"][..., :3] = 0
    states["human"][..., 0] = .075
    assert geometric_losses(states, batch)["noncontact"] == 0
    batch["contact_threshold"] = torch.tensor([.1])
    assert torch.allclose(geometric_losses(states, batch)["noncontact"], torch.tensor(.025 ** 2))
