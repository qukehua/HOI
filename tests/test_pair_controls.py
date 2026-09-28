import pytest
import torch

from unified_hoi.controls import sample_controls


@pytest.mark.parametrize("mode,signature", [("human_object", "H+O"),
                                            ("human_contact", "H+C"), ("object_contact", "O+C")])
def test_pairwise_sparse_controls_have_exact_signature_and_obey_holdout(mode, signature):
    states = {"human": torch.zeros(1, 12, 22, 9), "object": torch.zeros(1, 12, 9),
              "contact": torch.zeros(1, 12, 22)}
    valid = torch.ones(1, 12, dtype=torch.bool)
    controls = sample_controls(states, valid, mode, torch.Generator().manual_seed(0))
    assert controls.signatures == (signature,)
    for key in states:
        assert not controls.masks[key].all()
    with pytest.raises(ValueError, match="cannot sample"):
        sample_controls(states, valid, mode=mode, holdout_signatures=[signature], max_attempts=2)
