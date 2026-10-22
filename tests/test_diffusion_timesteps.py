"""Training/sampling noise-level contract for native H/O/C generation."""
import unittest

import torch
from torch import nn

from unified_hoi.parametric.body import human_dim
from unified_hoi.parametric.diffusion import (
    MODES, ParametricDiffusion, controls_for, training_timesteps,
)
from unified_hoi.parametric.model import MODALITIES
from unified_hoi.parametric.train import run_training


def batch_for(dataset, size=16):
    generator = torch.Generator().manual_seed(17)
    dimensions = (human_dim(dataset), 9, 128)
    batch = {k: torch.randn(size, 3, d, generator=generator)
             for k, d in zip(MODALITIES, dimensions)}
    batch["valid_frames"] = torch.ones(size, 3, dtype=torch.bool)
    return batch


class IdentityNormalizer:
    def encode(self, batch):
        return {k: batch[k] for k in MODALITIES}

    def decode(self, states):
        return states


class RecordingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = IdentityNormalizer()
        self.scale = nn.Parameter(torch.tensor(.5))

    def forward(self, states, times, masks, batch):
        self.states = {k: v.detach().clone() for k, v in states.items()}
        self.times = times.detach().clone()
        return {k: v * self.scale for k, v in states.items()}


class TimestepTests(unittest.TestCase):
    def assert_mode_times(self, times, mode, steps):
        unknown = [i for i, bit in enumerate(mode) if bit == "1"]
        shared = times[:, unknown[0]]
        self.assertTrue(((shared >= 1) & (shared <= steps)).all())
        for i, bit in enumerate(mode):
            expected = shared if bit == "1" else torch.zeros_like(shared)
            torch.testing.assert_close(times[:, i], expected, rtol=0, atol=0)

    def test_seven_modes_and_partial_omomo_hands(self):
        for dataset in ("behave", "omomo"):
            batch = batch_for(dataset)
            for mode in MODES:
                with self.subTest(dataset=dataset, mode=mode):
                    masks = controls_for(batch, dataset, mode)
                    times = training_timesteps(masks, 1000, generator=torch.Generator().manual_seed(42))
                    self.assert_mode_times(times, mode, 1000)
                    # Each example draws its own t, even in a joint-generation batch.
                    self.assertGreater(len(times.unique(dim=0)), 1)

    def test_mixed_modes_in_one_batch(self):
        batches = [batch_for("omomo", 1) for _ in MODES]
        rows = [controls_for(batch, "omomo", mode) for batch, mode in zip(batches, MODES)]
        masks = {k: torch.cat([row[k] for row in rows]) for k in MODALITIES}
        times = training_timesteps(masks, 1000, generator=torch.Generator().manual_seed(42))
        for row, mode in zip(times, MODES):
            self.assert_mode_times(row[None], mode, 1000)

    def test_independent_option_preserves_legacy_draws(self):
        batch = batch_for("omomo")
        for mode in MODES:
            with self.subTest(mode=mode):
                masks = controls_for(batch, "omomo", mode)
                expected = torch.randint(1, 1001, (16, 3), generator=torch.Generator().manual_seed(42))
                for i, bit in enumerate(mode):
                    if bit == "0":
                        expected[:, i] = 0
                actual = training_timesteps(masks, 1000, "independent", torch.Generator().manual_seed(42))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_training_uses_shared_times_and_preserves_conditions(self):
        for dataset in ("behave", "omomo"):
            batch = batch_for(dataset)
            for mode in MODES:
                with self.subTest(dataset=dataset, mode=mode):
                    masks = controls_for(batch, dataset, mode)
                    model = RecordingDenoiser()
                    loss, metrics = ParametricDiffusion().training_loss(
                        model, batch, masks, None, {"geometry_weight": 0},
                        generator=torch.Generator().manual_seed(42))
                    self.assert_mode_times(model.times, mode, 1000)
                    for k in MODALITIES:
                        torch.testing.assert_close(model.states[k][masks[k]], batch[k][masks[k]],
                                                   rtol=0, atol=0)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(model.scale.grad))
                    self.assertNotEqual(model.scale.grad.item(), 0.)
                    self.assertEqual(set(metrics), {"loss", *["denoise_" + k for k in MODALITIES]})

    def test_noise_is_independent_and_seeded(self):
        batch = batch_for("behave")
        masks = controls_for(batch, "behave", "111")
        diffusion = ParametricDiffusion()
        models = [RecordingDenoiser(), RecordingDenoiser()]
        for model in models:
            diffusion.training_loss(model, batch, masks, None, {"geometry_weight": 0},
                                    generator=torch.Generator().manual_seed(42))
        torch.testing.assert_close(models[0].times, models[1].times, rtol=0, atol=0)
        noises = []
        for i, k in enumerate(MODALITIES):
            torch.testing.assert_close(models[0].states[k], models[1].states[k], rtol=0, atol=0)
            alpha = diffusion.alpha[models[0].times[:, i]][:, None, None]
            noises.append((models[0].states[k][..., :9] - alpha.sqrt() * batch[k][..., :9])
                          / (1 - alpha).sqrt())
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(torch.allclose(noises[a], noises[b]))

    def test_invalid_policy_rejected(self):
        masks = controls_for(batch_for("behave"), "behave", "111")
        with self.assertRaisesRegex(ValueError, "timestep_sampling"):
            training_timesteps(masks, 1000, "sharedd")
        with self.assertRaisesRegex(ValueError, "timestep_sampling"):
            run_training({"timestep_sampling": "sharedd"})


if __name__ == "__main__":
    unittest.main()
