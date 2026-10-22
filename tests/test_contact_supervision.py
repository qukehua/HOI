"""Contact gradients, sparse class balance, masks, and complete mesh geometry."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from unified_hoi.geometry import object_to_world
from unified_hoi.parametric.contact import ContactAE
from unified_hoi.parametric.contact_supervision import (
    balanced_contact_mean, contact_loss_options, decoded_contact_loss, full_mesh_distances,
)
from unified_hoi.parametric.diffusion import MODES, ParametricDiffusion, controls_for
from unified_hoi.parametric.evaluate import mesh_contacts
from unified_hoi.parametric.model import Normalizer


class FixedPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = Normalizer("behave")
        self.normalizer.contact_mean.fill_(.4)
        self.normalizer.contact_std.fill_(2.)
        self.codec = ContactAE().requires_grad_(False)
        self.contact_threshold = .75
        self.latent = nn.Parameter(torch.randn(3, 2, 128) * .1)

    def forward(self, states, times, masks, batch):
        return {"human": torch.zeros_like(states["human"]),
                "object": torch.zeros_like(states["object"]), "contact": self.latent}


class ContactSupervisionTests(unittest.TestCase):
    def test_class_balance_is_not_diluted_by_many_negatives(self):
        for negatives in (1, 999):
            errors = torch.tensor([9.] + [1.] * negatives, requires_grad=True)
            target = torch.tensor([1.] + [0.] * negatives)
            loss, stats = balanced_contact_mean(errors, target)
            self.assertEqual(loss.item(), 5.)
            self.assertEqual(stats["positive_count"], 1)
            self.assertEqual(stats["negative_count"], negatives)
            loss.backward()
            self.assertEqual(errors.grad[0].item(), .5)
            self.assertAlmostEqual(errors.grad[1:].sum().item(), .5, places=6)

    def test_single_class_and_empty_support_are_finite(self):
        for label in (0., 1.):
            values = torch.tensor([2., 4.], requires_grad=True)
            target = torch.full_like(values, label)
            loss, _ = balanced_contact_mean(values, target)
            self.assertEqual(loss.item(), 3.)
            loss.backward()
            torch.testing.assert_close(values.grad, torch.full_like(values, .5))
        values = torch.tensor([float("nan"), 4.], requires_grad=True)
        loss, stats = balanced_contact_mean(values, torch.ones(2), torch.zeros(2, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(stats["positive_count"] + stats["negative_count"], 0)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.zeros(2))

    def test_decoder_is_frozen_but_latent_receives_gradients(self):
        torch.manual_seed(23)
        codec = ContactAE().requires_grad_(False)
        before = {k: v.clone() for k, v in codec.state_dict().items()}
        latent = torch.randn(2, 3, 128, requires_grad=True)
        target = torch.zeros(2, 3, 690)
        target[..., :20] = 1
        frames = torch.tensor([[True, False, False], [True, True, False]])
        loss, stats = decoded_contact_loss(codec, latent, target, frames, .75)
        loss.backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(latent.grad[frames].abs().sum().item(), 0.)
        self.assertEqual(latent.grad[~frames].abs().sum().item(), 0.)
        self.assertEqual(stats["frames"], 3)
        self.assertTrue(all(p.grad is None for p in codec.parameters()))
        for k, value in codec.state_dict().items():
            torch.testing.assert_close(value, before[k], rtol=0, atol=0)

    def test_training_bce_uses_physical_latents_and_only_generated_valid_frames(self):
        torch.manual_seed(42)
        model = FixedPrediction()
        batch = {"human": torch.zeros(3, 2, 325), "object": torch.zeros(3, 2, 9),
                 "contact": torch.zeros(3, 2, 128), "surface_contact": torch.zeros(3, 2, 690),
                 "valid_frames": torch.tensor([[True, True], [True, False], [False, False]])}
        batch["surface_contact"][..., :10] = 1
        diffusion = ParametricDiffusion()
        masks = controls_for(batch, "behave", "111")
        masks["contact"][0] = True
        old_config = {"geometry_weight": 0}
        new_config = old_config | {"contact_supervision": {"decoded_bce_weight": .3}}
        old_loss, _ = diffusion.training_loss(model, batch, masks, None, old_config)
        new_loss, metrics = diffusion.training_loss(model, batch, masks, None, new_config)
        logits = model.codec.decode_logits(model.latent[1, 0] * 2. + .4)
        elementwise = F.binary_cross_entropy_with_logits(logits, batch["surface_contact"][1, 0],
                                                        reduction="none")
        expected = .5 * (elementwise[:10].mean() + elementwise[10:].mean())
        torch.testing.assert_close(new_loss - old_loss, .3 * expected)
        self.assertAlmostEqual(metrics["contact_bce_weighted"], .3 * expected.item(), places=6)
        self.assertEqual(metrics["contact_decoded_frames"], 1)
        gradient = torch.autograd.grad(new_loss - old_loss, model.latent)[0]
        self.assertGreater(gradient[1, 0].abs().sum().item(), 0.)
        gradient[1, 0] = 0
        self.assertEqual(gradient.abs().sum().item(), 0.)

    def test_all_modes_exclude_observed_contact_from_bce(self):
        model = FixedPrediction()
        batch = {"human": torch.zeros(3, 2, 325), "object": torch.zeros(3, 2, 9),
                 "contact": torch.zeros(3, 2, 128), "surface_contact": torch.zeros(3, 2, 690),
                 "valid_frames": torch.ones(3, 2, dtype=torch.bool)}
        for mode in MODES:
            with self.subTest(mode=mode):
                masks = controls_for(batch, "behave", mode)
                loss, metrics = ParametricDiffusion().training_loss(
                    model, batch, masks, None,
                    {"geometry_weight": 0, "contact_supervision": {"decoded_bce_weight": 1.}})
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(metrics["contact_decoded_frames"], 6 if mode[-1] == "1" else 0)
                if mode[-1] == "0":
                    self.assertEqual(metrics["contact_bce"], 0.)

    def test_full_mesh_handles_mixed_templates_scale_and_nonconsecutive_frames(self):
        meshes = {"a": np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [.2, .2, 0.]], np.float32),
                  "b": np.array([[.4, 0., 0.], [0., .7, 0.], [0., 0., .3]], np.float32)}
        batch = {"object_mesh_path": ["a", "b", "a"], "object_mesh_scale": torch.tensor([1., 2., .5])}
        indices = torch.tensor([2, 0, 2, 1])
        pose = torch.tensor([[[.1, -.2, .3, 0., -1., 0., 1., 0., 0.]]], dtype=torch.float64)
        pose = pose.repeat(4, 1, 1).requires_grad_(True)
        points = torch.tensor([[[[.2, .2, .02], [.6, .8, .03]]]], dtype=torch.float64).repeat(4, 1, 1, 1)
        body = (points + pose.detach()[..., None, :3]).requires_grad_(True)
        with patch("unified_hoi.parametric.contact_supervision.load_mesh_asset",
                   side_effect=lambda path: (meshes[path], np.array([[0, 1, 2], [0, 1, len(meshes[path]) - 1]]))):
            actual = full_mesh_distances(body, pose, batch, indices)
        expected = []
        for row, bi in enumerate(indices.tolist()):
            vertices = meshes[batch["object_mesh_path"][bi]] * float(batch["object_mesh_scale"][bi])
            world = object_to_world(torch.tensor(vertices, dtype=torch.float64)[None], pose[row:row + 1])
            expected.append(torch.linalg.vector_norm(body[row:row + 1, :, :, None] - world[:, :, None],
                                                     dim=-1).min(-1).values)
            labels = mesh_contacts(body[row].detach().numpy(), pose[row].detach().numpy(), vertices)
            np.testing.assert_array_equal((actual[row] <= .05).detach().numpy(), labels)
        expected = torch.cat(expected)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual.sum(), (body, pose), retain_graph=True)
        expected_grad = torch.autograd.grad(expected.sum(), (body, pose))
        for actual_value, expected_value in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual_value, expected_value)

    def test_full_mesh_keeps_vertices_missing_from_sparse_support(self):
        vertices = np.zeros((2050, 3), np.float32)
        vertices[:, 0] = np.arange(2050) + 10.
        vertices[-1] = [0., 0., 0.]
        batch = {"object_mesh_path": ["mesh"], "object_mesh_scale": torch.ones(1)}
        body = torch.tensor([[[[0., 0., .02]]]], requires_grad=True)
        pose = torch.tensor([[[0., 0., 0., 1., 0., 0., 0., 1., 0.]]])
        faces = np.stack([np.zeros(2048, int), np.ones(2048, int), np.arange(2, 2050)], axis=-1)
        with patch("unified_hoi.parametric.contact_supervision.load_mesh_asset", return_value=(vertices, faces)):
            distance = full_mesh_distances(body, pose, batch, torch.tensor([0]))
        self.assertAlmostEqual(distance.item(), .02, places=6)
        distance.sum().backward()
        torch.testing.assert_close(body.grad, torch.tensor([[[[0., 0., 1.]]]]))

    def test_orphan_vertices_are_excluded_like_stored_contact_labels(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [1., 0., 1.]], np.float32)
        batch = {"object_mesh_path": ["mesh"], "object_mesh_scale": torch.ones(1)}
        body = torch.tensor([[[[0., 0., .02]]]])
        pose = torch.tensor([[[0., 0., 0., 1., 0., 0., 0., 1., 0.]]])
        with patch("unified_hoi.parametric.contact_supervision.load_mesh_asset",
                   return_value=(vertices, np.array([[1, 2, 3]]))):
            distance = full_mesh_distances(body, pose, batch, torch.tensor([0]))
        self.assertGreater(distance.item(), 1.)

    def test_options_preserve_old_objective_and_reject_bad_values(self):
        self.assertEqual(contact_loss_options({}), {"decoded_bce_weight": 0.,
                         "balance_surface_distance": False, "full_mesh_distance": False})
        for value in (-1, float("inf"), float("nan"), "1", True):
            with self.assertRaisesRegex(ValueError, "decoded_bce_weight"):
                contact_loss_options({"contact_supervision": {"decoded_bce_weight": value}})
        for options in (None, {"full_mesh_distance": "true"}, {"typo": True}):
            with self.assertRaises(ValueError):
                contact_loss_options({"contact_supervision": options})


if __name__ == "__main__":
    unittest.main()
