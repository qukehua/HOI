"""Safety/gradient invariants for continuous refinement; no pretrained downloads."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from unified_hoi.geometry import rotation6d_to_matrix
from unified_hoi.token_ar import MODALITIES, MODES
from unified_hoi.token_ar.body import beta_count, human_dim
from unified_hoi.token_ar.refinement import (
    ContinuousRefiner, LOSS_WEIGHTS, RefinerConfig, check_sampler, load_refiner,
    nearest_surface, object_vertex_mse, refinement_loss, refiner_payload, rotation_delta,
)


class ToyGeometry(nn.Module):
    """Differentiable pose/translation fixture, not a replacement for SMPL validation."""
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.layers = {"male": type("Asset", (), {"asset_hash": "male"})(),
                       "female": type("Asset", (), {"asset_hash": "female"})()}
        self.register_buffer("points", torch.randn(690, 3) * .1)

    def forward(self, h, gender):
        b = beta_count(self.dataset)
        rotation = rotation6d_to_matrix(h[..., b:b + 6])
        vertices = self.points @ rotation.transpose(-1, -2) + h[..., None, -3:]
        return vertices, vertices[..., :73, :]


def fixture(dataset="behave", frames=4):
    h = torch.zeros(frames, human_dim(dataset))
    h[:, beta_count(dataset):-3] = torch.tensor([1., 0, 0, 0, 1, 0]).repeat(52)
    o = torch.tensor([.1, 0, 0, 1., 0, 0, 0, 1, 0]).repeat(frames, 1)
    c = (torch.rand(frames, 690) > .9).float()
    return dict(human=h, object=o, contact=c)


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)
        self.mesh = torch.randn(80, 3) * .1
        self.bps = torch.randn(1024, 3) * .1
        self.gender = torch.tensor(1)

    def test_rotation_identity_has_finite_nonzero_gradient(self):
        v = torch.zeros(3, requires_grad=True)
        r = rotation_delta(v)
        torch.testing.assert_close(r, torch.eye(3))
        r[0, 1].backward()
        self.assertTrue(torch.isfinite(v.grad).all())
        self.assertGreater(float(v.grad.norm()), 0)

    def test_exact_mesh_moment_loss_and_distance_gradients(self):
        pred = fixture()["object"].clone().requires_grad_()
        target = pred.detach().clone()
        target[:, :3] += .2
        actual = object_vertex_mse(pred, target, self.mesh)
        pv = self.mesh @ rotation6d_to_matrix(pred[:, 3:]).transpose(-1, -2) + pred[:, None, :3]
        gv = self.mesh @ rotation6d_to_matrix(target[:, 3:]).transpose(-1, -2) + target[:, None, :3]
        torch.testing.assert_close(actual, (pv - gv).square().mean())
        points = torch.randn(4, 7, 3, requires_grad=True)
        distance, _, local = nearest_surface(points, pred, self.mesh)
        torch.testing.assert_close(distance, torch.cdist(local, self.mesh).min(-1).values, atol=1e-6, rtol=1e-5)
        (actual + distance.mean()).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad[:, :3].norm()), 0)
        self.assertGreater(float(points.grad.norm()), 0)

    def test_all_masks_preserve_observed_and_valid_rotations(self):
        for dataset in ("behave", "omomo"):
            model = ContinuousRefiner(RefinerConfig(dataset, width=16, layers=1))
            nn.init.normal_(model.head.weight, std=.03)
            geometry = ToyGeometry(dataset)
            coarse = fixture(dataset)
            for mode in MODES:
                result, _ = model(coarse, mode, self.bps, self.gender, 10., geometry, self.mesh)
                for key, bit in zip(MODALITIES, mode):
                    if bit == "0":
                        self.assertTrue(torch.equal(result[key], coarse[key]), (dataset, mode, key))
                for six in (result["human"][:, beta_count(dataset):-3].reshape(-1, 6), result["object"][:, 3:]):
                    r = rotation6d_to_matrix(six)
                    torch.testing.assert_close(torch.linalg.det(r), torch.ones(len(r)), atol=1e-5, rtol=1e-5)
                self.assertTrue(((result["contact"] >= 0) & (result["contact"] <= 1)).all())
                if dataset == "omomo":
                    torch.testing.assert_close(result["human"][:, beta_count(dataset) + 132:-3],
                                               coarse["human"][:, beta_count(dataset) + 132:-3])

    def test_geometry_losses_backpropagate_only_to_refiner(self):
        for mode in MODES:
            model = ContinuousRefiner(RefinerConfig("behave", width=16, layers=1))
            target = fixture()
            coarse = {k: v.clone() for k, v in target.items()}
            if mode[0] == "1":
                coarse["human"][:, -3:] += .15
            if mode[1] == "1":
                coarse["object"][:, :3] += .25
            if mode[2] == "1":
                coarse["contact"] = .2 + .6 * coarse["contact"]
            for v in coarse.values():
                v.requires_grad_()
            loss, _, _ = refinement_loss(model, coarse, target, mode, self.bps, self.gender, 10.,
                                         ToyGeometry("behave"), self.mesh)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(model.head.weight.grad).all())
            self.assertGreater(float(model.head.weight.grad.norm()), 0)
            # Coarse tensors include the frozen AR/VQ output; no gradients may escape.
            for v in coarse.values():
                self.assertIsNone(v.grad)

    def test_observed_corruption_rejected(self):
        model = ContinuousRefiner(RefinerConfig("behave", width=16, layers=1))
        target = fixture()
        coarse = {k: v.clone() for k, v in target.items()}
        coarse["human"][:, -3:] += 1
        with self.assertRaisesRegex(ValueError, "Observed human"):
            refinement_loss(model, coarse, target, "010", self.bps, self.gender, 10., ToyGeometry("behave"), self.mesh)

    def test_short_window_and_unknown_targets_not_forward_inputs(self):
        model = ContinuousRefiner(RefinerConfig("behave", width=16, layers=1))
        coarse = fixture(frames=1)
        geometry = ToyGeometry("behave")
        first, _, output = refinement_loss(model, coarse, coarse, "111", self.bps, self.gender, 10., geometry, self.mesh)
        changed = {k: v.clone() for k, v in coarse.items()}
        changed["human"][:, -3:] += 9
        changed["object"][:, :3] -= 7
        changed["contact"] = 1 - changed["contact"]
        second, _, other = refinement_loss(model, coarse, changed, "111", self.bps, self.gender, 10., geometry, self.mesh)
        self.assertNotEqual(float(first.detach()), float(second.detach()))
        for k in MODALITIES:
            self.assertTrue(torch.equal(output[k], other[k]))

    def test_optimizer_can_reduce_object_geometry_error(self):
        model = ContinuousRefiner(RefinerConfig("behave", width=16, layers=1))
        target = fixture()
        coarse = {k: v.clone() for k, v in target.items()}
        coarse["object"][:, :3] += torch.tensor([.2, -.2, .1])
        geometry = ToyGeometry("behave")
        weights = dict.fromkeys(LOSS_WEIGHTS, 0.) | {"object_vertices": 1.}
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        values = []
        for _ in range(25):
            optimizer.zero_grad()
            loss, _, _ = refinement_loss(model, coarse, target, "010", self.bps, self.gender, 10., geometry, self.mesh, weights)
            values.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        self.assertLess(values[-1], values[0] * .3)

    def test_checkpoint_source_and_sampler_binding(self):
        model = ContinuousRefiner(RefinerConfig("behave", width=16, layers=1))
        geometry = ToyGeometry("behave")
        source = dict(dataset="behave", codec_sha256="vq", support_sha256="support", ar_metadata_sha256="ar",
                      body_assets={"male": "male", "female": "female"}, temperature=0., top_k=50, text_conditioned=True)
        meta = dict(config={"dataset": "behave"}, codec_sha256="vq", support_sha256="support")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "refiner.pt"
            torch.save(refiner_payload(model, source), path)
            loaded, saved = load_refiner(path, meta, "ar", geometry)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, loaded.state_dict()[k])
            check_sampler(saved, 0., 1, True)
            with self.assertRaises(ValueError):
                check_sampler(saved, 1., 50, True)
            with self.assertRaises(ValueError):
                load_refiner(path, meta, "different-ar", geometry)


if __name__ == "__main__":
    unittest.main()
