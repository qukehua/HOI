"""Numerical and protocol regressions; no trained weights or optional sklearn required."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from unified_hoi.parametric.evaluate import (generation_metrics, generation_features,
                                           contact_scores, reconstruction, _cached_states)
from unified_hoi.parametric.evaluation_protocol import (protocol_options, frame_indices,
    window_selection, aggregate_reports, assert_comparable, VERSION)


class EvaluationTests(unittest.TestCase):
    def test_squared_distance_not_square_of_mean(self):
        gen = np.array([[0., 0.], [10., 0.]])
        ref = np.array([[0., 2.], [10., 4.]])
        result = generation_metrics(gen, ref, ref)
        self.assertEqual(result["MMD"], 10.)  # mean([2^2, 4^2]), not 3 or 9
        self.assertEqual(result["SD"], 10.)
        self.assertEqual(result["COV"], 1.)

    def test_nearest_neighbor_metrics_match_brute_force(self):
        rng = np.random.default_rng(42)
        gen, ref, train = (rng.normal(size=(n, 7)).astype(np.float32) for n in (23, 23, 31))
        result = generation_metrics(gen, ref, train)
        distances = ((gen[:, None].astype(float) - ref.astype(float)) ** 2).sum(-1)
        self.assertAlmostEqual(result["MMD"], distances.min(0).mean())
        self.assertAlmostEqual(result["COV"], len(np.unique(distances.argmin(1))) / len(ref))
        pooled = np.concatenate([gen, ref]).astype(float)
        full = ((pooled[:, None] - pooled) ** 2).sum(-1)
        np.fill_diagonal(full, np.inf)
        labels = np.arange(len(pooled)) < len(gen)
        self.assertEqual(result["1-NNA"], np.mean(labels[full.argmin(1)] == labels))

    def test_duplicates_exclude_self_by_index(self):
        a = np.array([[1., 2.]])
        self.assertEqual(generation_metrics(a, a, a)["1-NNA"], 0.)

    def test_generation_rejects_unbalanced_or_nonfinite_inputs(self):
        a = np.zeros((2, 3))
        for invalid in (np.zeros((1, 3)), np.full((2, 3), np.nan), np.zeros((2, 4))):
            with self.assertRaises(ValueError):
                generation_metrics(invalid, a, a)

    def test_frame_selection_does_not_restart_at_window_boundary(self):
        ts = 5.3 + np.arange(35) / 10
        all_indices = frame_indices(ts, 10, 1, 5.3)
        np.testing.assert_array_equal(all_indices, [0, 10, 20, 30])
        plan = dict(indices=all_indices, frame_ids={int(i): str(i) for i in all_indices})
        a, _ = window_selection(plan, 0, 13)
        b, _ = window_selection(plan, 13, 35)
        np.testing.assert_array_equal(np.concatenate([a, b + 13]), all_indices)
        np.testing.assert_array_equal(frame_indices(ts[13:], 10, 1, 5.3), b)

    def test_frame_rate_requires_real_frames(self):
        for fps in (20., 3., -1.):
            with self.assertRaises(ValueError):
                frame_indices(np.arange(10) / 10, 10, fps, 0)
        np.testing.assert_array_equal(frame_indices([.1, .2], 10, 0, 0), [0, 1])

    def test_profile_defaults(self):
        self.assertEqual(protocol_options()["eval_fps"], 1.)
        self.assertEqual(protocol_options()["train_reference_fps"], 10.)
        self.assertEqual(protocol_options("native")["eval_fps"], 0.)
        self.assertEqual(protocol_options("auto"), protocol_options("tridi"))

    def test_object_features_use_mesh_centroid_and_declared_origin(self):
        pose = np.array([[3., 4., 5., 1., 0., 0., 0., 1., 0.]])
        joints = np.zeros((1, 73, 3))
        joints[:, 0] = [1., 2., 3.]
        vertices = np.array([[1., 0., 0.], [3., 0., 0.]])
        feature = generation_features("object_pose", joints, pose, vertices=vertices, coordinate_frame="pelvis")
        np.testing.assert_allclose(feature[0, 9:], [4., 2., 2.])
        shifted = pose.copy()
        shifted[:, :3] += [50., -20., 30.]
        same = generation_features("object_pose", joints + [50., -20., 30.], shifted,
                                   vertices=vertices, coordinate_frame="pelvis")
        np.testing.assert_allclose(feature, same)

    def test_contact_global_f1_is_not_window_average(self):
        a = contact_scores([True], [True])
        b = contact_scores([False, False, False], [True, True, False])
        rows = [dict(metrics={"MPJPE": 1., "SBJ_CONTACT_F1": a["f1"]}, details={"SBJ_CONTACT": a}, frames=1),
                dict(metrics={"MPJPE": 3., "SBJ_CONTACT_F1": b["f1"]}, details={"SBJ_CONTACT": b}, frames=3)]
        metrics, primary, micro = aggregate_reports(rows)
        self.assertEqual(metrics["MPJPE"]["macro_mean_per_window"], 2.)
        self.assertEqual(primary["MPJPE"], 2.5)
        self.assertEqual(primary["SBJ_CONTACT_F1"], .5)
        self.assertEqual(micro["SBJ_CONTACT"]["FN"], 2)

    def test_reconstruction_selects_only_requested_valid_frames(self):
        class Geometry:
            dataset, support_hash = "behave", "fixture"

            def __call__(self, h, gender):
                base = h[..., :3, None].transpose(-1, -2)
                return base.expand(-1, -1, 690, -1), base.expand(-1, -1, 73, -1)

        obj = torch.tensor([[[0., 0., 0., 1., 0., 0., 0., 1., 0.]]]).repeat(1, 3, 1)
        batch = dict(human=torch.zeros(1, 3, 3), object=obj, valid_frames=torch.tensor([[True, True, False]]),
                     gender=torch.tensor([1]))
        pred = {"human": batch["human"], "object": obj.clone()}
        pred["object"][0, :, 0] = torch.tensor([1., 3., 100.])
        report = reconstruction(pred, batch, Geometry(), "010", np.zeros((1, 3)), frame_indices=[1])
        self.assertEqual(report["frames"], 1)
        self.assertEqual(report["metrics"]["OBJ_V2V"], 3.)
        with self.assertRaises(ValueError):
            reconstruction(pred, batch, Geometry(), "010", np.zeros((1, 3)), frame_indices=[2])

    def test_comparison_rejects_wrong_modes_protocol_and_frames(self):
        summary = dict(dataset="behave", protocol=protocol_options(), reconstruction={"100": {}},
                       evaluation_frames_sha256="a", reference_train_frames_sha256="b",
                       reference_values_sha256="c", text_conditioned=False, support_sha256="support",
                       body_assets_sha256={"male": "m", "female": "f"},
                       split="test", steps=50, seed=42, window=120, sampler="DDIM_eta0")
        assert_comparable(summary, summary, "100")
        for key, value in (("dataset", "omomo"), ("text_conditioned", True),
                           ("evaluation_frames_sha256", "other"), ("steps", 100)):
            other = deepcopy(summary)
            other[key] = value
            with self.assertRaises(ValueError):
                assert_comparable(summary, other, "100")
        other = deepcopy(summary)
        other["reconstruction"] = {"101": {}}
        with self.assertRaises(ValueError):
            assert_comparable(summary, other, "100")
        other = deepcopy(summary)
        other["protocol"]["version"] = VERSION + "_old"
        with self.assertRaises(ValueError):
            assert_comparable(summary, other, "100")

    def test_cache_rejects_wrong_weights_and_changed_conditions(self):
        batch = dict(human=torch.zeros(1, 2, 325), object=torch.zeros(1, 2, 9),
                     contact=torch.zeros(1, 2, 128), valid_frames=torch.ones(1, 2, dtype=torch.bool))
        record = dict(sequence_id="seq", gender="male", body_model_sha256="body", timestamps=np.array([0., .1]))
        values = dict(human_params=np.zeros((2, 325), np.float32), object=np.zeros((2, 9), np.float32),
                      contact_latent=np.zeros((2, 128), np.float32), timestamps=record["timestamps"],
                      representation="parametric_v1", dataset="behave", mode="100", sequence_id="seq",
                      window_start=0, gender="male", body_model_sha256="body", support_sha256="support",
                      codec_sha256="codec", checkpoint_sha256="checkpoint")
        geometry = SimpleNamespace(dataset="behave", support_hash="support")
        saved = {"codec_metadata": {"file_sha256": "codec"}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prediction.npz"
            np.savez(path, **values)
            result = _cached_states(path, batch, geometry, saved, record, "100", 0, "cpu", "checkpoint")
            self.assertTrue(torch.equal(result["object"], batch["object"]))
            with self.assertRaisesRegex(ValueError, "checkpoint hash"):
                _cached_states(path, batch, geometry, saved, record, "100", 0, "cpu", "wrong")
            values["object"][0, 0] = 1.
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "observed object"):
                _cached_states(path, batch, geometry, saved, record, "100", 0, "cpu", "checkpoint")


if __name__ == "__main__":
    unittest.main()
