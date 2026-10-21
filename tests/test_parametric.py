"""Native-path regression checks. Licensed-asset tests skip on source-only checkouts.

Run from the repository root: python -m unittest discover -s tests -v
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from unified_hoi.geometry import SMPL_PARENTS, matrix_to_rotation6d, rotation6d_to_matrix
from unified_hoi.parametric.body import (BodyGeometry, beta_count, decode_full, human_dim,
                                         legacy_to_parameters, split_human, standard_parameters)
from unified_hoi.parametric.contact import ContactAE, load_codec
from unified_hoi.parametric.data import ParametricDataset, load_record
from unified_hoi.parametric.diffusion import MODES, ParametricDiffusion, controls_for
from unified_hoi.parametric.evaluate import contact_scores, mesh_contacts, reconstruction, similarity
from unified_hoi.parametric.model import ParametricConfig, ParametricDenoiser
from unified_hoi.parametric.train import prepare_batch

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class ProtocolTests(unittest.TestCase):
    def test_similarity_and_contact_statistics(self):
        rng = np.random.default_rng(12)
        points = rng.normal(size=(73, 3))
        rotation = rotation6d_to_matrix(torch.tensor(rng.normal(size=6))).numpy()
        target = 1.7 * points @ rotation.T + [1, -2, 3]
        np.testing.assert_allclose(similarity(points, target), target, atol=1e-12)
        s = contact_scores([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(s["accuracy_percent"], 50.)
        self.assertEqual(s["f1"], .5)

    def test_contact_frame_and_threshold(self):
        rotation = rotation6d_to_matrix(torch.tensor([[.3, .8, .2, -.7, .1, .4]])).numpy()
        trans = np.array([[2., -1., 3.]])
        local = np.array([[[.01, 0, 0], [0, .06, 0]]])
        body = local @ rotation[0].T + trans[:, None]
        pose = np.concatenate((trans, rotation[:, :2].reshape(1, 6)), -1)
        np.testing.assert_array_equal(mesh_contacts(body, pose, np.zeros((1, 3))), [[True, False]])

    def test_mode_and_dataset_masks(self):
        for dataset in ("behave", "omomo"):
            batch = {"human": torch.zeros(2, 3, human_dim(dataset)), "object": torch.zeros(2, 3, 9),
                     "contact": torch.zeros(2, 3, 128)}
            for mode in MODES:
                masks = controls_for(batch, dataset, mode)
                for key, bit in zip(batch, mode):
                    mask = masks[key]
                    if key == "human" and dataset == "omomo":
                        n = beta_count(dataset)
                        self.assertTrue(mask[..., n + 132:n + 312].all())
                        mask = torch.cat((mask[..., :n + 132], mask[..., -3:]), -1)
                    self.assertTrue((mask == (bit == "0")).all())
        with self.assertRaises(ValueError):
            controls_for(batch, "omomo", "mixed")

    def test_latent_is_continuous(self):
        codec = ContactAE()
        z = codec.encode(torch.randint(0, 2, (2, 3, 690)).float())
        self.assertEqual(z.shape, (2, 3, 128))
        self.assertTrue((z < 0).any())
        p = codec.decode(z)
        self.assertEqual(p.shape, (2, 3, 690))
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_mixed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.jsonl"
            path.write_text("\n".join(json.dumps({"dataset": d, "path": "missing.npz", "split": "train",
                                                  "sequence_id": d, "fps": 10})
                                      for d in ("behave", "omomo")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Mixed datasets"):
                ParametricDataset(path, "train", "behave", "bad")


class NativeAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = ROOT / "data/parametric_assets"
        if not (cls.assets / "behave/contact_support.npz").exists():
            raise unittest.SkipTest("Prepare native assets to run exact LBS integration checks")
        cls.geometry = {name: BodyGeometry(name, cls.assets / name / "contact_support.npz")
                        for name in ("behave", "omomo")}

    def test_exact_selected_skinning_all_models_and_gradients(self):
        generator = torch.Generator().manual_seed(43)
        for dataset, geometry in self.geometry.items():
            with np.load(self.assets / dataset / "contact_support.npz") as data:
                indices = data["indices"]
            for gender in ("male", "female"):
                rotations = rotation6d_to_matrix(torch.randn(2, 52, 6, generator=generator))
                h = torch.cat((torch.randn(2, beta_count(dataset), generator=generator) * .2,
                               matrix_to_rotation6d(rotations).flatten(1),
                               torch.randn(2, 3, generator=generator)), -1).requires_grad_()
                vertices, joints = geometry.layers[gender](h)
                full, _, full_joints = decode_full(h.detach(), dataset, gender)
                params = standard_parameters(h.detach(), dataset)
                self.assertEqual(params["body_pose"].shape, (2, 63))
                self.assertEqual(params["left_hand_pose"].shape, (2, 45))
                from scipy.spatial.transform import Rotation
                rotvec = np.concatenate((params["global_orient"], params["body_pose"],
                                         params["left_hand_pose"], params["right_hand_pose"]), -1).reshape(-1, 3)
                np.testing.assert_allclose(Rotation.from_rotvec(rotvec).as_matrix().reshape(2, 52, 3, 3),
                                           rotations, atol=2e-6, rtol=0)
                np.testing.assert_allclose(vertices.detach(), full[:, indices], atol=2e-6, rtol=0)
                np.testing.assert_allclose(joints.detach(), full_joints, atol=2e-6, rtol=0)
                (vertices.square().mean() + joints.square().mean()).backward()
                self.assertTrue(torch.isfinite(h.grad).all())
                self.assertGreater(float(h.grad.abs().sum()), 0)

    def test_support_and_codec_isolation(self):
        with self.assertRaisesRegex(ValueError, "Cross-dataset"):
            BodyGeometry("omomo", self.assets / "behave/contact_support.npz")
        codec = self.assets / "behave/contact_ae/best.pt"
        if codec.exists():
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_codec(codec, "omomo", self.geometry["behave"].support_hash)

    def test_legacy_adapter_preserves_body_and_uses_declared_hands(self):
        for dataset, geometry in self.geometry.items():
            from unified_hoi.data import read_manifest
            # Build a legacy-format fixture from native data so this regression
            # does not require retaining obsolete training datasets.
            entries = read_manifest(ROOT / f"data/processed/{dataset}_parametric.jsonl")
            native = load_record(entries[0]["path"], dataset, geometry.support_hash)
            human = torch.from_numpy(native["human_params"][:3])
            beta, pose, _ = split_human(human, dataset)
            local = rotation6d_to_matrix(pose)
            globals_ = [local[:, 0]]
            for joint in range(1, 22):
                globals_.append(globals_[SMPL_PARENTS[joint]] @ local[:, joint])
            with torch.no_grad():
                _, original_joints = geometry.layers[str(native["gender"])](human)
            legacy_human = torch.cat((original_joints[:, :22],
                                      matrix_to_rotation6d(torch.stack(globals_, 1))), -1)
            record = {"human": legacy_human.numpy(), "betas": beta[0].numpy(), "gender": native["gender"]}
            hands = local[0, 22:].numpy()
            params = legacy_to_parameters(record, geometry, hands)
            _, joints = geometry.layers[str(record["gender"])](torch.from_numpy(params))
            np.testing.assert_allclose(joints[:, :22].detach(), record["human"][..., :3], atol=2e-5, rtol=0)
            n = beta_count(dataset)
            decoded = rotation6d_to_matrix(torch.from_numpy(params[:, n + 132:n + 312].reshape(3, 30, 6)))
            np.testing.assert_allclose(decoded, np.broadcast_to(hands, (3, 30, 3, 3)), atol=2e-6)

    def _batch(self):
        manifest = ROOT / "data/processed/behave_parametric.jsonl"
        if not manifest.exists():
            self.skipTest("Prepare BEHAVE native data first")
        geometry = self.geometry["behave"]
        ds = ParametricDataset(manifest, "val", "behave", geometry.support_hash, window=3, text_condition=False)
        model = ParametricDenoiser(ParametricConfig(width=24, heads=4, layers=1, dropout=0.))
        batch = prepare_batch(model, next(iter(DataLoader(ds, batch_size=1))), geometry, 8)
        return model, batch, ds

    def test_sample_exact_anchors_and_no_generated_gt_leak(self):
        model, batch, _ = self._batch()
        diffusion = ParametricDiffusion(8)
        for mode in MODES:
            masks = controls_for(batch, "behave", mode)
            first = diffusion.sample(model, batch, masks, steps=2, seed=123)
            perturbed = dict(batch)
            for k in masks:
                perturbed[k] = torch.where(masks[k], batch[k], batch[k] + 17.)
            second = diffusion.sample(model, perturbed, masks, steps=2, seed=123)
            for k in masks:
                torch.testing.assert_close(first[k][masks[k]], batch[k][masks[k]], rtol=0, atol=0)
                torch.testing.assert_close(first[k], second[k], rtol=0, atol=0)

    def test_surface_loss_reaches_generated_object(self):
        model, batch, _ = self._batch()
        masks = controls_for(batch, "behave", "010")
        diffusion = ParametricDiffusion(8)
        loss, metrics = diffusion.training_loss(model, batch, masks, self.geometry["behave"],
                                               {"geometry_max_t": 8, "geometry_frames": 2})
        self.assertGreater(metrics["object_v2v"], 0.)
        self.assertGreater(metrics["surface_distance"], 0.)
        loss.backward()
        grad = model.object_head.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.)
        self.assertIsNone(next(model.codec.parameters()).grad)

    def test_metrics_identity_translation_and_corrupt_provenance(self):
        from unified_hoi.mesh_geometry import load_mesh_asset
        model, batch, ds = self._batch()
        entry = ds.records[int(batch["record_index"][0])]
        vertices, _ = load_mesh_asset(str(Path(entry["object_mesh_path"]).resolve()))
        vertices = vertices * entry["object_mesh_scale"]
        geometry = self.geometry["behave"]
        states = {key: batch[key].clone() for key in ("human", "object", "contact")}
        score = reconstruction(states, batch, geometry, "110", vertices)["metrics"]
        for key in ("MPJPE", "MPJPE_PA", "OBJ_CENTER", "OBJ_V2V"):
            self.assertLess(score[key], 1e-10)
        self.assertEqual(score["OBJ_CONTACT_MESHES"], 100.)
        states["object"][..., :3] += torch.tensor([.3, .4, 0])
        score = reconstruction(states, batch, geometry, "010", vertices)["metrics"]
        self.assertAlmostEqual(score["OBJ_V2V"], .5, places=6)
        self.assertAlmostEqual(score["OBJ_CENTER"], .5, places=6)
        with self.assertRaisesRegex(ValueError, "Observed object"):
            reconstruction(states, batch, geometry, "100", vertices)
        batch["body_model_sha256"] = ["corrupt"]
        with self.assertRaisesRegex(ValueError, "different body assets"):
            prepare_batch(model, batch, geometry, 8)

    def test_native_record_rejects_wrong_support(self):
        files = sorted((ROOT / "data/processed/behave_parametric").glob("*.npz"))
        if not files:
            self.skipTest("Prepare native data first")
        with self.assertRaisesRegex(ValueError, "support mismatch"):
            load_record(files[0], "behave", "wrong")

    def test_stored_contacts_match_evaluation_both_datasets(self):
        from unified_hoi.data import read_manifest
        from unified_hoi.mesh_geometry import load_mesh_asset
        for dataset, geometry in self.geometry.items():
            manifest = ROOT / f"data/processed/{dataset}_parametric.jsonl"
            if not manifest.exists():
                self.skipTest("Prepare both native manifests first")
            entries = read_manifest(manifest)
            for split in ("train", "val", "test"):
                entry = next(e for e in entries if e["split"] == split)
                record = load_record(entry["path"], dataset, geometry.support_hash)
                human = torch.from_numpy(record["human_params"][:3])
                with torch.no_grad():
                    body, _ = geometry.layers[str(record["gender"])](human)
                obj, _ = load_mesh_asset(str(Path(entry["object_mesh_path"]).resolve()))
                detected = mesh_contacts(body.numpy(), record["object"][:3], obj * entry["object_mesh_scale"])
                np.testing.assert_array_equal(detected, record["surface_contact"][:3])


if __name__ == "__main__":
    unittest.main()
