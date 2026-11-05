"""EMA, temporal padding, independent budgets and native evaluator contracts."""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from unified_hoi.data import read_manifest
from unified_hoi.token_ar.body import digest_file
from unified_hoi.geometry import matrix_to_rotation6d, rotation6d_to_matrix
from unified_hoi.token_ar.canonical import restore_scene, to_canonical
from unified_hoi.token_ar.codecs import CodecBundle, CodecConfig, MotionVQVAE, codec_payload, load_codecs
from unified_hoi.token_ar.vq_training import _cpu_module_state
from unified_hoi.token_ar.embedding_metrics import embedding_scores, frechet_distance
from unified_hoi.token_ar.evaluate_codecs import evaluate_codecs
from unified_hoi.token_ar.motion_text import EvaluatorConfig, MotionTextEvaluator, load_evaluator
from unified_hoi.token_ar import train_codecs, train_evaluator


class TemporalEMATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)

    def test_ema_arithmetic_and_gradients(self):
        config = CodecConfig(dataset="behave", network="residual", quantizer="ema", width=8,
                             codebook_size=2, latent_dim=2, ema_decay=.5, ema_reset_threshold=0.)
        codec = MotionVQVAE(3, config)
        codec.ema_count.fill_(1)
        codec.ema_sum.copy_(torch.tensor([[2., 4.], [8., 10.]]))
        codec._update_ema(torch.tensor([[4., 6.], [6., 8.]]), torch.tensor([0, 0]), torch.tensor([2., 0.]))
        torch.testing.assert_close(codec.ema_count, torch.tensor([1.5, .5]))
        torch.testing.assert_close(codec.codebook.weight, torch.tensor([[4., 6.], [8., 10.]]))
        loss, _ = codec.loss(torch.randn(2, 7, 3), torch.ones(2, 7, dtype=torch.bool))
        loss.backward()
        self.assertIsNone(codec.codebook.weight.grad)
        for module in (codec.encoder, codec.decoder):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)

    def test_padding_eval_freeze_and_declared_book_shape(self):
        config = CodecConfig(dataset="behave", network="residual", quantizer="ema", width=8,
                             codebook_size=512, latent_dim=4096)
        codec = MotionVQVAE(3, config).train()
        self.assertEqual(tuple(codec.codebook.weight.shape), (512, 4096))
        values = torch.randn(3, 7, 3)
        valid = torch.arange(7)[None] < torch.tensor([7, 3, 1])[:, None]
        loss, _ = codec.loss(values, valid)
        loss.backward()
        codec.eval()
        snapshot = {k: v.clone() for k, v in codec.state_dict().items()}
        first, _ = codec.loss(values, valid)
        tokens = codec.encode(values, valid)
        self.assertEqual([len(t) for t in tokens], [4, 2, 1])
        for i, n in enumerate((7, 3, 1)):
            self.assertEqual(tuple(codec.decode(tokens[i], n).shape), (n, 3))
        values[~valid] = 1e6
        second, _ = codec.loss(values, valid)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for k, v in codec.state_dict().items():
            torch.testing.assert_close(v, snapshot[k], rtol=0, atol=0)

    def test_scene_anchor_roundtrip_and_geometric_loss(self):
        identity = torch.tensor([1., 0., 0., 0., 1., 0.])
        yaw = torch.tensor([.7, -1.2])
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        zeros, ones = torch.zeros_like(cosine), torch.ones_like(cosine)
        rotation = torch.stack((
            torch.stack((cosine, zeros, sine), -1),
            torch.stack((zeros, ones, zeros), -1),
            torch.stack((-sine, zeros, cosine), -1),
        ), -2)
        pose = identity.view(1, 1, 1, 6).expand(2, 6, 52, 6).clone()
        pose[:, :, 0] = matrix_to_rotation6d(rotation)[:, None]
        human = torch.cat((torch.zeros(2, 6, 10), pose.flatten(2), torch.randn(2, 6, 3)), -1)
        obj = torch.cat((torch.randn(2, 6, 3), identity.view(1, 1, 6).expand(2, 6, 6).clone()), -1)
        canonical_human, canonical_obj, anchor = to_canonical(human, obj, "behave")
        restored_human, restored_obj = restore_scene(canonical_human, canonical_obj, anchor, "behave")
        torch.testing.assert_close(restored_human, human, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(restored_obj, obj, atol=1e-4, rtol=1e-4)
        self.assertLess(canonical_human[:, 0, -3].abs().max(), 1e-4)
        self.assertLess(canonical_human[:, 0, -1].abs().max(), 1e-4)
        root = rotation6d_to_matrix(canonical_human[:, 0, 10:16])
        self.assertLess(root[:, 0, 2].abs().max(), 1e-4)
        config = CodecConfig(dataset="behave", network="residual", quantizer="ema", width=8,
                             codebook_size=4, latent_dim=4, canonical_scene=True, commitment_weight=.02,
                             velocity_weight=.5)
        codec = MotionVQVAE(325, config, role="human")
        loss, detail = codec.loss(human, torch.ones(2, 6, dtype=torch.bool))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(detail["root_deg"]), 0)
        self.assertIsNone(codec.codebook.weight.grad)
        heavier = CodecConfig(dataset="behave", network="residual", quantizer="ema", width=8,
                              codebook_size=4, latent_dim=4, canonical_scene=True, body_weight=1.)
        other = MotionVQVAE(325, heavier, role="human")
        other.load_state_dict(codec.state_dict())
        values = torch.ones(2, 6, dtype=torch.bool)
        loss_low, _ = codec.loss(human, values)
        loss_high, _ = other.loss(human, values)
        self.assertGreater(float(loss_high.detach()), float(loss_low.detach()))
        optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3)
        probe, _ = codec.loss(human, values)
        probe.backward()
        optimizer.step()
        live = next(iter(optimizer.state.values()))["exp_avg"]
        saved = next(iter(_cpu_module_state(codec, optimizer)["optimizer"]["state"].values()))["exp_avg"]
        self.assertIsNot(live, saved)
        live.add_(1)
        self.assertFalse(torch.equal(live.cpu(), saved))
        contact = MotionVQVAE(4, config, contact=True)
        labels = torch.randint(0, 2, (2, 6, 4), dtype=torch.float32)
        _, contact_detail = contact.loss(labels, values)
        self.assertTrue(0 <= float(contact_detail["f1"]) <= 1)

    def test_legacy_checkpoint_without_new_fields(self):
        root = Path("runs").resolve()
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            codec = CodecBundle(CodecConfig(dataset="behave", width=8, latent_dim=4, codebook_size=4)).eval()
            saved = codec_payload(codec, {})
            for field in ("network", "quantizer", "depth", "dilation_growth", "ema_decay", "ema_reset_threshold"):
                saved["codec_config"].pop(field)
            path = Path(directory) / "legacy.pt"
            torch.save(saved, path)
            restored, _ = load_codecs(path)
            for key, value in codec.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


class EmbeddingTests(unittest.TestCase):
    def test_figure_table_uses_fid_rprecision_and_diversity(self):
        from unified_hoi.token_ar.evaluate_motion_text import figure_table
        def row(fid, diversity):
            return {"reconstructed": {"FID": fid, "Diversity": diversity,
                                      "R_Precision": {"top1": .1, "top2": .2, "top3": .3}},
                    "ground_truth": {"Diversity": 1.4}}
        scores = {name: row(fid, 1.1 + fid) for name, fid in
                  (("human", .4), ("object", .2), ("contact", .5), ("joint", .3))}
        table = figure_table(scores, "joint")
        self.assertEqual(table["view"], "joint")
        self.assertEqual(table["FID"], .3)
        self.assertEqual(set(table["R_Precision"]), {"top1", "top2", "top3"})
        self.assertEqual(table["Diversity_real"], 1.4)
        self.assertEqual(table["views"]["human"]["FID"], .4)
        self.assertLess(table["FID"], table["views"]["contact"]["FID"])

    def test_fid_and_retrieval_controls(self):
        real = np.eye(8)
        self.assertLess(frechet_distance(real, real), 1e-7)
        self.assertAlmostEqual(frechet_distance(real, real + 2), 32., places=6)
        kwargs = dict(source_ids=list(range(8)), captions=list(range(8)), repeats=3)
        perfect = embedding_scores(real, real, real, **kwargs)
        self.assertEqual(perfect["reconstructed"]["R_Precision"]["top1"], 1.)
        self.assertEqual(perfect["reconstructed"]["Matching_Score"], 0.)
        self.assertEqual(perfect["protocol"]["effective_candidates"], 8)
        shuffled = embedding_scores(real, np.roll(real, 1, 0), real, **kwargs)
        self.assertEqual(shuffled["reconstructed"]["R_Precision"]["top1"], 0.)
        self.assertLess(shuffled["reconstructed"]["FID"], 1e-7)  # Distribution is not pairing.

    def test_ties_not_perfect_and_same_control_protocol(self):
        zero = np.zeros((32, 4))
        scores = embedding_scores(zero, zero, zero, source_ids=list(range(32)), captions=["same"] * 32)
        real, predicted = scores["ground_truth"], scores["reconstructed"]
        self.assertEqual(real, predicted)
        self.assertLess(real["R_Precision"]["top1"], .1)
        self.assertEqual(real["semantic_R_Precision"]["top1"], 1.)

    def test_motion_encoder_ignores_padding(self):
        torch.set_num_threads(2)
        model = MotionTextEvaluator(EvaluatorConfig(dataset="behave", width=8, embedding_dim=4, layers=1)).eval()
        batch = {"human": torch.randn(2, 7, 325), "fps": torch.tensor([10., 10.]),
                 "valid_frames": torch.arange(7)[None] < torch.tensor([7, 3])[:, None]}
        first = model.encode_motion(batch, "human")
        batch["human"][1, 3:] = 1e6
        torch.testing.assert_close(first, model.encode_motion(batch, "human"), rtol=0, atol=0)


@unittest.skipUnless(Path("data/processed/behave_parametric.jsonl").exists(), "Requires native assets and text caches")
class NativeUpgradeTests(unittest.TestCase):
    def test_budgets_resume_evaluator_and_roundtrip(self):
        torch.set_num_threads(2)
        root = Path("runs").resolve()
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vq_upgrade_", dir=root) as directory:
            folder = Path(directory)
            rows = read_manifest(Path("data/processed/behave_parametric.jsonl").resolve())
            picked, used = [], set()
            for row in rows:
                source = row.get("source_sequence_id", row["sequence_id"])
                if row["split"] == "train" and source not in used:
                    picked.append(row)
                    used.add(source)
                if len(picked) == 3:
                    break
            picked.append(next(r for r in rows if r["split"] == "val"))
            for i, row in enumerate(picked):
                with np.load(row["path"], allow_pickle=False) as archive:
                    record = dict(archive)
                frames = len(record["human_params"])
                self.assertGreaterEqual(frames, 8)
                for key, value in record.items():
                    if value.ndim and value.shape[0] == frames:
                        record[key] = value[:8].copy()
                path = folder / f"native{i}.npz"
                np.savez(path, **record)
                row.update(path=str(path), frames=8)
            manifest = folder / "manifest.jsonl"
            unlabelled = dict(picked[-1], sequence_id="unlabelled_val", source_sequence_id="unlabelled_val",
                              text="", text_variants=[], text_features_path=None, text_encoder=None)
            manifest.write_text("\n".join(json.dumps(r) for r in [*picked, unlabelled]), encoding="utf-8")
            support = "data/parametric_assets/behave/contact_support.npz"
            common = dict(dataset="behave", manifest=str(manifest), contact_support=support,
                          seed=17, cpu_threads=2, window=4, stride=4, device="cpu",
                          validate_every=1, log_every=1)
            eval_config = dict(common, output=str(folder / "evaluator"), batch_size=3, max_steps=1,
                               model=dict(width=8, embedding_dim=4, layers=1))
            with contextlib.redirect_stdout(io.StringIO()):
                train_evaluator.run(eval_config)
                eval_config.update(max_steps=3)
                eval_path = folder / "evaluator/last.pt"
                train_evaluator.run(eval_config, eval_path)
                train_evaluator.run(dict(eval_config, output=str(folder / "eval_full")))
            resumed_eval = torch.load(eval_path, weights_only=True)
            full_eval = torch.load(folder / "eval_full/last.pt", weights_only=True)
            for key, value in full_eval["model"].items():
                torch.testing.assert_close(value, resumed_eval["model"][key], rtol=0, atol=0)
            frozen, _ = load_evaluator(eval_path, "behave", digest_file(support), [picked[-1]])
            self.assertFalse(frozen.training)
            self.assertFalse(any(p.requires_grad for p in frozen.parameters()))
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_evaluator(eval_path, "behave", digest_file(support), [picked[0]])
            config = dict(common, architecture="token_ar_v1", output=str(folder / "vq"),
                          batch_size=2, max_steps=1, validation_batches=1,
                          modality_steps=dict(human=3, object=1, contact=2),
                          motion_text_evaluator=str(eval_path),
                          codec=dict(width=8, codebook_size=4, latent_dim=4, network="residual", quantizer="ema"))
            snapshots = []
            vq_path = folder / "vq/last.pt"
            with contextlib.redirect_stdout(io.StringIO()):
                for maximum in (1, 2, 3):
                    config.update(max_steps=maximum)
                    train_codecs.run(config, vq_path if maximum > 1 else None)
                    snapshots.append(torch.load(vq_path, weights_only=True))
                train_codecs.run(dict(config, output=str(folder / "vq_full")))
                before_eval = digest_file(eval_path)
                before_vq = digest_file(vq_path)
                report = evaluate_codecs(vq_path, folder / "report", window=4, candidates=32, repeats=2, diversity_pairs=3)
            final = snapshots[-1]
            self.assertEqual(final["modality_updates"], dict(human=3, object=1, contact=2))
            self.assertFalse((folder / "vq/best.pt").exists())
            full = torch.load(folder / "vq_full/last.pt", weights_only=True)
            for key, value in full["model"].items():
                torch.testing.assert_close(value, final["model"][key], rtol=0, atol=0)
                for modality, snapshot in (("object", snapshots[0]), ("contact", snapshots[1])):
                    if key.startswith(f"models.{modality}."):
                        torch.testing.assert_close(value, snapshot["model"][key], rtol=0, atol=0)
            self.assertEqual(digest_file(eval_path), before_eval)
            self.assertEqual(digest_file(vq_path), before_vq)
            self.assertEqual(report["evaluated_windows"], 2)
            self.assertEqual(report["evaluated_native_frames"], 8)
            self.assertEqual(report["excluded_unlabelled_sequences"], ["unlabelled_val"])
            self.assertEqual(report["excluded_unlabelled_windows"], 2)
            self.assertEqual(set(report["embedding_metrics"]), {"human", "object", "contact", "joint"})
            for view in report["embedding_metrics"].values():
                self.assertTrue(np.isfinite(view["reconstructed"]["FID"]))
            self.assertEqual(asdict(frozen.config)["dataset"], "behave")


if __name__ == "__main__":
    unittest.main()
