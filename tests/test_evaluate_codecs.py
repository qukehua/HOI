"""VQ oracle evaluation contracts; random tiny codecs do not test trained quality."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from unified_hoi.data import read_manifest
from unified_hoi.token_ar import MODALITIES, MODES
from unified_hoi.token_ar.body import digest_file
from unified_hoi.token_ar.codecs import CodecBundle, CodecConfig, codec_payload
from unified_hoi.token_ar import evaluate_codecs as evaluator


class CodecDiagnosticTests(unittest.TestCase):
    def test_corpus_usage(self):
        stats = evaluator.usage_statistics([2, 2, 0, 0])
        self.assertEqual(stats["tokens"], 4)
        self.assertEqual(stats["used_codes"], 2)
        self.assertEqual(stats["usage_fraction"], .5)
        self.assertAlmostEqual(stats["perplexity"], 2.)

    def test_encode_excludes_padding_even_for_short_odd_windows(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        codec = CodecBundle(CodecConfig(dataset="behave", codebook_size=4, latent_dim=8, width=16)).eval()
        batch = {k: torch.randn(2, 7, len(v.mean)) for k, v in codec.models.items()}
        batch["valid_frames"] = torch.arange(7)[None] < torch.tensor([7, 3])[:, None]
        first = codec.encode(batch)
        for k in MODALITIES:
            batch[k][1, 3:] = 1e6
        second = codec.encode(batch)
        for key in MODALITIES:
            self.assertEqual([len(ids) for ids in first[key]], [4, 2])
            for a, b in zip(first[key], second[key]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


@unittest.skipUnless(Path("data/processed/behave_parametric.jsonl").exists(), "Requires licensed native assets")
class OracleIntegrationTests(unittest.TestCase):
    def test_real_roundtrip_and_identity_control(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        root = Path("runs").resolve()
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vq_eval_test_", dir=root) as directory:
            folder = Path(directory)
            self.assertTrue(folder.is_relative_to(root))
            entry = next(r for r in read_manifest("data/processed/behave_parametric.jsonl") if r["split"] == "val")
            with np.load(entry["path"], allow_pickle=False) as archive:
                record = dict(archive)
            frames = len(record["human_params"])
            self.assertGreaterEqual(frames, 23)
            for key, value in record.items():
                if value.ndim and value.shape[0] == frames:
                    record[key] = value[:23].copy()
            np.savez(folder / "native.npz", **record)
            entry.update(path=str(folder / "native.npz"), frames=23,
                         object_mesh_path=str(Path(entry["object_mesh_path"]).resolve()))
            manifest = folder / "manifest.jsonl"
            manifest.write_text(json.dumps(entry), encoding="utf-8")
            support = "data/parametric_assets/behave/contact_support.npz"
            codec = CodecBundle(CodecConfig(dataset="behave", codebook_size=4, latent_dim=8, width=16)).eval()
            metadata = dict(config=dict(manifest=str(manifest), contact_support=support),
                            support_sha256=digest_file(support), manifest_sha256=digest_file(manifest), step=17)
            checkpoint = folder / "codecs.pt"
            torch.save(codec_payload(codec, metadata), checkpoint)
            checkpoint_hash = digest_file(checkpoint)
            original_reconstruction = evaluator.reconstruction
            calls = []

            def checked_reconstruction(states, target, geometry, mode, *args):
                for key, bit in zip(MODALITIES, mode):
                    if bit == "0":
                        expected = target["surface_contact" if key == "contact" else key]
                        torch.testing.assert_close(states[key], expected, rtol=0, atol=0)
                calls.append(mode)
                return original_reconstruction(states, target, geometry, mode, *args)

            with patch.object(evaluator, "reconstruction", side_effect=checked_reconstruction), \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = evaluator.evaluate_codecs(checkpoint, folder / "random", window=13, modes=MODES)
            self.assertEqual(len(calls), 2 * len(MODES))
            self.assertEqual(summary["evaluated_frames"], 3)  # 0, 10, 20: no window-local fps reset.
            self.assertEqual(summary["processed_windows"], 2)
            self.assertEqual(summary["checkpoint_step"], 17)
            self.assertTrue(summary["manifest_matches_training"])
            for stats in summary["token_usage"].values():
                self.assertEqual(stats["tokens"], 12)  # ceil(13/2) + ceil(10/2), no padded tokens.
            per_clip = [json.loads(line) for line in (folder / "random/per_clip.jsonl").read_text().splitlines()]
            for mode in MODES:
                rows = [r for r in per_clip if r["mode"] == mode]
                counts = {k: sum(r["details"]["CONTACT_TOKEN"][k] for r in rows) for k in ("TP", "FP", "FN", "TN")}
                tp, fp, fn, tn = (counts[k] for k in ("TP", "FP", "FN", "TN"))
                self.assertEqual(tp + fp + fn + tn, 3 * 690)
                self.assertAlmostEqual(summary["reconstruction"][mode]["CONTACT_TOKEN_F1"],
                                       2 * tp / max(2 * tp + fp + fn, 1))
            self.assertEqual(digest_file(checkpoint), checkpoint_hash)
            # A lossless control must produce zero geometric error. This catches target/mask/coordinate errors.
            class IdentityMotion:
                def decode(self, indices, length):
                    return self.values[:length].clone()

            class IdentityBundle:
                config = codec.config
                models = {key: IdentityMotion() for key in MODALITIES}

                def encode(self, batch):
                    n = int(batch["valid_frames"].sum())
                    for key in MODALITIES:
                        self.models[key].values = batch[key][0, :n]
                    return {key: [torch.zeros((n + 1) // 2, dtype=torch.long)] for key in MODALITIES}

            with patch.object(evaluator, "load_codecs", return_value=(IdentityBundle(), metadata)), \
                    contextlib.redirect_stdout(io.StringIO()):
                perfect = evaluator.evaluate_codecs(checkpoint, folder / "identity", window=13)
            self.assertEqual(perfect["evaluation_frames_sha256"], summary["evaluation_frames_sha256"])
            self.assertEqual(perfect["reference_values_sha256"], summary["reference_values_sha256"])
            for metric in ("MPJPE", "MPJPE_PA", "HUMAN_TRANSL_L2", "HUMAN_V2V_690"):
                self.assertLess(perfect["reconstruction"]["100"][metric], 1e-5)
            for metric in ("OBJ_V2V", "OBJ_CENTER"):
                self.assertLess(perfect["reconstruction"]["010"][metric], 1e-5)
            native = perfect["contact_diagnostics"]["VQ_VS_NATIVE"]
            self.assertEqual(native["FP"] + native["FN"], 0)
            self.assertEqual(native["accuracy_percent"], 100.)
            with self.assertRaises(FileExistsError):
                evaluator.evaluate_codecs(checkpoint, folder / "random")
            with self.assertRaisesRegex(ValueError, "support mismatch"):
                evaluator.evaluate_codecs(checkpoint, folder / "wrong_support",
                                          contact_support="data/parametric_assets/omomo/contact_support.npz")


if __name__ == "__main__":
    unittest.main()
