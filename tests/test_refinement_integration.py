"""Offline integration with real native assets and a random tiny Qwen (not quality tests)."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from unified_hoi.data import read_manifest
from unified_hoi.token_ar import ARCHITECTURE
from unified_hoi.token_ar.body import BodyGeometry, digest_file
from unified_hoi.token_ar.codecs import CodecBundle, CodecConfig, codec_payload
from unified_hoi.token_ar.model import build_model
from unified_hoi.token_ar.prepare_refinement import prepare
from unified_hoi.token_ar.refinement import KIND, ContinuousRefiner, RefinerConfig, refinement_loss
from unified_hoi.token_ar.refinement_data import RefinementDataset
from unified_hoi.token_ar.train_refiner import run


@unittest.skipUnless(Path("data/processed/behave_parametric.jsonl").exists(), "Requires local licensed native assets")
class IntegrationTests(unittest.TestCase):
    def test_cache_train_resume_sample_benchmark(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from unified_hoi.token_ar.benchmark import benchmark
        from unified_hoi.token_ar.sample import sample
        from unified_hoi.token_ar.evaluate import evaluate

        torch.set_num_threads(2)
        torch.manual_seed(11)
        root = Path("runs").resolve()
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="refiner_test_", dir=root) as directory:
            folder = Path(directory).resolve()
            self.assertTrue(folder.is_relative_to(root))
            rows = read_manifest("data/processed/behave_parametric.jsonl")
            entries = [next(r for r in rows if r["split"] == split) for split in ("train", "val")]
            for entry in entries:
                for key in ("path", "object_mesh_path", "text_features_path"):
                    if entry.get(key):
                        entry[key] = str(Path(entry[key]).resolve())
            manifest = folder / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(r) for r in entries), encoding="utf-8")
            support = "data/parametric_assets/behave/contact_support.npz"
            config = dict(dataset="behave", manifest=str(manifest), contact_support=support,
                          window=4, stride=4, load_dtype="float32", max_sequence_tokens=256)
            codec = CodecBundle(CodecConfig(dataset="behave", codebook_size=16, latent_dim=8, width=16))
            backbone = dict(model_type="qwen3", vocab_size=32, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
            model, _ = build_model(config, codec.config, backbone)
            checkpoint = folder / "ar"
            checkpoint.mkdir()
            torch.save(codec_payload(codec, dict(support_sha256=digest_file(support),
                                                 manifest_sha256=digest_file(manifest))), checkpoint / "codecs.pt")
            torch.save(model.state_dict(), checkpoint / "pytorch_model_fsdp.bin")
            core = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "Dataset": 2}, unk_token="[UNK]"))
            core.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]")
            tokenizer.save_pretrained(checkpoint / "tokenizer")
            metadata = dict(architecture=ARCHITECTURE, config=config, codec_config=vars(codec.config),
                            support_sha256=digest_file(support), manifest_sha256=digest_file(manifest),
                            codec_sha256=digest_file(checkpoint / "codecs.pt"),
                            backbone_config=model.backbone.config.to_dict(), step=1)
            (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (checkpoint / "COMPLETE").write_text(ARCHITECTURE)
            cache = folder / "cache"
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(checkpoint, cache, device="cpu", dtype="float32", limit=1,
                        modes=("100", "010", "001", "111"))
                index_hash = digest_file(cache / "index.jsonl")
                prepare(checkpoint, cache, device="cpu", dtype="float32", limit=1,
                        modes=("100", "010", "001", "111"), resume=True)
            self.assertEqual(index_hash, digest_file(cache / "index.jsonl"))
            self.assertEqual(len(RefinementDataset(cache, "train")), 4)
            cfg = dict(kind=KIND, cache=str(cache), output=str(folder / "refiner"), device="cpu",
                       model=dict(width=16, layers=1), max_steps=1, schedule_steps=4, warmup_steps=0,
                       gradient_accumulation_steps=1, validate_every=1, log_every=1, geometry_frames=2,
                       early_stopping_patience=0)
            with contextlib.redirect_stdout(io.StringIO()):
                run(cfg)
                run(cfg | {"max_steps": 2}, resume=folder / "refiner/last.pt")
                run(cfg | {"max_steps": 2, "output": str(folder / "uninterrupted")})
            resumed = torch.load(folder / "refiner/last.pt", weights_only=True)
            uninterrupted = torch.load(folder / "uninterrupted/last.pt", weights_only=True)
            for key, value in resumed["model"].items():
                torch.testing.assert_close(value, uninterrupted["model"][key], atol=0, rtol=0)
            refined = folder / "prediction.npz"
            sample(checkpoint, manifest, entries[1]["path"], refined, "010", frames=4, text_conditioned=True,
                   temperature=0., refiner=folder / "refiner/last.pt")
            with np.load(refined) as a:
                self.assertIn("refiner_sha256", a)
                np.testing.assert_array_equal(a["human_params"], a["human_observed"])
                np.testing.assert_array_equal(a["contact_probability"], a["contact_observed"])
            report = evaluate(refined, entries[1]["path"], manifest, checkpoint, folder / "evaluate.json", eval_fps=0.)
            self.assertEqual(report["sampler"], "autoregressive_vq_continuous_refinement")
            with contextlib.redirect_stdout(io.StringIO()):
                scored = benchmark(checkpoint, folder / "benchmark", split="val", modes=("010", "001"),
                                   device="cpu", dtype="float32", eval_fps=0., train_reference_fps=0., limit=1,
                                   refiner=folder / "refiner/last.pt")
                baseline = benchmark(checkpoint, folder / "baseline", split="val", modes=("010", "001"),
                                     device="cpu", dtype="float32", eval_fps=0., train_reference_fps=0., limit=1)
            self.assertEqual(scored["evaluation_frames_sha256"], baseline["evaluation_frames_sha256"])
            self.assertEqual(scored["coarse_reconstruction"], baseline["reconstruction"])
            self.assertEqual(scored["evaluated_frames"], 4)
            self.assertTrue(np.isfinite(scored["reconstruction"]["010"]["OBJ_V2V"]))

    def test_real_omomo_body_gradients(self):
        from unified_hoi.token_ar.data import MotionDataset
        from unified_hoi.mesh_geometry import load_mesh_asset
        torch.set_num_threads(2)
        support = "data/parametric_assets/omomo/contact_support.npz"
        rows = read_manifest("data/processed/omomo_parametric.jsonl")
        row = next(r for r in rows if r["split"] == "val")
        # Load just one record via a small temporary manifest, preserving native inputs.
        with tempfile.TemporaryDirectory(prefix="refiner_omomo_", dir=Path("runs").resolve()) as directory:
            for key in ("path", "object_mesh_path", "text_features_path"):
                if row.get(key):
                    row[key] = str(Path(row[key]).resolve())
            path = Path(directory) / "manifest.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            ds = MotionDataset(path, "val", "omomo", digest_file(support), window=2)
            batch = ds[0]
            target = {k: batch[k].clone() for k in ("human", "object", "contact")}
            coarse = {k: v.clone() for k, v in target.items()}
            coarse["human"][:, -3:] += .03
            coarse["object"][:, :3] += .04
            coarse["contact"] = .1 + .8 * coarse["contact"]
            geometry = BodyGeometry("omomo", support)
            model = ContinuousRefiner(RefinerConfig("omomo", width=16, layers=1))
            vertices, _ = load_mesh_asset(row["object_mesh_path"])
            loss, _, _ = refinement_loss(model, coarse, target, "111", batch["object_bps"], batch["gender"],
                                         float(batch["fps"]), geometry,
                                         torch.tensor(vertices * row["object_mesh_scale"]), geometry_frames=2)
            loss.backward()
            self.assertTrue(torch.isfinite(model.head.weight.grad).all())
            self.assertGreater(float(model.head.weight.grad.norm()), 0)


if __name__ == "__main__":
    unittest.main()
