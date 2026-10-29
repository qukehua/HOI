"""Oracle H/O/C VQ reconstruction, without loading or running a language model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..mesh_geometry import load_mesh_asset
from ..runtime import move_batch, write_json
from . import MODALITIES, MODES
from .body import BodyGeometry, digest_file
from .codecs import canonicalize, load_codecs
from .data import MotionDataset
from .evaluation_protocol import aggregate_reports, fingerprint, frame_plan, protocol_options, window_selection
from .metrics import contact_scores, mesh_contacts, reconstruction


class _ProbabilityDecoder:
    @staticmethod
    def decode(probabilities):
        return probabilities


def usage_statistics(counts):
    """Corpus-level usage, not the checkpoint's periodically reset training counters."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    probabilities = counts[counts > 0] / max(total, 1)
    return dict(tokens=total, used_codes=int((counts > 0).sum()), codebook_size=len(counts),
                usage_fraction=float((counts > 0).mean()),
                perplexity=float(np.exp(-(probabilities * np.log(probabilities)).sum())) if total else 0.,
                counts=counts.tolist())


@torch.no_grad()
def evaluate_codecs(checkpoint, output, *, manifest=None, contact_support=None, body_models="data/smplx_models",
                    split="val", window=120, modes=("100", "010", "001", "111"), device="cpu",
                    profile="tridi", eval_fps=None, limit=None, evaluator=None, geometry_only=False,
                    seed=42, candidates=32, repeats=20, diversity_pairs=300):
    if (split not in ("val", "test") or window < 1 or (limit is not None and limit < 1)
            or not modes or any(m not in MODES for m in modes) or len(set(modes)) != len(modes)):
        raise ValueError("Require val/test, positive window/limit and distinct H/O/C masks")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol = protocol_options(profile, eval_fps)
    device = torch.device(device)
    codecs, saved = load_codecs(checkpoint, device=device)
    dataset_name = codecs.config.dataset
    config = saved.get("config", {})
    manifest = manifest or config.get("manifest")
    if not manifest:
        raise ValueError("Specify --manifest; this VQ checkpoint has no saved manifest path")
    contact_support = contact_support or config.get("contact_support") or (
        f"data/parametric_assets/{dataset_name}/contact_support.npz")
    if not geometry_only and (evaluator or config.get("motion_text_evaluator")):
        from .evaluate_motion_text import evaluate_motion_text
        return evaluate_motion_text(codecs, saved, checkpoint, output,
            evaluator=evaluator or config["motion_text_evaluator"], manifest=manifest, contact_support=contact_support,
            split=split, window=window, limit=limit, device=device, seed=seed,
            candidates=candidates, repeats=repeats, diversity_pairs=diversity_pairs)
    if not geometry_only and codecs.config.network == "residual":
        raise ValueError("New VQ evaluation requires --evaluator; use --geometry-only for the geometry diagnostic")
    support_hash = digest_file(contact_support)
    if saved.get("support_sha256") != support_hash:
        raise ValueError("VQ checkpoint/contact support mismatch")
    manifest_hash = digest_file(manifest)
    manifest_matches = (manifest_hash == saved["manifest_sha256"] if saved.get("manifest_sha256") else None)
    if manifest_matches is False:
        print("Warning: manifest differs from VQ training; verify evaluation frame/reference hashes.", flush=True)
    geometry = BodyGeometry(dataset_name, contact_support, body_models).to(device).eval()
    dataset = MotionDataset(manifest, split, dataset_name, support_hash, window, include_geometry=False)
    if not len(dataset):
        raise ValueError("No evaluation windows")
    # Evaluation uses nonoverlapping windows, independent of the VQ training stride.
    plan = frame_plan(dataset.records, protocol["eval_fps"])
    total = len(dataset) if limit is None else min(limit, len(dataset))
    output.mkdir(parents=True)
    reports, contact_reports, frame_ids, meshes = {}, [], [], {}
    counts = {k: np.zeros(codecs.config.codebook_size, np.int64) for k in MODALITIES}
    reference_digest = hashlib.sha256()
    scored_windows = 0
    for index in range(total):
        record_index, start, end = dataset.windows[index]
        n = end - start
        batch = move_batch(next(iter(DataLoader(dataset, batch_sampler=[[index]]))), device)
        gender_name = "male" if int(batch["gender"][0]) else "female"
        if batch["body_model_sha256"][0] != geometry.layers[gender_name].asset_hash:
            raise ValueError("Native record/body model mismatch")
        encoded = codecs.encode(batch)
        for key in MODALITIES:
            counts[key] += np.bincount(encoded[key][0].cpu().numpy(), minlength=len(counts[key]))
        selected, ids = window_selection(plan[record_index], start, end)
        if not len(selected):
            continue
        decoded = {k: codecs.models[k].decode(encoded[k][0], n) for k in MODALITIES}
        decoded["human"], decoded["object"] = canonicalize(decoded["human"], decoded["object"], dataset_name)
        if any(not torch.isfinite(v).all() for v in decoded.values()):
            raise FloatingPointError("Nonfinite VQ reconstruction")
        entry = dataset.records[record_index]
        mesh_path = str(Path(entry["object_mesh_path"]).resolve())
        mesh_key = (mesh_path, float(entry["object_mesh_scale"]))
        if mesh_key not in meshes:
            vertices, _ = load_mesh_asset(mesh_path)
            meshes[mesh_key] = (vertices * mesh_key[1], digest_file(mesh_path))
        vertices, mesh_hash = meshes[mesh_key]
        reference_digest.update(fingerprint([mesh_hash, entry["object_mesh_scale"]]).encode())
        target = {"human": batch["human"][:, :n], "object": batch["object"][:, :n],
                  "surface_contact": batch["contact"][:, :n], "gender": batch["gender"],
                  "valid_frames": torch.ones(1, n, dtype=torch.bool, device=device)}
        valid = torch.as_tensor(selected, device=device)
        for key in ("human", "object", "surface_contact"):
            reference_digest.update(target[key][0, valid].cpu().numpy().astype("<f4").tobytes())
        frame_ids.extend(ids)
        gt_vertices, _ = geometry(target["human"][:, valid], target["gender"])
        pred_vertices, _ = geometry(decoded["human"][None, valid], target["gender"])
        human_extra = dict(
            HUMAN_TRANSL_L2=float((decoded["human"][valid, -3:] - target["human"][0, valid, -3:]).norm(dim=-1).mean()),
            HUMAN_V2V_690=float((pred_vertices - gt_vertices).norm(dim=-1).mean()))
        native_contact = target["surface_contact"][0, valid].cpu().numpy() >= .5
        geometry_contact = mesh_contacts(gt_vertices[0].cpu().numpy().astype(np.float64),
                                         target["object"][0, valid].cpu().numpy(), vertices)
        contact_report = dict(frames=len(selected), metrics={}, details={
            "VQ_VS_NATIVE": contact_scores(decoded["contact"][valid].cpu().numpy() >= .5, native_contact),
            "NATIVE_VS_GT_MESH": contact_scores(native_contact, geometry_contact),
            "ALL_NEGATIVE_VS_NATIVE": contact_scores(np.zeros_like(native_contact), native_contact)})
        contact_reports.append(contact_report)
        for mode in modes:
            # In oracle mode, target tokens are supplied directly, never predicted by Qwen.
            # Observed modalities stay exact, as in complete(); do not round-trip them.
            states = {key: decoded[key][None] if bit == "1" else batch[key][:, :n]
                      for key, bit in zip(MODALITIES, mode)}
            report = reconstruction(states, target, geometry, mode, vertices, _ProbabilityDecoder(), .5, selected)
            if mode[0] == "1":
                report["metrics"].update(human_extra)
            report.update(mode=mode, sequence_id=entry["sequence_id"], window_start=start,
                          input_frames=n, selected_local_indices=selected.tolist(),
                          contact_diagnostics=contact_report["details"])
            reports.setdefault(mode, []).append(report)
            with (output / "per_clip.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
        scored_windows += 1
        print(f"VQ reconstruction window {index + 1}/{total}", flush=True)
    if not reports:
        raise ValueError("No frames selected; increase --limit or change --eval-fps")
    _, _, contact_diagnostics = aggregate_reports(contact_reports)
    summary = dict(
        kind="vq_oracle_reconstruction", sampler="ground_truth_tokens", dataset=dataset_name,
        checkpoint=str(checkpoint), checkpoint_sha256=digest_file(checkpoint), checkpoint_step=saved.get("step"),
        manifest=str(manifest), manifest_sha256=manifest_hash, manifest_matches_training=manifest_matches,
        support_sha256=support_hash, body_model_sha256={k: v.asset_hash for k, v in geometry.layers.items()},
        split=split, window=window, limit_windows=limit, processed_windows=total, scored_windows=scored_windows,
        protocol=protocol, modes={m: "reconstruct " + ",".join(k for k, b in zip(MODALITIES, m) if b == "1")
                                  for m in modes},
        evaluated_frames=len(frame_ids), evaluation_frames_sha256=fingerprint(sorted(frame_ids)),
        reference_values_sha256=reference_digest.hexdigest(),
        token_usage_scope="all valid native-rate tokens of processed nonoverlapping windows, excluding padding",
        token_usage={k: usage_statistics(v) for k, v in counts.items()}, contact_diagnostics=contact_diagnostics,
        metrics={}, reconstruction={}, contact_micro={},
        notes=["Target tokens are supplied by the encoder. This is not conditional generation or a strict bound.",
               "CONTACT_TOKEN F1 is against GT mesh contact; VQ_VS_NATIVE uses the original binary VQ target.",
               "Known contact in modes ending in 0 is copied GT, not evidence of contact VQ quality.",
               "Distances are metres; object rotation is degrees; contact PR/F1 are fractions.",
               "MPJPE is pelvis-relative; HUMAN_TRANSL_L2 is world SMPL translation error.",
               "HUMAN_V2V_690 measures the 690 contact vertices, not the full human mesh.",
               "Generation distribution metrics and train references are not evaluated."])
    for mode, rows in reports.items():
        metrics, primary, micro = aggregate_reports(rows)
        summary["metrics"][mode], summary["reconstruction"][mode], summary["contact_micro"][mode] = metrics, primary, micro
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="VQ .pt file; prefer the AR checkpoint's codecs.pt")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--manifest", help="Override saved native manifest path")
    parser.add_argument("--contact-support", help="Override saved contact support path")
    parser.add_argument("--body-models", default="data/smplx_models")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--window", type=int, default=120, help="Match AR evaluation window length")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["100", "010", "001", "111"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", choices=("tridi", "native"), default="tridi")
    parser.add_argument("--eval-fps", type=float, help="Default: tridi 1fps; native all frames (0)")
    parser.add_argument("--limit", type=int, help="First N windows for smoke tests only; omit for full evaluation")
    parser.add_argument("--evaluator", help="Frozen native motion-text evaluator; defaults to the VQ config path")
    parser.add_argument("--geometry-only", action="store_true", help="Use the earlier MPJPE/contact diagnostic instead")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--diversity-pairs", type=int, default=300)
    args = vars(parser.parse_args())
    torch.set_num_threads(4)
    summary = evaluate_codecs(**args)
    print(json.dumps({"output": args["output"], "reconstruction": summary.get("reconstruction"),
                      "embedding_metrics": summary.get("embedding_metrics"),
                      "contact_diagnostics": summary.get("contact_diagnostics")}, allow_nan=False))


if __name__ == "__main__":
    main()
