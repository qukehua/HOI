"""Load licensed native body models and export full SMPL-H/SMPL-X meshes."""
import argparse
from pathlib import Path

import numpy as np

from .body_metadata import body_model_settings
from .geometry import SMPL_PARENTS

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "smplx_models"
_LAYERS: dict = {}


def model_asset_path(model_path, gender, *, model_type):
    model_type, count = body_model_settings(model_type=model_type)
    gender = str(gender).lower()
    if gender not in {"male", "female", "neutral"}:
        raise ValueError(f"Unsupported body gender: {gender}")
    root = Path(model_path).resolve()
    name = f"{model_type.upper()}_{gender.upper()}"
    extensions = ("npz",) if model_type == "smplx" else ("pkl", "npz")
    candidates = ([root] if root.is_file() else
                  [folder / f"{name}.{ext}" for folder in (root / model_type, root) for ext in extensions])
    asset = next((p for p in candidates if p.is_file()), None)
    if asset is None:
        raise FileNotFoundError(f"Missing licensed {model_type} asset: {candidates[0]}. "
                                "Use the dataset's official model; no fallback to another body model.")
    return asset


def _layer(model_path, gender, batch_size=1, *, model_type):
    model_type, count = body_model_settings(model_type=model_type)
    gender = str(gender).lower()
    asset = model_asset_path(model_path, gender, model_type=model_type)
    stat = asset.stat()
    key = (str(asset), model_type, gender, count, stat.st_size, stat.st_mtime_ns)
    if key not in _LAYERS:
        import smplx
        layer = smplx.build_layer(str(asset), model_type=model_type, gender=gender,
                                 ext=asset.suffix.lstrip("."), use_pca=False,
                                 flat_hand_mean=True, num_betas=count)
        if layer.num_betas != count:
            raise ValueError(f"{asset} supplies {layer.num_betas} betas; {model_type} requires {count}")
        if tuple(layer.parents[:22].tolist()) != SMPL_PARENTS:
            raise ValueError("Body-model parents do not match the HOI 22-joint skeleton")
        _LAYERS[key] = layer
    # Layer models infer batch size at forward time; do not duplicate large assets per clip length.
    return _LAYERS[key]


def body_skinning_weights(weights, parents):
    """Fold omitted joints into their nearest retained body ancestor.

    This sums all finger weights into the corresponding wrist, and SMPL-X
    jaw/eye weights into the head. It works with both 52- and 55-joint trees.
    """
    weights, parents = np.asarray(weights), np.asarray(parents)
    if weights.ndim != 2 or weights.shape[1] != len(parents) or len(parents) < 22:
        raise ValueError("Skinning weights and parent tree disagree")
    if tuple(parents[:22]) != SMPL_PARENTS:
        raise ValueError("Expected the HOI 22-joint body hierarchy")
    ancestors = np.arange(len(parents))
    for joint in range(22, len(parents)):
        parent = int(parents[joint])
        if not 0 <= parent < joint:
            raise ValueError("Expected topologically ordered skinning parents")
        ancestors[joint] = ancestors[parent]
    result = np.zeros((len(weights), 22), dtype=weights.dtype)
    for joint, ancestor in enumerate(ancestors):
        result[:, ancestor] += weights[:, joint]
    return result


def record_to_smpl(record, model_path=DEFAULT_MODEL_PATH):
    """Decode a processed/prediction archive using its saved body provenance."""
    if str(record.get("representation")) == "parametric_v1":
        from .token_ar.body import decode_full, digest_file
        from .body_metadata import DATASET_MODELS
        dataset, gender = str(record["dataset"]), str(record["gender"])
        model_type = DATASET_MODELS[dataset]
        if (str(record["body_model_type"]) != model_type or
                digest_file(model_asset_path(model_path, gender, model_type=model_type)) != str(record["body_model_sha256"])):
            raise ValueError("Native mesh export requires the recorded body assets")
        vertices, faces, _ = decode_full(record["human_params"], dataset, gender, model_path)
        layer = _layer(model_path, gender, model_type=model_type)
        weights = body_skinning_weights(layer.lbs_weights.numpy(), layer.parents.numpy())
        return vertices, faces, weights.argmax(1)
    raise ValueError("Mesh export requires a parametric_v1 record; old body-FK export has been retired")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as archive:
        record = dict(archive)
    vertices, faces, dominant = record_to_smpl(record, args.models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {k: record[k] for k in ("dataset", "gender", "body_model_type", "body_model_sha256", "representation")}
    np.savez_compressed(args.output, vertices=vertices, faces=faces, dominant=dominant,
                        **metadata, **{k: record[k] for k in ("fps", "timestamps") if k in record})
    print(args.output.resolve())


if __name__ == "__main__":
    main()
