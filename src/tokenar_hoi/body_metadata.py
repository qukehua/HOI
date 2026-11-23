"""Dataset body-model provenance, independent of licensed assets and training."""
import numpy as np


BODY_METADATA_KEYS = ("body_model_type", "gender", "betas", "body_dataset")
DATASET_MODELS = {"omomo": "smplx", "behave": "smplh"}
MODEL_BETAS = {"smplx": 16, "smplh": 10}


def body_model_settings(dataset=None, model_type=None):
    dataset = None if dataset is None else str(np.asarray(dataset).item()).lower()
    model_type = None if model_type is None else str(np.asarray(model_type).item()).lower()
    expected = DATASET_MODELS.get(dataset)
    if expected and model_type and model_type != expected:
        raise ValueError(f"{dataset} requires {expected}, not {model_type}")
    model_type = model_type or expected
    if model_type not in MODEL_BETAS:
        raise ValueError("Specify dataset='omomo'/'behave' or model_type='smplx'/'smplh'; "
                         "22 joints alone do not identify the body model")
    return model_type, MODEL_BETAS[model_type]


def body_metadata(record):
    """Copy available provenance; legacy archives may omit gender and betas."""
    if not any(key in record for key in BODY_METADATA_KEYS) and str(record.get("dataset", "")) not in DATASET_MODELS:
        return {}
    dataset = record.get("body_dataset", record.get("dataset"))
    model_type, count = body_model_settings(dataset, record.get("body_model_type"))
    result = {"body_model_type": np.array(model_type)}
    if dataset is not None:
        result["body_dataset"] = np.array(str(np.asarray(dataset).item()))
    if "gender" in record:
        gender = str(np.asarray(record["gender"]).item()).lower()
        if gender not in {"male", "female", "neutral"}:
            raise ValueError(f"Unsupported body gender: {gender}")
        result["gender"] = np.array(gender)
    if "betas" in record:
        betas = np.asarray(record["betas"], dtype=np.float32)
        if betas.shape not in {(count,), (1, count)} or not np.isfinite(betas).all():
            raise ValueError(f"{model_type} requires one finite [{count}] beta vector")
        result["betas"] = betas.reshape(count).copy()
    return result
