"""Three HOI tasks and the metric tables that score them.

Masks are H/O/C with 1 = generate. The vocabulary contains only these tokens:

- text_to_hoi (111): caption -> human, object, contact
- object_to_human (101): object motion, with or without text -> human, contact
- human_to_object (011): human motion, no text -> object, contact
"""
import torch

from . import MODES

TASKS = {
    "text_to_hoi": "111",
    "object_to_human": "101",
    "human_to_object": "011",
}
TASK_MODES = ("111", "101", "011")
TASK_OF_MODE = {mode: name for name, mode in TASKS.items()}

# Printed with every task summary so a row can be read without the paper figure.
TASK_PROTOCOL = {
    "text_to_hoi": {
        "mode": "111",
        "condition": "caption; the object mesh is a geometric prefix and is not generated",
        "FID": "lower",
        "R_Precision": "higher",
        "Diversity": "closer_to_real",
    },
    "object_to_human": {
        "mode": "101",
        "condition": "observed object motion; report without_text and with_text separately",
        "HandJPE": "lower; world-frame metres over joints 20:52 (both wrists and 30 hand joints)",
        "MPJPE": "lower; world-frame metres over the 22 body joints",
        "C_prec": "higher",
        "C_rec": "higher",
        "C_acc": "higher; fraction, not percent",
        "c_percent": "higher; fraction of frames with any body vertex within 5cm of the object",
        "contact": "690 object-frame nearest points plus capped distance; binary contact is under 5cm",
        "units": "joint errors in metres; multiply by 100 for centimetres",
    },
    "human_to_object": {
        "mode": "011",
        "condition": "observed human motion and no text",
        "Ec": "lower; object centre error in metres",
        "Ev2v": "lower; mean object vertex error in metres",
    },
}


def require_task_modes(modes):
    if set(modes) != set(TASK_MODES) or len(modes) != len(TASK_MODES):
        raise ValueError("Train text_to_hoi (111), object_to_human (101) and human_to_object (011)")
    return list(modes)


def candidate_modes(modes, text):
    """Drop text-to-HOI when the window has no caption."""
    if not modes or any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError("Invalid H/O/C masks")
    pool = [mode for mode in modes if mode != "111" or str(text).strip()]
    if not pool:
        raise ValueError("Text-to-HOI needs a caption")
    return pool


def choose_mode(modes, text):
    pool = candidate_modes(modes, text)
    return pool[int(torch.randint(len(pool), ()))]


def condition_text(mode, text, dropout):
    """Text is part of the task, not a global switch."""
    if mode not in MODES:
        raise ValueError("Invalid H/O/C mask")
    if not 0 <= float(dropout) <= 1:
        raise ValueError("text_dropout must be in [0, 1]")
    if mode == "111":
        return text
    if mode == "011":
        return ""
    if mode == "101":
        return "" if float(torch.rand(())) < float(dropout) else text
    raise ValueError("Unknown task mask")


def validation_conditions(modes):
    rows = []
    for mode in modes:
        if mode == "111":
            rows.append((mode, "with_text"))
        elif mode == "101":
            rows.append((mode, "without_text"))
            rows.append((mode, "with_text"))
        elif mode == "011":
            rows.append((mode, "without_text"))
        else:
            raise ValueError("Validation tasks are the three HOI tasks")
    return rows


def condition_key(mode, condition):
    return f"{TASK_OF_MODE[mode]}.{condition}"


def evaluation_settings(mode, caption):
    """(task, condition, prompt) rows for one task. An empty caption skips text-to-HOI."""
    caption = "" if caption is None else str(caption)
    if mode not in MODES:
        raise ValueError("Unknown task mask")
    if mode == "111":
        return [("text_to_hoi", "with_text", caption)] if caption.strip() else []
    if mode == "101":
        rows = [("object_to_human", "without_text", "")]
        if caption.strip():
            rows.append(("object_to_human", "with_text", caption))
        return rows
    return [("human_to_object", "without_text", "")]


def report_key(mode, condition):
    """Object-to-human keeps both text rows; the other tasks have one row."""
    if mode == "101":
        return f"{mode}/{condition}"
    return mode


def contact_fractions(counts):
    tp, fp, fn, tn = (int(counts[name]) for name in ("TP", "FP", "FN", "TN"))
    total = tp + fp + fn + tn
    if total <= 0:
        raise ValueError("Contact counts are empty")
    return {"C_prec": tp / (tp + fp) if tp + fp else 0.,
            "C_rec": tp / (tp + fn) if tp + fn else 0.,
            "C_acc": (tp + tn) / total}


def object_to_human_row(bundle):
    primary, micro = bundle["primary"], bundle["micro"]
    row = {"HandJPE": primary["HandJPE"], "MPJPE": primary["MPJPE_BODY"],
           **contact_fractions(micro["SBJ_CONTACT"]),
           "c_percent": primary["c_percent"], "c_percent_real": primary["c_percent_real"],
           "evaluated_windows": bundle["windows"], "frames": bundle["frames"]}
    return row


def human_to_object_row(bundle):
    primary = bundle["primary"]
    return {"Ec": primary["OBJ_CENTER"], "Ev2v": primary["OBJ_V2V"],
            "evaluated_windows": bundle["windows"], "frames": bundle["frames"]}


def text_to_hoi_row(figure):
    return {"FID": figure["FID"], "R_Precision": dict(figure["R_Precision"]),
            "Diversity": figure["Diversity"], "Diversity_real": figure["Diversity_real"],
            "view": figure["view"], "evaluated_windows": figure.get("evaluated_windows")}


def build_task_summary(aggregated, figure=None):
    """Official three-task block. `aggregated` maps report keys to metric bundles."""
    tasks = {"protocol": TASK_PROTOCOL}
    if "111" in aggregated:
        if figure is None:
            tasks["text_to_hoi"] = {"mode": "111", "status": "needs motion_text_evaluator",
                                    "evaluated_windows": aggregated["111"]["windows"]}
        else:
            tasks["text_to_hoi"] = {"mode": "111", "condition": "with_text", **text_to_hoi_row(figure)}
    rows = {}
    for condition in ("without_text", "with_text"):
        key = f"101/{condition}"
        if key in aggregated:
            rows[condition] = object_to_human_row(aggregated[key])
    if rows:
        tasks["object_to_human"] = {"mode": "101", **rows}
    if "011" in aggregated:
        tasks["human_to_object"] = {"mode": "011", "condition": "without_text",
                                    **human_to_object_row(aggregated["011"])}
    return tasks


def task_mean(scores):
    """Equal weight per task. Object-to-human averages its two text conditions first."""
    groups = {}
    for key, value in scores.items():
        groups.setdefault(key.split(".", 1)[0], []).append(value)
    if set(groups) != set(TASKS):
        raise ValueError("Validation missed one of the three HOI tasks")
    per_task = {name: sum(values) / len(values) for name, values in groups.items()}
    return sum(per_task.values()) / len(per_task), per_task
