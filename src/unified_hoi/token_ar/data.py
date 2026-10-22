"""Raw captions and native motion, without loading the diffusion branch's CLIP cache."""
import torch

from .native_data import ParametricDataset, load_record

__all__ = ["MotionDataset", "load_record"]


class MotionDataset(ParametricDataset):
    def __init__(self, manifest, split, dataset, support_hash, window=120, stride=None,
                 include_geometry=True):
        super().__init__(manifest, split, dataset, support_hash, window, stride,
                         text_condition=False, include_geometry=include_geometry)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        record_index, _, _ = self.windows[index]
        entry = self.records[record_index]
        captions = [v["text"] for v in entry.get("text_variants", []) if v.get("text", "").strip()]
        if not captions:
            captions = [entry.get("text", "")]
        selected = int(torch.randint(len(captions), ())) if self.split == "train" else 0
        result["text"] = captions[selected]
        result["contact"] = result.pop("surface_contact")
        result.pop("text_features")
        result.pop("text_available")
        return result
