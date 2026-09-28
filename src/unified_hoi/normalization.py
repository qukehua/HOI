"""Training-split-only statistics, saved inside every model checkpoint."""
import torch
from torch import nn

MODALITIES = ("human", "object", "contact")


class StateNormalizer(nn.Module):
    def __init__(self, joints=22):
        super().__init__()
        for key, shape in (("human", (joints, 9)), ("object", (9,)), ("contact", (joints,))):
            self.register_buffer(key + "_mean", torch.full(shape, .5) if key == "contact" else torch.zeros(shape))
            self.register_buffer(key + "_std", torch.full(shape, .5) if key == "contact" else torch.ones(shape))

    @torch.no_grad()
    def fit(self, loader):
        sums, squares, count = {}, {}, 0
        for batch in loader:
            valid = batch["valid_frames"].bool()
            count += int(valid.sum())
            for key in ("human", "object"):
                values = batch[key][valid].double()
                sums[key] = sums.get(key, 0) + values.sum(0)
                squares[key] = squares.get(key, 0) + values.square().sum(0)
        if not count:
            raise ValueError("Cannot fit normalization on an empty training split")
        for key in sums:
            mean = sums[key] / count
            std = (squares[key] / count - mean.square()).clamp_min(0).sqrt().clamp_min(.05)
            getattr(self, key + "_mean").copy_(mean.float())
            getattr(self, key + "_std").copy_(std.float())

    def encode(self, states):
        return {k: (states[k] - getattr(self, k + "_mean")) / getattr(self, k + "_std") for k in MODALITIES}

    def decode(self, states):
        return {k: states[k] * getattr(self, k + "_std") + getattr(self, k + "_mean") for k in MODALITIES}
