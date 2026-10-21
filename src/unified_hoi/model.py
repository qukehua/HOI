"""Attention blocks and positional encoding shared by the native denoiser."""
import math

import torch
from torch import nn


def positions(length, width, device, dtype):
    idx = torch.arange(length, device=device, dtype=dtype)[:, None]
    freq = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10000) / width))
    return torch.stack(((idx * freq).sin(), (idx * freq).cos()), -1).flatten(-2)[:, :width]


class FactorizedBlock(nn.Module):
    """Spatial attention per frame, temporal attention per token; avoids O((T*J)^2)."""
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.spatial = nn.TransformerEncoderLayer(width, heads, width * 4, dropout,
                                                 batch_first=True, norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoderLayer(width, heads, width * 4, dropout,
                                                  batch_first=True, norm_first=True, activation="gelu")

    def forward(self, x, valid):
        b, t, n, d = x.shape
        x = self.spatial(x.reshape(b * t, n, d)).reshape(b, t, n, d)
        y = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
        padding = (~valid)[:, None].expand(b, n, t).reshape(b * n, t)
        y = self.temporal(y, src_key_padding_mask=padding)
        return y.reshape(b, n, t, d).permute(0, 2, 1, 3) * valid[:, :, None, None]
