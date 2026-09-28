# Copyright (c) 2025 Ilia Petrov
# SPDX-License-Identifier: MIT
# Exact function/class bodies from ptrvilya/tridi commit
# afa9631dc2b3a250588ab64026eeaa37f18f0d38:
# tridi/model/denoising/transformer_uni_3.py.
# Extraction and this provenance header are the only modifications.
# Full license: licenses/TRIDI_LICENSE.txt.

import torch
from torch import nn
import numpy as np


def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Note that this should work just as well for
    continuous values as for discrete values.
    """

    assert len(timesteps.shape) == 1
    half_dim = embed_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
    return emb


class Projection(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.projection(x)
