"""Native local-pose temporal denoiser; H parameters / O pose / contact latent."""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..geometry import rotation6d_to_matrix, human_in_object_frame, nearest_surface
from ..integrations._tridi_primitives import Projection, get_timestep_embedding
from ..model import FactorizedBlock, positions
from .body import beta_count, human_dim, split_human
from .contact import ContactAE

MODALITIES = ("human", "object", "contact")


@dataclass
class ParametricConfig:
    dataset: str = "behave"
    width: int = 192
    heads: int = 6
    layers: int = 4
    dropout: float = .1
    text_dim: int = 512
    relations: bool = True
    two_stage: bool = True

    def __post_init__(self):
        if (self.dataset not in ("behave", "omomo") or self.heads < 1 or self.width < 2 or self.layers < 1
                or self.width % self.heads or self.width % 2 or not 0 <= self.dropout < 1):
            raise ValueError("Invalid dataset/attention dimensions")

    def asdict(self):
        return asdict(self)


class Normalizer(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        for key, dim in (("human", human_dim(dataset)), ("object", 9), ("contact", 128)):
            self.register_buffer(key + "_mean", torch.zeros(dim))
            self.register_buffer(key + "_std", torch.ones(dim))

    def encode(self, states):
        return {k: (states[k] - getattr(self, k + "_mean")) / getattr(self, k + "_std") for k in MODALITIES}

    def decode(self, states):
        return {k: states[k] * getattr(self, k + "_std") + getattr(self, k + "_mean") for k in MODALITIES}

    @torch.no_grad()
    def fit(self, loader, codec):
        sums, squares, count = {}, {}, 0
        for batch in loader:
            batch["contact"] = codec.encode(batch["surface_contact"])
            valid = batch["valid_frames"]
            count += int(valid.sum())
            for key in MODALITIES:
                values = batch[key][valid].double()
                sums[key] = sums.get(key, 0) + values.sum(0)
                squares[key] = squares.get(key, 0) + values.square().sum(0)
        if count == 0:
            raise ValueError("Empty training split")
        for key in MODALITIES:
            mean = sums[key] / count
            std = (squares[key] / count - mean.square()).clamp_min(0).sqrt().clamp_min(.05)
            getattr(self, key + "_mean").copy_(mean.float())
            getattr(self, key + "_std").copy_(std.float())


def add_geometry_batch(batch, geometry):
    """Gender-specific neutral shape bases only; no ground-truth beta conditioning."""
    base, basis = [], []
    for value in batch["gender"].tolist():
        layer = geometry.layers["male" if value else "female"]
        base.append(layer.joint_template[:22])
        basis.append(layer.joint_shape_dirs[:22])
    return {**batch, "joint_base": torch.stack(base), "joint_basis": torch.stack(basis)}


def body_joints22(human, batch, dataset):
    from ..geometry import SMPL_PARENTS
    beta, pose, trans = split_human(human, dataset)
    rest = batch["joint_base"][:, None] + torch.einsum("btk,bjck->btjc", beta, batch["joint_basis"])
    local = rotation6d_to_matrix(pose[..., :22, :])
    xyz, rotations = [rest[..., 0, :] + trans], [local[..., 0, :, :]]
    for j in range(1, 22):
        parent = SMPL_PARENTS[j]
        xyz.append(xyz[parent] + (rotations[parent] @ (rest[..., j, :] - rest[..., parent, :]).unsqueeze(-1)).squeeze(-1))
        rotations.append(rotations[parent] @ local[..., j, :, :])
    return torch.stack(xyz, -2)


class ParametricDenoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, n = config.width, beta_count(config.dataset)
        self.normalizer = Normalizer(config.dataset)
        self.codec = ContactAE().requires_grad_(False)
        self.register_buffer("contact_threshold", torch.tensor(.5))
        self.pose_in, self.shape_in = Projection(12, d), Projection(n * 2, d)
        self.trans_in, self.object_in, self.contact_in = Projection(6, d), Projection(18, d), Projection(256, d)
        self.token_embed = nn.Parameter(torch.randn(1, 1, 56, d) * .02)
        self.geometry_net = nn.Sequential(nn.Linear(3072, 768), nn.ReLU(), nn.Linear(768, d))
        self.text_net, self.time_net, self.fps_net = Projection(config.text_dim, d), Projection(d, d), Projection(1, d)
        if config.two_stage:
            self.coarse_in = Projection(2 * (human_dim(config.dataset) + 9 + 128), d)
            self.coarse_block = nn.TransformerEncoderLayer(d, config.heads, 4 * d, config.dropout,
                                                          batch_first=True, norm_first=True)
            self.coarse_out = Projection(d, 18)
            self.coarse_embed = Projection(18, d)
        if config.relations:
            self.relation_net = Projection(7, d)
            self.relation_query = nn.Linear(d, d)
        self.blocks = nn.ModuleList(FactorizedBlock(d, config.heads, config.dropout) for _ in range(config.layers))
        self.pose_head, self.shape_head = nn.Linear(d, 6), nn.Linear(d, n)
        self.trans_head, self.object_head, self.contact_head = nn.Linear(d, 3), nn.Linear(d, 9), nn.Linear(d, 128)

    def forward(self, states, times, masks, batch):
        h, o, c = (states[k] for k in MODALITIES)
        b, t = h.shape[:2]
        n, d = beta_count(self.config.dataset), self.config.width
        valid = batch["valid_frames"]
        beta, pose, trans = split_human(h, self.config.dataset)
        bm, pm, tm = split_human(masks["human"].to(h), self.config.dataset)
        hp = self.pose_in(torch.cat((pose, pm), -1))
        hs = self.shape_in(torch.cat((beta, bm), -1)).unsqueeze(2)
        ht = self.trans_in(torch.cat((trans, tm), -1)).unsqueeze(2)
        ot = self.object_in(torch.cat((o, masks["object"].to(o)), -1)).unsqueeze(2)
        ct = self.contact_in(torch.cat((c, masks["contact"].to(c)), -1)).unsqueeze(2)
        time = [self.time_net(get_timestep_embedding(d, times[:, i], h.device))[:, None, None] for i in range(3)]
        tokens = torch.cat((hp + time[0], hs + time[0], ht + time[0], ot + time[1], ct + time[2]), 2)
        condition = self.geometry_net(batch["object_bps"].flatten(1)) + self.text_net(batch["text_features"])
        condition = condition + self.fps_net(batch["fps"].log()[:, None])
        pe = positions(t, d, h.device, h.dtype)[None, :, None]
        tokens = tokens + self.token_embed + pe + condition[:, None, None]
        if self.config.two_stage:
            x = torch.cat([torch.cat((states[k], masks[k].to(h)), -1) for k in MODALITIES], -1)
            coarse = self.coarse_out(self.coarse_block(self.coarse_in(x) + condition[:, None] + pe.squeeze(2)
                                                      + sum(time).squeeze(2), src_key_padding_mask=~valid))
            tokens = tokens + self.coarse_embed(coarse)[:, :, None]
        if self.config.relations:
            physical = self.normalizer.decode(states)
            points = body_joints22(physical["human"], batch, self.config.dataset)
            local = human_in_object_frame(points, physical["object"])
            distance, near = nearest_surface(points, physical["object"], batch["object_points"])
            rel = self.relation_net(torch.cat((local, near - local, distance[..., None]), -1))
            # The object coordinate frame is unreliable at high O noise. Known O has gate=1.
            gate = (1 - times[:, 1].float() / batch["diffusion_steps"]).clamp(0, 1)[:, None, None, None]
            rel = rel * gate
            attention = (self.relation_query(ot) * rel).sum(-1).softmax(-1)
            pooled = (attention[..., None] * rel).sum(2, keepdim=True)
            zero = tokens.new_zeros(b, t, 32, d)
            tokens = tokens + torch.cat((rel, zero, pooled, pooled), 2)
        for block in self.blocks:
            tokens = block(tokens, valid)
        shape = self.shape_head(tokens[:, :, 52])
        shape = (shape * valid[..., None]).sum(1, keepdim=True) / valid.sum(1)[:, None, None]
        result = {"human": torch.cat((shape.expand(b, t, n), self.pose_head(tokens[:, :, :52]).flatten(2),
                                      self.trans_head(tokens[:, :, 53])), -1),
                  "object": self.object_head(tokens[:, :, 54]), "contact": self.contact_head(tokens[:, :, 55])}
        return result
