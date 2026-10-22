"""Three independently quantized temporal VQ-VAEs; no diffusion contact-AE reuse."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..geometry import matrix_to_rotation6d, rotation6d_to_matrix
from . import ARCHITECTURE, MODALITIES
from .body import beta_count, human_dim


@dataclass
class CodecConfig:
    dataset: str = "omomo"
    codebook_size: int = 512
    latent_dim: int = 128
    width: int = 256
    stride: int = 2
    commitment_weight: float = .25
    velocity_weight: float = .1

    def __post_init__(self):
        if self.dataset not in ("behave", "omomo"):
            raise ValueError("Unknown dataset")
        if min(self.codebook_size, self.latent_dim, self.width, self.stride) < 1:
            raise ValueError("Codec dimensions must be positive")


def valid_lengths(valid):
    lengths = valid.long().sum(-1)
    expected = torch.arange(valid.shape[1], device=valid.device)[None] < lengths[:, None]
    if (lengths < 1).any() or not torch.equal(valid.bool(), expected):
        raise ValueError("Require nonempty, contiguous valid frames followed by padding")
    return lengths.tolist()


class MotionVQVAE(nn.Module):
    def __init__(self, dim, config, contact=False):
        super().__init__()
        self.config, self.contact = config, contact
        w, d, s = config.width, config.latent_dim, config.stride
        self.encoder = nn.Sequential(nn.Conv1d(dim, w, s, stride=s), nn.GELU(),
                                     nn.Conv1d(w, w, 3, padding=1), nn.GELU(), nn.Conv1d(w, d, 1))
        self.decoder = nn.Sequential(nn.Conv1d(d, w, 3, padding=1), nn.GELU(),
                                     nn.ConvTranspose1d(w, dim, s, stride=s))
        self.codebook = nn.Embedding(config.codebook_size, d)
        nn.init.normal_(self.codebook.weight, std=.1)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("usage", torch.zeros(config.codebook_size))
        self.register_buffer("initialized", torch.tensor(False))

    def _latent(self, values):
        x = ((values - self.mean) / self.std).T[None]
        extra = (-values.shape[0]) % self.config.stride
        if extra:
            x = F.pad(x, (0, extra), mode="replicate")
        return self.encoder(x)[0].T

    def _nearest(self, z):
        # Compute distance in fp32, even under a caller's autocast context.
        with torch.autocast(device_type=z.device.type, enabled=False):
            x, book = z.float(), self.codebook.weight.float()
            return (x.square().sum(-1, keepdim=True) + book.square().sum(-1)[None]
                    - 2 * x @ book.T).argmin(-1)

    def _decode(self, z, length):
        value = self.decoder(z.T[None])[0].T[:length]
        return value if self.contact else value * self.std + self.mean

    @torch.no_grad()
    def encode(self, values, valid):
        # Crop BEFORE convolution: padded frames must never influence valid tokens.
        return [self._nearest(self._latent(x[:n])) for x, n in zip(values, valid_lengths(valid))]

    def decode(self, indices, length):
        if indices.ndim != 1 or len(indices) != (length + self.config.stride - 1) // self.config.stride:
            raise ValueError("Token count does not match requested frame length")
        value = self._decode(self.codebook(indices), length)
        return value.sigmoid() if self.contact else value

    def loss(self, values, valid):
        latent = [self._latent(x[:n]) for x, n in zip(values, valid_lengths(valid))]
        flat = torch.cat(latent)
        if self.training and not bool(self.initialized):
            with torch.no_grad():
                index = torch.arange(self.config.codebook_size, device=flat.device) % len(flat)
                self.codebook.weight.copy_(flat[index] + .01 * torch.randn_like(flat[index]))
                self.initialized.fill_(True)
        ids = [self._nearest(z) for z in latent]
        if self.training:
            with torch.no_grad():
                self.usage.add_(torch.bincount(torch.cat(ids), minlength=self.config.codebook_size))
        rec, quant, velocity = [], [], []
        for x, n, z, indices in zip(values, valid_lengths(valid), latent, ids):
            q = self.codebook(indices)
            quant.append(F.mse_loss(q, z.detach()) + self.config.commitment_weight * F.mse_loss(z, q.detach()))
            pred = self._decode(z + (q - z).detach(), n)
            target = x[:n]
            if self.contact:
                error = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                classes = [error[mask].mean() for mask in (target >= .5, target < .5) if mask.any()]
                rec.append(torch.stack(classes).mean())
            else:
                rec.append(((pred - target) / self.std).square().mean())
                if n > 1:
                    velocity.append(((pred.diff(dim=0) - target.diff(dim=0)) / self.std).square().mean())
        reconstruction, vq = torch.stack(rec).mean(), torch.stack(quant).mean()
        vel = torch.stack(velocity).mean() if velocity else reconstruction * 0
        counts = torch.bincount(torch.cat(ids), minlength=self.config.codebook_size).float()
        probabilities = counts / counts.sum()
        perplexity = (-(probabilities * probabilities.clamp_min(1e-9).log()).sum()).exp()
        loss = reconstruction + vq + self.config.velocity_weight * vel
        return loss, {"reconstruction": reconstruction, "vq": vq, "velocity": vel, "perplexity": perplexity}

    @torch.no_grad()
    def reset_unused(self, values, valid):
        z = torch.cat([self._latent(x[:n]) for x, n in zip(values, valid_lengths(valid))])
        unused = self.usage == 0
        count = int(unused.sum())
        if count:
            index = torch.randint(len(z), (count,), device=z.device)
            self.codebook.weight[unused] = z[index] + .01 * torch.randn_like(z[index])
        self.usage.zero_()
        return count


class CodecBundle(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dims = {"human": human_dim(config.dataset), "object": 9, "contact": 690}
        self.models = nn.ModuleDict({k: MotionVQVAE(dim, config, k == "contact") for k, dim in dims.items()})

    @torch.no_grad()
    def fit_normalization(self, dataset):
        count, sums, squares = 0, {}, {}
        for batch in dataset.normalization_batches():
            count += int(batch["valid_frames"].sum())
            for key in ("human", "object"):
                values = batch[key][batch["valid_frames"]].double()
                sums[key] = sums.get(key, 0) + values.sum(0)
                squares[key] = squares.get(key, 0) + values.square().sum(0)
        if not count:
            raise ValueError("Empty train split")
        for key in sums:
            mean = sums[key] / count
            self.models[key].mean.copy_(mean.float())
            self.models[key].std.copy_((squares[key] / count - mean.square()).clamp_min(0).sqrt().clamp_min(.05).float())

    @torch.no_grad()
    def encode(self, batch):
        return {k: self.models[k].encode(batch[k], batch["valid_frames"]) for k in MODALITIES}

    def loss(self, batch):
        total, stats = 0, {}
        for key in MODALITIES:
            value, detail = self.models[key].loss(batch[key], batch["valid_frames"])
            total = total + value
            stats.update({key + "_" + k: float(v.detach()) for k, v in detail.items()})
        return total, stats


def canonicalize(human, obj, dataset):
    """Project generated rotations and fixed conventions, without GT body/shape inputs."""
    n = beta_count(dataset)
    pose = matrix_to_rotation6d(rotation6d_to_matrix(human[:, n:-3].reshape(-1, 52, 6)))
    if dataset == "omomo":
        pose[:, 22:] = pose.new_tensor([1, 0, 0, 0, 1, 0])
    human = torch.cat((human[:, :n].mean(0, keepdim=True).expand(len(human), -1),
                       pose.flatten(1), human[:, -3:]), -1)
    obj = torch.cat((obj[:, :3], matrix_to_rotation6d(rotation6d_to_matrix(obj[:, 3:]))), -1)
    return human, obj


def load_codecs(path, dataset=None, support_hash=None, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("architecture") != ARCHITECTURE or saved.get("kind") != "vq_codecs":
        raise ValueError("Expected token_ar VQ checkpoint, not diffusion/contact AE weights")
    config = CodecConfig(**saved["codec_config"])
    if (dataset and dataset != config.dataset) or (support_hash and saved["support_sha256"] != support_hash):
        raise ValueError("VQ checkpoint dataset/contact support mismatch")
    codec = CodecBundle(config)
    codec.load_state_dict(saved["model"])
    return codec.to(device).eval().requires_grad_(False), saved


def codec_payload(codec, metadata):
    return {"architecture": ARCHITECTURE, "kind": "vq_codecs", "codec_config": asdict(codec.config),
            "model": codec.state_dict(), **metadata}
