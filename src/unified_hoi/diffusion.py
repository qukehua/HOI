"""Independent H/O/C noise levels and mask-preserving DDIM sampling."""
from __future__ import annotations

import math
from dataclasses import dataclass
import torch

from .geometry import matrix_to_rotation6d, rotation6d_to_matrix
from .normalization import MODALITIES
from .objectives import geometric_losses, reconstruction_loss


@dataclass
class NormalizedControls:
    """Internal model-space values; physical binary-contact validation is done before encoding."""
    values: dict
    masks: dict
    valid_frames: torch.Tensor
    signatures: tuple | None

    def apply(self, states):
        return {k: torch.where(self.masks[k], self.values[k], states[k]) for k in MODALITIES}


def normalized_controls(controls, normalizer):
    controls.validate()
    return NormalizedControls(normalizer.encode(controls.values), controls.masks,
                              controls.valid_frames, controls.signatures)


class HOIDiffusion:
    def __init__(self, steps=1000):
        if steps < 2:
            raise ValueError("Diffusion requires at least two timesteps")
        self.steps = steps
        t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        alpha = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
        self.alpha = (alpha / alpha[0]).clamp_min(1e-8).float()

    def coefficients(self, times, tensor):
        return self.alpha.to(tensor.device)[times].reshape(-1, *([1] * (tensor.ndim - 1)))

    def training_loss(self, model, batch, controls, geometry_weight=.05, generator=None):
        clean = model.normalizer.encode(batch)
        cond = normalized_controls(controls, model.normalizer)
        b = clean["object"].shape[0]
        ts = torch.randint(1, self.steps + 1, (b, 3), device=clean["object"].device, generator=generator)
        noisy = {}
        for i, key in enumerate(MODALITIES):
            valid = controls.valid_frames.reshape(b, -1, *([1] * (clean[key].ndim - 2)))
            all_known = (controls.masks[key] | ~valid).flatten(1).all(1)
            ts[:, i] = torch.where(all_known, 0, ts[:, i])
            a = self.coefficients(ts[:, i], clean[key])
            noise = torch.randn(clean[key].shape, device=clean[key].device, generator=generator)
            noisy[key] = a.sqrt() * clean[key] + (1 - a).sqrt() * noise
        prediction = model(noisy, ts, cond, batch)
        rec = reconstruction_loss(prediction, clean, controls.masks, controls.valid_frames)
        total = sum(rec.values())
        metrics = {"denoise_" + k: v for k, v in rec.items()}
        if geometry_weight:
            physical = model.normalizer.decode(cond.apply(prediction))
            geo = geometric_losses(physical, batch, batch["contact"])
            total = total + geometry_weight * (geo["contact"] + geo["noncontact"] + geo["fk"]
                                                + .1 * geo["floor"] + .01 * geo["slip"])
            metrics.update(geo)
        metrics["loss"] = total
        return total, {k: float(v.detach()) for k, v in metrics.items()}

    @torch.no_grad()
    def sample(self, model, batch, controls, steps=50, seed=0, text_guidance=1., projection_steps=0):
        if not 1 <= steps <= self.steps:
            raise ValueError("Sampling steps must be between 1 and training diffusion steps")
        device = batch["object_points"].device
        generator = torch.Generator(device=device).manual_seed(seed)
        cond = normalized_controls(controls, model.normalizer)
        x = {k: torch.randn(v.shape, device=device, generator=generator) for k, v in cond.values.items()}
        x = cond.apply(x)
        schedule = torch.linspace(self.steps, 0, steps + 1, device=device).round().long()
        b = controls.valid_frames.shape[0]
        model.eval()
        for current, following in zip(schedule[:-1], schedule[1:]):
            times = current.expand(b, 3).clone()
            for i, k in enumerate(MODALITIES):
                valid = controls.valid_frames.reshape(b, -1, *([1] * (x[k].ndim - 2)))
                times[:, i] = torch.where((controls.masks[k] | ~valid).flatten(1).all(1), 0, times[:, i])
            pred = model(x, times, cond, batch)
            if text_guidance != 1:
                uncond_batch = {**batch, "text_features": torch.zeros_like(batch["text_features"])}
                uncond = model(x, times, cond, uncond_batch)
                pred = {k: uncond[k] + text_guidance * (pred[k] - uncond[k]) for k in pred}
            pred = cond.apply(pred)
            for k in x:
                a = self.alpha.to(device)[current]
                a_next = self.alpha.to(device)[following]
                epsilon = (x[k] - a.sqrt() * pred[k]) / (1 - a).sqrt().clamp_min(1e-6)
                x[k] = a_next.sqrt() * pred[k] + (1 - a_next).sqrt() * epsilon
            x = cond.apply(x)
        output = model.normalizer.decode(x)
        output["contact"] = output["contact"].clamp(0, 1)
        for k in ("human", "object"):
            # Reproject only wholly unspecified rotations; never silently alter observed 6D components.
            rot = matrix_to_rotation6d(rotation6d_to_matrix(output[k][..., 3:]))
            free = ~controls.masks[k][..., 3:].any(-1, keepdim=True)
            output[k] = torch.cat((output[k][..., :3], torch.where(free, rot, output[k][..., 3:])), -1)
        output = controls.apply(output)
        if projection_steps:
            output = project_relations(output, controls, batch, steps=projection_steps)
        for k in output:
            valid = controls.valid_frames.reshape(b, -1, *([1] * (output[k].ndim - 2)))
            output[k] = output[k] * valid
        return output


def project_relations(states, controls, batch, steps=30, learning_rate=.01):
    """Optimize unobserved H/O coordinates only; explicitly known 0-contact is repulsive.

    Partial contact constraints propagate across both entities. Hard anchors are reinserted
    at every iteration. Incompatible constraints remain violations in the returned metrics.
    """
    with torch.enable_grad():
        variables = {k: states[k].detach().clone().requires_grad_(True) for k in ("human", "object")}
        optimizer = torch.optim.Adam(list(variables.values()), lr=learning_rate)
        target = torch.where(controls.masks["contact"], controls.values["contact"],
                             states["contact"].detach().clamp(0, 1))
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            current = controls.apply({**variables, "contact": states["contact"].detach()})
            losses = geometric_losses(current, batch, target)
            fidelity = sum((current[k] - states[k].detach()).square().mean() for k in variables)
            loss = (10 * losses["contact"] + 10 * losses["noncontact"] + losses["fk"]
                    + .1 * losses["floor"] + .005 * losses["slip"] + .2 * fidelity)
            loss.backward()
            optimizer.step()
        result = {k: variables[k].detach() if k in variables else v.detach() for k, v in states.items()}
        for key in variables:
            rotation = matrix_to_rotation6d(rotation6d_to_matrix(result[key][..., 3:]))
            free = ~controls.masks[key][..., 3:].any(-1, keepdim=True)
            result[key] = torch.cat((result[key][..., :3],
                                     torch.where(free, rotation, result[key][..., 3:])), -1)
        result = controls.apply(result)
    return result
