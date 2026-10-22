"""Native-parameter diffusion with exact observed modalities and surface objectives."""
import math

import torch

from ..geometry import matrix_to_rotation6d, rotation6d_to_matrix, object_to_world, nearest_surface
from .body import beta_count, split_human
from .contact_supervision import (contact_loss_options, decoded_contact_loss,
                                  balanced_contact_mean, full_mesh_distances)
from .model import MODALITIES

MODES = ("100", "010", "001", "110", "101", "011", "111")
TIMESTEP_SAMPLING = ("shared", "independent")


def controls_for(batch, dataset, mode=None, generator=None, allowed=None):
    pool = tuple(allowed or MODES)
    if not pool or any(m not in MODES for m in pool) or (mode and mode not in MODES):
        raise ValueError("Native controls require H/O/C generated-bit modes")
    b = len(batch["human"])
    selected = [mode or pool[int(torch.randint(len(pool), (), generator=generator))] for _ in range(b)]
    masks = {k: torch.zeros_like(batch[k], dtype=torch.bool) for k in MODALITIES}
    for i, chosen in enumerate(selected):
        for k, bit in zip(MODALITIES, chosen):
            masks[k][i] = bit == "0"
    if dataset == "omomo":
        # OMOMO has no hand annotations: neutral hands are a declared convention,
        # not a hidden ground-truth condition or a claim of learned hand motion.
        n = beta_count(dataset)
        masks["human"][..., n + 22 * 6:n + 52 * 6] = True
    return masks


def apply_masks(states, values, masks):
    return {k: torch.where(masks[k], values[k], states[k]) for k in MODALITIES}


def training_timesteps(masks, steps, sampling="shared", generator=None):
    """One noise level per example for generated modalities; observed modalities use zero.

    Sharing the timestep does not share the Gaussian noise, drawn separately below.
    Partial conditions (e.g. OMOMO neutral hands) do not mark the whole human as known.
    """
    if sampling not in TIMESTEP_SAMPLING:
        raise ValueError(f"timestep_sampling must be one of {TIMESTEP_SAMPLING}")
    human_mask = masks["human"]
    columns = 1 if sampling == "shared" else len(MODALITIES)
    times = torch.randint(1, steps + 1, (len(human_mask), columns),
                          device=human_mask.device, generator=generator)
    if sampling == "shared":
        times = times.expand(-1, len(MODALITIES))
    known = torch.stack([masks[k].flatten(1).all(1) for k in MODALITIES], -1)
    return times.masked_fill(known, 0)


class ParametricDiffusion:
    def __init__(self, steps=1000):
        if steps < 2:
            raise ValueError("Diffusion needs at least two steps")
        self.steps = steps
        t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        a = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
        self.alpha = (a / a[0]).clamp_min(1e-8).float()

    def training_loss(self, model, batch, masks, geometry, config, generator=None):
        contact_options = contact_loss_options(config)
        clean = model.normalizer.encode(batch)
        times = training_timesteps(masks, self.steps, config.get("timestep_sampling", "shared"), generator)
        noisy = {}
        for i, k in enumerate(MODALITIES):
            a = self.alpha.to(times.device)[times[:, i]][:, None, None]
            noise = torch.randn(clean[k].shape, device=times.device, generator=generator)
            noisy[k] = a.sqrt() * clean[k] + (1 - a).sqrt() * noise
        prediction = model(apply_masks(noisy, clean, masks), times, masks, batch)
        valid = batch["valid_frames"]
        losses, diagnostics = {}, {}
        for k in MODALITIES:
            weight = (~masks[k]).to(prediction[k]) + .05 * masks[k]
            # Denominator is valid coordinates, so observed .05 does not cancel.
            support = valid[..., None].expand_as(prediction[k])
            losses["denoise_" + k] = ((prediction[k] - clean[k]).square() * weight * support).sum() / support.sum().clamp_min(1)
        physical = model.normalizer.decode(apply_masks(prediction, clean, masks))
        if contact_options["decoded_bce_weight"]:
            frames = valid & (~masks["contact"]).any(-1)
            losses["contact_bce"], stats = decoded_contact_loss(
                model.codec, physical["contact"], batch["surface_contact"], frames, model.contact_threshold)
            diagnostics.update({"contact_decoded_" + k: v for k, v in stats.items()})
        low = (times[:, :2].max(-1).values <= config.get("geometry_max_t", self.steps // 4))
        indices = (valid & low[:, None]).nonzero()
        if len(indices) and config.get("geometry_weight", 1):
            cap = config.get("geometry_frames", 16)
            choose = torch.linspace(0, len(indices) - 1, min(cap, len(indices)), device=indices.device).long()
            bi, ti = indices[choose].unbind(-1)
            h = physical["human"][bi, ti][:, None]
            o = physical["object"][bi, ti][:, None]
            gender, points = batch["gender"][bi], batch["object_points"][bi]
            vertices, _ = geometry(h, gender)
            with torch.no_grad():
                ref_vertices, _ = geometry(batch["human"][bi, ti][:, None], gender)
                ref_obj = batch["object"][bi, ti][:, None]
                target_vertices = object_to_world(points, ref_obj)
                if contact_options["full_mesh_distance"]:
                    ref_distance = full_mesh_distances(ref_vertices, ref_obj, batch, bi)
                else:
                    ref_distance, _ = nearest_surface(ref_vertices, ref_obj, points)
            pred_vertices = object_to_world(points, o)
            losses["object_v2v"] = (pred_vertices - target_vertices).square().mean()
            losses["body_v2v"] = (vertices - ref_vertices).square().mean()
            if contact_options["full_mesh_distance"]:
                distance = full_mesh_distances(vertices, o, batch, bi)
            else:
                distance, _ = nearest_surface(vertices, o, points)
            # Match GT distances, rather than pulling every positive joint to 2 cm.
            target = batch["surface_contact"][bi, ti][:, None]
            if contact_options["balance_surface_distance"]:
                generated = ((~masks["human"][bi, ti]).any(-1) | (~masks["object"][bi, ti]).any(-1))
                support = generated[:, None, None].expand_as(target)
                losses["surface_distance"], stats = balanced_contact_mean(
                    (distance - ref_distance).square(), target, support)
                diagnostics.update({"contact_distance_" + k: v for k, v in stats.items()})
            else:
                contact_weight = 1 + 4 * target
                losses["surface_distance"] = ((distance - ref_distance).square() * contact_weight).mean()
        total = sum(losses[k] for k in losses if k.startswith("denoise_"))
        weights = {"object_v2v": 10., "body_v2v": 1., "surface_distance": 5.}
        weights.update(config.get("geometry_losses", {}))
        total = total + config.get("geometry_weight", 1.) * sum(weights[k] * losses[k] for k in weights if k in losses)
        if "contact_bce" in losses:
            weighted = contact_options["decoded_bce_weight"] * losses["contact_bce"]
            total = total + weighted
            diagnostics["contact_bce_weighted"] = weighted
        losses["loss"] = total
        return total, {k: float(v.detach()) for k, v in (losses | diagnostics).items()}

    @torch.no_grad()
    def sample(self, model, batch, masks, steps=50, seed=42):
        if not 1 <= steps <= self.steps:
            raise ValueError("Sampling steps outside training schedule")
        model.eval()
        clean = model.normalizer.encode(batch)
        generator = torch.Generator(device=batch["human"].device).manual_seed(seed)
        x = {k: torch.randn(v.shape, device=v.device, generator=generator) for k, v in clean.items()}
        x = apply_masks(x, clean, masks)
        schedule = torch.linspace(self.steps, 0, steps + 1).round().long().tolist()
        alpha = self.alpha.to(batch["human"].device)
        for current, following in zip(schedule[:-1], schedule[1:]):
            times = torch.full((len(x["human"]), 3), current, device=x["human"].device, dtype=torch.long)
            for i, k in enumerate(MODALITIES):
                times[:, i] = torch.where(masks[k].flatten(1).all(1), 0, times[:, i])
            prediction = apply_masks(model(x, times, masks, batch), clean, masks)
            for k in x:
                epsilon = (x[k] - alpha[current].sqrt() * prediction[k]) / (1 - alpha[current]).sqrt().clamp_min(1e-6)
                x[k] = alpha[following].sqrt() * prediction[k] + (1 - alpha[following]).sqrt() * epsilon
            x = apply_masks(x, clean, masks)
        output = model.normalizer.decode(x)
        n = beta_count(model.config.dataset)
        beta, pose, trans = split_human(output["human"], model.config.dataset)
        pose = matrix_to_rotation6d(rotation6d_to_matrix(pose)).flatten(2)
        output["human"] = torch.cat((beta, pose, trans), -1)
        output["object"] = torch.cat((output["object"][..., :3],
                            matrix_to_rotation6d(rotation6d_to_matrix(output["object"][..., 3:]))), -1)
        output = apply_masks(output, batch, masks)
        if output["human"].shape[-1] != n + 315:
            raise ValueError("Invalid output layout")
        return output
