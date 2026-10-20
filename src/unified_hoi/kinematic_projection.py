"""Pose-space refinement and explicit validation of final decoded anchors.

Position observations are constraints on FK, never values pasted over FK.
Fixed root/rotation components and object features remain fixed. Unresolved
position constraints are reported, not declared infeasible by this local solver.
"""
import warnings

import torch

from .geometry import SMPL_PARENTS, human_fk, matrix_to_rotation6d, rotation6d_to_matrix
from .objectives import geometric_losses, masked_mean

ANCHOR_TOLERANCE_M = .001


def kinematic_diagnostics(states, controls, batch, tolerance=ANCHOR_TOLERANCE_M):
    human = states["human"]
    fk = human_fk(human, batch["rest_offsets"], batch.get("parents", SMPL_PARENTS))
    valid = controls.valid_frames[..., None, None]
    mask = controls.masks["human"][..., :3] & valid
    component = (fk - controls.values["human"][..., :3]).abs()
    # Norm over observed axes only: a partial-axis anchor does not constrain other axes.
    error = torch.linalg.vector_norm(torch.where(mask, component, 0.), dim=-1)
    supported = mask.any(-1)
    maximum = float(error[supported].max()) if supported.any() else 0.
    residual = torch.linalg.vector_norm(human[..., :3] - fk, dim=-1)
    return {"human_decoded_anchor_count": int(supported.sum()),
            "human_decoded_anchor_mean_m": float(error[supported].mean()) if supported.any() else None,
            "human_decoded_anchor_max_m": maximum if supported.any() else None,
            "human_decoded_anchor_unsatisfied_count": int(((error > tolerance) & supported).sum()),
            "human_decoded_anchor_tolerance_m": tolerance,
            "human_decoded_anchor_status": ("none" if not supported.any() else
                                             "satisfied" if maximum <= tolerance else "unsatisfied"),
            "human_fk_error_max_m": float(residual[controls.valid_frames].max())}


def project_kinematics(states, controls, batch, *, steps=90, contact_steps=0,
                       tolerance=ANCHOR_TOLERANCE_M, warn=True):
    """Return an FK-consistent pose, with optional constrained IK/contact refinement.

    Increasing-penalty LBFGS gives position constraints priority over fidelity,
    temporal correction smoothing, and optional contact terms. This is a local
    numerical solver, not a guarantee that every requested anchor is feasible.
    """
    if steps < 0 or contact_steps < 0:
        raise ValueError("IK/projection steps must be nonnegative")
    valid = controls.valid_frames.bool()
    original = {k: v.detach().clone() for k, v in controls.apply(states).items()}
    hmask, omask = controls.masks["human"], controls.masks["object"]
    target = controls.values["human"]
    position_mask = hmask[..., :3] & valid[..., None, None]
    nonroot = position_mask.clone()
    nonroot[..., 0, :] = False
    rot0 = original["human"][..., 3:]
    root0 = original["human"][..., 0, :3]
    initial_rotations = rotation6d_to_matrix(rot0)
    with torch.enable_grad():
        rotation = rot0.clone().requires_grad_(True)
        root = root0.clone().requires_grad_(True)
        obj = original["object"].clone().requires_grad_(True)

        def decoded():
            r = torch.where(hmask[..., 3:], target[..., 3:], rotation)
            p = torch.where(hmask[..., 0, :3], target[..., 0, :3], root)
            seed = torch.cat((p[..., None, :].expand(*r.shape[:-1], 3), r), -1)
            xyz = human_fk(seed, batch["rest_offsets"])
            # Never call controls.apply on these FK-derived nonroot positions.
            return {"human": torch.cat((xyz, r), -1),
                    "object": torch.where(omask, controls.values["object"], obj),
                    "contact": original["contact"]}

        current = decoded()
        error = torch.linalg.vector_norm(torch.where(position_mask,
                                         current["human"][..., :3] - target[..., :3], 0.), dim=-1)
        needs_ik = bool(nonroot.any() and (error[nonroot.any(-1)] > tolerance * .1).any())
        budget = max(steps if needs_ik else 0, contact_steps)
        if budget:
            contact_target = torch.where(controls.masks["contact"], controls.values["contact"],
                                         original["contact"].clamp(0, 1))
            # Allocate an exact total iteration budget across up to three penalty stages.
            stages = min(3, budget)
            for stage in range(stages):
                with torch.no_grad():
                    previous_anchor = masked_mean((decoded()["human"][..., :3] - target[..., :3]).square(), position_mask)
                iterations = budget // stages + (stage < budget % stages)
                penalty = (1e2, 1e4, 1e6)[stage + 3 - stages]
                optimizer = torch.optim.LBFGS([rotation, root, obj], lr=1., max_iter=iterations,
                                             history_size=10, line_search_fn="strong_wolfe",
                                             tolerance_grad=1e-10, tolerance_change=1e-12)

                def closure():
                    optimizer.zero_grad()
                    current = decoded()
                    h = current["human"]
                    anchor = masked_mean((h[..., :3] - target[..., :3]).square(), position_mask)
                    delta_r = rotation6d_to_matrix(h[..., 3:]) - initial_rotations
                    delta_p = h[..., 0, :3] - root0
                    loss = penalty * anchor + .001 * masked_mean(delta_r.square(), valid[..., None, None, None])
                    loss = loss + .01 * masked_mean(delta_p.square(), valid[..., None])
                    loss = loss + .01 * masked_mean((current["object"] - original["object"]).square(), valid[..., None])
                    if valid.shape[1] > 1:
                        pairs = valid[:, 1:] & valid[:, :-1]
                        loss = loss + .001 * masked_mean((delta_r[:, 1:] - delta_r[:, :-1]).square(),
                                                        pairs[..., None, None, None])
                        loss = loss + .001 * masked_mean((delta_p[:, 1:] - delta_p[:, :-1]).square(), pairs[..., None])
                    if contact_steps:
                        geo = geometric_losses(current, batch, contact_target)
                        loss = loss + 10 * geo["contact"] + 10 * geo["noncontact"] + .1 * geo["floor"] + .005 * geo["slip"]
                    loss.backward()
                    return loss

                optimizer.step(closure)
                # A straight bone with only its axial coordinate observed has
                # zero first derivative even when bending can reach the target.
                # One deterministic, temporally constant restart breaks that
                # symmetry, without changing observed components or global RNG.
                if stage == 0 and stages > 1 and needs_ik:
                    with torch.no_grad():
                        remaining = masked_mean((decoded()["human"][..., :3] - target[..., :3]).square(), position_mask)
                        if remaining > tolerance**2 and remaining >= previous_anchor * .999:
                            generator = torch.Generator(device=rotation.device).manual_seed(901)
                            jitter = torch.randn((rotation.shape[0], 1, *rotation.shape[2:]),
                                                 generator=generator, device=rotation.device, dtype=rotation.dtype) * .01
                            rotation.add_(jitter * (~hmask[..., 3:] & valid[..., None, None]))
        result = {k: v.detach() for k, v in decoded().items()}
    # Canonicalize only fully free 6D encodings, preserving known components.
    for key in ("human", "object"):
        raw = result[key][..., 3:]
        canonical = matrix_to_rotation6d(rotation6d_to_matrix(raw))
        free = ~controls.masks[key][..., 3:].any(-1, keepdim=True)
        result[key] = torch.cat((result[key][..., :3], torch.where(free, canonical, raw)), -1)
    result["human"] = torch.cat((human_fk(result["human"], batch["rest_offsets"]), result["human"][..., 3:]), -1)
    for key in result:
        result[key] *= valid.reshape(*valid.shape, *([1] * (result[key].ndim - 2)))
    report = kinematic_diagnostics(result, controls, batch, tolerance)
    if warn and report["human_decoded_anchor_status"] == "unsatisfied":
        warnings.warn(f"IK left {report['human_decoded_anchor_unsatisfied_count']} position anchors unsatisfied; "
                      f"max residual {report['human_decoded_anchor_max_m'] * 1000:.3f} mm. "
                      "Final xyz remains FK-consistent; requested anchors are retained separately.",
                      UserWarning, stacklevel=2)
    return result
