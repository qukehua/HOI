"""Pinned upstream provenance and lightweight access to copied primitives."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess


UPSTREAMS = {
    "omomo": {
        "url": "https://github.com/lijiaman/omomo_release.git",
        "commit": "e9d8c52f41866ed0c07ac574cb8f7f9424b6d1e7",
        "code_license": "MIT",
    },
    "kimodo": {
        "url": "https://github.com/nv-tlabs/kimodo.git",
        "commit": "58e781898b3d7e328a676a75d3e338c45dce3ad9",
        "code_license": "Apache-2.0",
    },
    "tridi": {
        "url": "https://github.com/ptrvilya/tridi.git",
        "commit": "afa9631dc2b3a250588ab64026eeaa37f18f0d38",
        "code_license": "MIT",
        "primitive_source": "tridi/model/denoising/transformer_uni_3.py",
        "primitive_source_git_blob": "9115e4abe94956d0f6b8708ca6d1a734fbd4d2ad",
    },
}


def checkout_report(path: str | Path, name: str) -> dict:
    """Inspect an upstream without importing it, modifying it, or using the network."""
    if name not in UPSTREAMS:
        raise ValueError(f"Unknown upstream: {name}")
    path = Path(path).resolve()
    report = {"name": name, "path": str(path), **UPSTREAMS[name], "present": path.is_dir()}
    if not path.is_dir() or not (path / ".git").exists():
        report.update(pinned=False, error="Missing Git checkout")
        return report
    try:
        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(path), *args], text=True, stderr=subprocess.PIPE
            ).strip()
        actual = git("rev-parse", "HEAD")
        report.update(
            actual_commit=actual,
            remote=git("remote", "get-url", "origin"),
            dirty=bool(git("status", "--porcelain")),
            pinned=actual == UPSTREAMS[name]["commit"],
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        report.update(pinned=False, error=str(exc))
    return report


def verify_tridi_primitives(checkout: str | Path) -> dict[str, bool]:
    """Compare ASTs of vendored functions to the source, ignoring formatting only."""
    source = Path(checkout) / UPSTREAMS["tridi"]["primitive_source"]
    vendored = Path(__file__).with_name("_tridi_primitives.py")
    def definitions(path):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
    original, local = definitions(source), definitions(vendored)
    return {name: original.get(name) == local.get(name) for name in ("Projection", "get_timestep_embedding")}


def get_projection_class():
    """Load the exact upstream Projection without importing TriDi's asset stack."""
    from ._tridi_primitives import Projection
    return Projection


def get_timestep_embedding(embed_dim, timesteps, device):
    """Call TriDi's exact timestep embedding (requires only torch and numpy)."""
    from ._tridi_primitives import get_timestep_embedding as implementation
    return implementation(embed_dim, timesteps, device)
