"""Evaluate native checkpoints using isolated 73-joint and surface reconstruction metrics."""
import argparse
import json

import torch

from unified_hoi.parametric.diffusion import MODES
from unified_hoi.parametric.evaluate import benchmark
from unified_hoi.parametric.evaluation_protocol import add_protocol_arguments
from unified_hoi.runtime import device_for


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    add_protocol_arguments(parser)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["010", "100"])
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-conditioned", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--limit", type=int, help="Explicit subset; otherwise evaluate all test windows")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    options = vars(args)
    options["device"] = device_for(options["device"])
    return benchmark(**options)


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False))
