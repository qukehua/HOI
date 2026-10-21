"""Sample full H/O/C modes with the current native-parameter model."""
import argparse

import torch

from .parametric.diffusion import MODES
from .parametric.sample import sample
from .runtime import device_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted parametric_v1 checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--mode", choices=MODES, required=True, help="H/O/C: 1=generate, 0=condition")
    parser.add_argument("--text-conditioned", action="store_true", help="Use the manifest's cached text")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(sample(args.checkpoint, args.manifest, args.reference, args.output, args.mode,
                 frames=args.frames, start=args.start, steps=args.steps, seed=args.seed,
                 device=device_for(args.device), text_conditioned=args.text_conditioned))


if __name__ == "__main__":
    main()
