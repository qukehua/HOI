"""Re-evaluate a saved native prediction with its checkpoint and dataset."""
import argparse
import json

import torch

from .parametric.evaluate import evaluate_prediction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-start", type=int, help="Default: window_start saved in the prediction")
    parser.add_argument("--output")
    parser.add_argument("--profile", choices=("tridi", "native", "auto"), default="tridi")
    parser.add_argument("--eval-fps", type=float, help="0 keeps all native frames; tridi defaults to 1fps")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    report = evaluate_prediction(args.prediction, args.checkpoint, args.reference, args.manifest,
                                 args.output, start=args.reference_start, profile=args.profile, eval_fps=args.eval_fps)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
