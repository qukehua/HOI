"""Generate budget-matched configurations; run every variant with multiple seeds."""
import argparse
from copy import deepcopy
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="configs/unified.yaml")
    parser.add_argument("--output", default="configs/ablations")
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    variants = {"mask": (False, False), "mask_coarse": (False, True), "relation": (True, True)}
    for heldout in (False, True):
        for name, (relations, two_stage) in variants.items():
            config = deepcopy(base)
            label = name + ("_heldout_HO" if heldout else "")
            config["output"] = "runs/" + label
            config["model"].update(relations=relations, two_stage=two_stage)
            config["holdout_signatures"] = ["H+O"] if heldout else []
            (out / (label + ".yaml")).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print("Generated six training configs. Evaluate each with projection_steps=0 AND 30.")
    print("Equal training steps, widths, data and masks; report parameter counts and sampling/projection runtime.")


if __name__ == "__main__":
    main()
