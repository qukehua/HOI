"""Train the current native-parameter model with restartable checkpoints."""
import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from .parametric.train import run_training


def make_loader(dataset, config, epoch, shuffle=True):
    generator = torch.Generator().manual_seed(config.get("seed", 42) + epoch)
    # Replay caption draws from the start of an epoch when skipping resumed batches,
    # without consuming the model RNG restored from the checkpoint. Worker processes
    # receive their own reproducible torch RNG seed from DataLoader instead.
    dataset.text_generator = (torch.Generator().manual_seed(config.get("seed", 42) + epoch)
                              if config.get("workers", 0) == 0 else None)
    sampler = None
    if shuffle and config.get("balance_datasets", True):
        labels = [dataset.records[i]["dataset"] for i, _, _ in dataset.windows]
        count = Counter(labels)
        sampler = WeightedRandomSampler([1 / count[label] for label in labels], len(labels),
                                        replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=config.get("batch_size", 8),
                      shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=config.get("workers", 0), generator=generator, drop_last=False)


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source, 1 - decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def init_swanlab(config, output, run_info):
    if not config.get("use_swanlab", False):
        return None
    try:
        import swanlab
    except ImportError:
        print("SwanLab is enabled but swanlab is not installed. "
              "Install it with `pip install swanlab` or `pip install '.[swanlab]'`.", flush=True)
        return None

    api_key = config.get("swanlab_api_key")
    if api_key in (None, "", "null", "None"):
        import os
        api_key = os.environ.get("SWANLAB_API_KEY")
    if api_key:
        try:
            swanlab.login(api_key=api_key, save=True)
        except Exception as exc:
            print(f"SwanLab login failed, continuing without SwanLab: {exc}", flush=True)
            return None

    run_name = config.get("swanlab_run_name") or output.name
    logged_config = {key: value for key, value in config.items() if key != "swanlab_api_key"}
    kwargs = {
        "project": config.get("swanlab_project", "HOI"),
        "experiment_name": run_name,
        "mode": config.get("swanlab_mode", "online"),
        "config": {**logged_config, "run_info": run_info},
        "logdir": str(output / "swanlog"),
    }
    workspace = config.get("swanlab_workspace")
    if workspace not in (None, "", "null", "None"):
        kwargs["workspace"] = workspace
    if config.get("swanlab_id"):
        kwargs["id"] = config["swanlab_id"]
        kwargs["resume"] = config.get("swanlab_resume", "allow")
    try:
        return swanlab.init(**kwargs)
    except Exception as exc:
        print(f"SwanLab initialization failed, continuing without SwanLab: {exc}", flush=True)
        return None


def log_swanlab(run, payload, step):
    if run is None:
        return
    run.log(payload, step=step)


def finish_swanlab(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", help="Trusted checkpoint produced by this project")
    parser.add_argument("--swanlab", dest="use_swanlab", action="store_true",
                        help="Enable SwanLab logging")
    parser.add_argument("--no-swanlab", dest="use_swanlab", action="store_false",
                        help="Disable SwanLab logging")
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-run-name")
    parser.add_argument("--swanlab-workspace")
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline", "disabled"))
    parser.add_argument("--progress", dest="progress", action="store_true",
                        help="Show a tqdm training progress bar")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="Disable the tqdm training progress bar")
    parser.set_defaults(use_swanlab=None, progress=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for field in ("manifest", "output", "max_steps"):
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    if args.use_swanlab is not None:
        config["use_swanlab"] = args.use_swanlab
    if args.progress is not None:
        config["progress"] = args.progress
    for field, key in (("swanlab_project", "swanlab_project"),
                       ("swanlab_run_name", "swanlab_run_name"),
                       ("swanlab_workspace", "swanlab_workspace"),
                       ("swanlab_mode", "swanlab_mode")):
        value = getattr(args, field)
        if value is not None:
            config[key] = value
    print(run_training(config, args.resume))


if __name__ == "__main__":
    main()
