"""Run RSL-RL experiment sweeps from a small YAML file."""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mjlab.tasks  # noqa: F401
import wheeled_legged_mjlab  # noqa: F401
from mjlab.utils.os import dump_yaml
from train import TrainConfig, launch_training
from wheeled_legged_mjlab.experiments import apply_experiment_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path, help="Path to sweep YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded runs without launching training.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional variant names to run.")
    args = parser.parse_args()

    sweep = _load_sweep(args.sweep)
    run_specs = list(_expand_runs(sweep, only=set(args.only) if args.only else None))
    if args.dry_run:
        for spec in run_specs:
            print(f"{spec['name']}: task={spec['base_task']} seed={spec['seed']}")
        print(f"Total runs: {len(run_specs)}")
        return

    for spec in run_specs:
        train_cfg = TrainConfig.from_task(spec["base_task"])
        train_cfg = copy.deepcopy(train_cfg)
        apply_experiment_overrides(train_cfg.env, train_cfg.agent, spec["overrides"])
        _apply_run_metadata(train_cfg, spec)
        _write_resolved_config(spec, train_cfg)
        launch_training(task_id=spec["base_task"], args=train_cfg)


def _load_sweep(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"Sweep file {path} must contain a YAML mapping")
    return data


def _expand_runs(sweep: Mapping[str, Any], only: set[str] | None = None):
    sweep_name = str(sweep["name"])
    base_task = str(sweep["base_task"])
    seeds = [int(seed) for seed in sweep.get("seeds", [0])]
    variants = sweep.get("variants", {})
    if not isinstance(variants, Mapping):
        raise ValueError("sweep.variants must be a mapping")

    for variant_name, variant in variants.items():
        if only is not None and variant_name not in only:
            continue
        variant = variant or {}
        if not isinstance(variant, Mapping):
            raise ValueError(f"variant {variant_name!r} must be a mapping")
        overrides = {
            "env": copy.deepcopy(variant.get("env", {})),
            "rl": copy.deepcopy(variant.get("rl", {})),
        }
        for seed in seeds:
            yield {
                "sweep": sweep_name,
                "variant": variant_name,
                "base_task": base_task,
                "seed": seed,
                "name": f"{sweep_name}/{variant_name}_seed{seed}",
                "max_iterations": sweep.get("max_iterations"),
                "overrides": overrides,
            }


def _apply_run_metadata(train_cfg: TrainConfig, spec: Mapping[str, Any]) -> None:
    train_cfg.env.seed = int(spec["seed"])
    train_cfg.agent.seed = int(spec["seed"])
    if spec.get("max_iterations") is not None:
        train_cfg.agent.max_iterations = int(spec["max_iterations"])
    train_cfg.agent.experiment_name = str(spec["sweep"])
    train_cfg.agent.run_name = f"{spec['variant']}_seed{spec['seed']}"
    if hasattr(train_cfg.agent, "trial_message"):
        train_cfg.agent.trial_message = str(spec["overrides"])


def _write_resolved_config(spec: Mapping[str, Any], train_cfg: TrainConfig) -> None:
    out_dir = Path("experiments") / "results" / str(spec["sweep"]) / f"{spec['variant']}_seed{spec['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(out_dir / "resolved_env.yaml", asdict(train_cfg.env))
    dump_yaml(out_dir / "resolved_agent.yaml", asdict(train_cfg.agent))
    dump_yaml(out_dir / "variant.yaml", dict(spec))


if __name__ == "__main__":
    main()
