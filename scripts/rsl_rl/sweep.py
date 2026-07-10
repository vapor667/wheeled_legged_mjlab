"""Run RSL-RL experiment sweeps from a small YAML file."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
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
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seeds to run.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have a final checkpoint.")
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fail-fast", action="store_true", help="Stop the sweep after the first failed run.")
    args = parser.parse_args()

    sweep = _load_sweep(args.sweep)
    run_specs = list(
        _expand_runs(
            sweep,
            only=set(args.only) if args.only else None,
            seeds=set(args.seeds) if args.seeds else None,
        )
    )
    if args.skip_existing:
        run_specs = [spec for spec in run_specs if not _has_final_checkpoint(spec)]
    if args.dry_run:
        for spec in run_specs:
            print(f"{spec['name']}: task={spec['base_task']} seed={spec['seed']}")
        print(f"Total runs: {len(run_specs)}")
        return

    if not args.in_process:
        failed = _run_specs_in_subprocesses(args.sweep, run_specs, fail_fast=args.fail_fast)
        if failed:
            raise SystemExit(1)
        return

    for spec in run_specs:
        _run_one(spec)


def _load_sweep(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"Sweep file {path} must contain a YAML mapping")
    return data


def _expand_runs(
    sweep: Mapping[str, Any],
    only: set[str] | None = None,
    seeds: set[int] | None = None,
):
    sweep_name = str(sweep["name"])
    base_task = str(sweep["base_task"])
    seed_filter = seeds
    sweep_seeds = [int(seed) for seed in sweep.get("seeds", [0])]
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
        for seed in sweep_seeds:
            if seed_filter is not None and seed not in seed_filter:
                continue
            yield {
                "sweep": sweep_name,
                "variant": variant_name,
                "base_task": base_task,
                "seed": seed,
                "name": f"{sweep_name}/{variant_name}_seed{seed}",
                "max_iterations": sweep.get("max_iterations"),
                "overrides": overrides,
            }


def _run_specs_in_subprocesses(sweep_path: Path, run_specs: list[Mapping[str, Any]], *, fail_fast: bool) -> list[str]:
    failed: list[str] = []
    for spec in run_specs:
        print(f"[SWEEP] Starting {spec['name']}", flush=True)
        status_path = _status_path(spec)
        _write_status(
            status_path,
            {
                "status": "running",
                "name": spec["name"],
                "base_task": spec["base_task"],
                "variant": spec["variant"],
                "seed": spec["seed"],
            },
        )
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(sweep_path),
                "--in-process",
                "--only",
                str(spec["variant"]),
                "--seeds",
                str(spec["seed"]),
            ],
            cwd=REPO_ROOT,
            env=dict(os.environ),
            check=False,
        )
        if result.returncode == 0:
            _write_status(
                status_path,
                {
                    "status": "completed",
                    "name": spec["name"],
                    "base_task": spec["base_task"],
                    "variant": spec["variant"],
                    "seed": spec["seed"],
                    "returncode": result.returncode,
                },
            )
            print(f"[SWEEP] Completed {spec['name']}", flush=True)
        else:
            failed.append(str(spec["name"]))
            _write_status(
                status_path,
                {
                    "status": "failed",
                    "name": spec["name"],
                    "base_task": spec["base_task"],
                    "variant": spec["variant"],
                    "seed": spec["seed"],
                    "returncode": result.returncode,
                },
            )
            print(f"[SWEEP] Failed {spec['name']} with return code {result.returncode}", flush=True)
            if fail_fast:
                break
    if failed:
        print("[SWEEP] Failed runs:", flush=True)
        for name in failed:
            print(f"  - {name}", flush=True)
    return failed


def _has_final_checkpoint(spec: Mapping[str, Any]) -> bool:
    max_iterations = spec.get("max_iterations")
    if max_iterations is None:
        return False
    final_checkpoint = f"model_{int(max_iterations) - 1}.pt"
    pattern = f"*_{spec['variant']}_seed{spec['seed']}/{final_checkpoint}"
    return any((Path("logs") / "rsl_rl" / str(spec["sweep"])).glob(pattern))


def _run_one(spec: Mapping[str, Any]) -> None:
    train_cfg = TrainConfig.from_task(spec["base_task"])
    train_cfg = copy.deepcopy(train_cfg)
    apply_experiment_overrides(train_cfg.env, train_cfg.agent, spec["overrides"])
    _apply_run_metadata(train_cfg, spec)
    _write_resolved_config(spec, train_cfg)
    launch_training(task_id=spec["base_task"], args=train_cfg)


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
    out_dir = _result_dir(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(out_dir / "resolved_env.yaml", asdict(train_cfg.env))
    dump_yaml(out_dir / "resolved_agent.yaml", asdict(train_cfg.agent))
    dump_yaml(out_dir / "variant.yaml", dict(spec))


def _result_dir(spec: Mapping[str, Any]) -> Path:
    return Path("experiments") / "results" / str(spec["sweep"]) / f"{spec['variant']}_seed{spec['seed']}"


def _status_path(spec: Mapping[str, Any]) -> Path:
    out_dir = _result_dir(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "status.yaml"


def _write_status(path: Path, status: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(status), stream, sort_keys=False)


if __name__ == "__main__":
    main()
