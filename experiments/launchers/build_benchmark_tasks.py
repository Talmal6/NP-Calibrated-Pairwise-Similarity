"""Build NeighborCache benchmark tasks without SLURM or subprocess concerns."""
from __future__ import annotations

import itertools
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import benchmark_registry as registry

CLI_MODULE = "NeighborCache.region_local_threshold.cli"
VALID_TAU_MODES = {"local", "global", "shrink_local", "cluster_local"}


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: int
    group: str
    dataset: str
    embedder: str
    seed: int
    alpha: float
    tau_mode: str
    n_train: int
    n_calib: int
    n_eval: int
    hadamard: bool
    n_trials: int
    extra_cli_args: tuple[str, ...]
    output_dir: Path
    npz_path: Path


def _make_output_dir(
    exp_root: Path,
    *,
    dataset: str,
    embedder: str,
    seed: int,
    alpha: float,
    tau_mode: str,
    group: str,
) -> Path:
    return (
        Path(exp_root)
        / dataset
        / embedder
        / f"seed_{seed:03d}"
        / f"alpha_{alpha:.3f}_taumode_{tau_mode}"
        / f"group_{group}"
    )


def output_dir(exp_root: Path, task: BenchmarkTask) -> Path:
    return _make_output_dir(
        exp_root,
        dataset=task.dataset,
        embedder=task.embedder,
        seed=task.seed,
        alpha=task.alpha,
        tau_mode=task.tau_mode,
        group=task.group,
    )


def _combination_error(dataset: str, embedder: str, exc: Exception) -> str:
    message = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
    return f"({dataset!r}, {embedder!r}): {message}"


def build_main_suite(
    exp_root: Path,
    datasets: Sequence[str],
    embedders: Sequence[str],
    seeds: Sequence[int],
    alphas: Sequence[float],
    n_trials: int,
    n_train: int,
    n_calib: int,
    n_eval: int,
    repo_root: Path,
    *,
    tau_mode: str = "global",
    hadamard: bool | None = None,
    extra_cli_args: Sequence[str] = (),
    group: str = "main",
) -> list[BenchmarkTask]:
    if tau_mode not in VALID_TAU_MODES:
        raise ValueError(f"Unknown tau_mode {tau_mode!r}; expected one of {sorted(VALID_TAU_MODES)}")

    root = Path(repo_root)
    missing: list[str] = []
    resolved: dict[tuple[str, str], Path] = {}
    for dataset, embedder in itertools.product(datasets, embedders):
        try:
            registry.validate_combination(dataset, embedder, root)
            resolved[(dataset, embedder)] = registry.resolve_npz(dataset, embedder, root)
        except (FileNotFoundError, KeyError) as exc:
            missing.append(_combination_error(dataset, embedder, exc))

    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing benchmark NPZ inputs:\n  - {details}")

    tasks: list[BenchmarkTask] = []
    task_id = 1
    for dataset, embedder, seed, alpha in itertools.product(datasets, embedders, seeds, alphas):
        task_hadamard = registry.DATASETS[dataset].hadamard_default if hadamard is None else bool(hadamard)
        out_dir = _make_output_dir(
            Path(exp_root),
            dataset=dataset,
            embedder=embedder,
            seed=int(seed),
            alpha=float(alpha),
            tau_mode=tau_mode,
            group=group,
        )
        tasks.append(
            BenchmarkTask(
                task_id=task_id,
                group=group,
                dataset=dataset,
                embedder=embedder,
                seed=int(seed),
                alpha=float(alpha),
                tau_mode=tau_mode,
                n_train=int(n_train),
                n_calib=int(n_calib),
                n_eval=int(n_eval),
                hadamard=task_hadamard,
                n_trials=int(n_trials),
                extra_cli_args=tuple(str(arg) for arg in extra_cli_args),
                output_dir=out_dir,
                npz_path=resolved[(dataset, embedder)],
            )
        )
        task_id += 1
    return tasks


def task_to_cli_args(task: BenchmarkTask) -> list[str]:
    args = [
        "--data",
        str(task.npz_path),
        "--region_key",
        registry.DATASETS[task.dataset].region_key,
        "--alpha",
        f"{task.alpha:.4g}",
        "--tau_mode",
        task.tau_mode,
        "--n_trials",
        str(task.n_trials),
        "--seed",
        str(task.seed),
        "--n_train",
        str(task.n_train),
        "--n_calib",
        str(task.n_calib),
        "--n_eval",
        str(task.n_eval),
        "--run_name",
        task.output_dir.name,
    ]
    if task.hadamard:
        args.append("--hadamard_preprocess")
    args.extend(task.extra_cli_args)
    return args


def task_to_command_line(python_bin: str, task: BenchmarkTask) -> str:
    out_dir = shlex.quote(str(task.output_dir))
    run_log = shlex.quote(str(task.output_dir / "run.log"))
    complete = shlex.quote(str(task.output_dir / "_COMPLETE"))
    cli = [python_bin, "-m", CLI_MODULE, *task_to_cli_args(task)]
    cli_text = " ".join(shlex.quote(str(part)) for part in cli)
    return f"set -e ; mkdir -p {out_dir} ; {cli_text} > {run_log} 2>&1 && touch {complete}"


def is_complete(task: BenchmarkTask) -> bool:
    return (task.output_dir / "_COMPLETE").exists()
