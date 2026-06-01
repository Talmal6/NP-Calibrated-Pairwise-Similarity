"""Submit NeighborCache benchmark tasks as one SLURM array."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import benchmark_registry as registry
from .build_benchmark_tasks import (
    BenchmarkTask,
    build_main_suite,
    is_complete,
    task_to_cli_args,
    task_to_command_line,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SBATCH_FILE = REPO_ROOT / "neighborcache_experiment_array.sbatch"
HEADERS = [
    "task_id", "group", "dataset", "embedder", "seed", "alpha", "tau_mode", "n_train", "n_calib",
    "n_eval", "hadamard", "n_trials", "extra_cli_args", "output_dir", "npz_path",
]
RESULT_CSV = "trial_summary.csv"
AGGREGATE_METRICS = [
    "micro_tpr", "micro_fpr", "macro_tpr", "macro_fpr",
    "train_tpr", "train_fpr", "tau", "tau_mean",
    "train_samples_needed", "train_total_samples", "time_ms",
]
T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


STREAMING_DEFAULT_DATASETS = [
    "SemCacheLMArena", "SemCacheClassification", "SemCacheSearchQueries", "SemCacheCombo",
]
STREAMING_DEFAULT_METHODS = [
    "vcache", "cosine", "ours_whitened_hadamard", "ours_weighted_ensemble", "vcache_whitened_cosine",
]
STREAMING_DEFAULT_DELTA_VALUES = [0.01, 0.02, 0.03, 0.05, 0.08]
STREAMING_DEFAULT_THRESHOLDS = [0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
STREAMING_DEFAULT_TARGET_BUDGETS = [0.01, 0.02, 0.03, 0.05, 0.08]
STREAMING_DEFAULT_HARD_THRESHOLDS = [0.80, 0.85, 0.90]


@dataclass(frozen=True)
class BenchmarkJob:
    job_id: int
    tasks: tuple[BenchmarkTask, ...]
    output_dir: Path


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _rewrite_extra_cli_args(argv: Sequence[str] | None) -> list[str]:
    argv = list(sys.argv[1:] if argv is None else argv)
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--extra-cli-arg" and i + 1 < len(argv):
            out.append(f"--extra-cli-arg={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally submit NeighborCache benchmark experiments as one SLURM array.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add = parser.add_argument
    add("--datasets", nargs="+", default=["wildchat_final", "lmsys_cluster"], help="Dataset registry keys.")
    add(
        "--include-vcache-original-datasets",
        action="store_true",
        help="Also include the official vCache SemBenchmark datasets registered by this launcher.",
    )
    add("--embedders", nargs="+", default=["default"], help="Embedder registry keys.")
    add("--seeds", nargs="+", type=int, default=list(range(20)), help="Seed values.")
    add("--alphas", nargs="+", type=float, default=[0.01, 0.03, 0.05, 0.10], help="Alpha values.")
    add("--tau-mode", default="global", help="Threshold mode passed to the benchmark CLI.")
    add("--n-train", type=int, default=1200, help="Number of training examples.")
    add("--n-calib", type=int, default=1200, help="Number of calibration examples.")
    add("--n-eval", type=int, default=1200, help="Number of evaluation examples.")
    add("--n-trials", type=int, default=1, help="Repetitions per seed.")
    add(
        "--hadamard",
        dest="hadamard",
        action="store_true",
        default=None,
        help="Force Hadamard preprocessing for all datasets instead of using registry defaults.",
    )
    add(
        "--no-hadamard",
        dest="hadamard",
        action="store_false",
        help="Disable Hadamard preprocessing for all datasets.",
    )
    add("--extra-cli-arg", action="append", default=[], help="Extra single argument passed to the benchmark CLI.")
    add("--exp-root", default=str(Path("NeighborCache/results") / f"benchmark_{_timestamp()}"), help="Experiment output root.")
    add("--conda-env", default=os.environ.get("CONDA_ENV_NAME", "ec"), help="Conda environment name.")
    add("--python-bin", default=os.environ.get("PYTHON_BIN"), help="Explicit Python executable.")
    add("--partition", default="", help="SLURM partition.")
    add("--time", default="12:00:00", help="SLURM time limit.")
    add("--cpus-per-task", default="4", help="SLURM CPUs per task.")
    add("--mem", default="16G", help="SLURM memory per task.")
    add("--gpus", default="0", help="SLURM GPU request, such as rtx_4090:1 or 1.")
    add("--account", default="", help="SLURM account.")
    add("--qos", default="", help="SLURM QoS.")
    add(
        "--max-parallel",
        type=int,
        default=0,
        help="Maximum simultaneous array jobs; 0 leaves concurrency uncapped by this launcher.",
    )
    add("--job-name", default="ncache_bench", help="SLURM job name.")
    add("--dry-run", action="store_true", help="Print the matrix and sample command without writing files.")
    add("--submit", action="store_true", help="Submit with sbatch after writing files.")
    add("--no-wait", dest="wait", action="store_false", help="Do not wait for the submitted array to finish.")
    add("--poll-interval", type=float, default=60.0, help="Seconds between SLURM status polls while waiting.")
    add("--wait-timeout", type=float, default=0.0, help="Maximum seconds to wait for the array; 0 disables the timeout.")
    add("--skip-aggregate", action="store_true", help="Do not build final per-dataset aggregate tables.")
    add("--allow-partial-aggregate", action="store_true", help="Build aggregate tables from available completed task outputs.")
    add("--force", action="store_true", help="Re-run tasks even when _COMPLETE exists.")
    add("--only-failed", type=Path, default=None, help="failed.tsv path used to retry matching rows.")
    add("--filter", default="", help="Comma-separated key=value task filter.")
    add("--suite", choices=["smoke", "paper"], default="paper", help="Benchmark suite preset.")
    # --- streaming evaluation (auto-enabled by --include-vcache-original-datasets) ---
    add("--streaming", dest="streaming", action="store_true", default=None,
        help="Submit a streaming evaluation job alongside the benchmark array.")
    add("--no-streaming", dest="streaming", action="store_false",
        help="Disable the streaming evaluation job.")
    add("--vcache-repo-path", default="../vCache",
        help="Path to the vCache source repo used by the streaming evaluation.")
    add("--streaming-datasets", nargs="+", default=STREAMING_DEFAULT_DATASETS,
        help="Datasets for the streaming evaluation.")
    add("--streaming-methods", nargs="+", default=STREAMING_DEFAULT_METHODS,
        help="Methods for the streaming evaluation.")
    add("--streaming-seed", type=int, default=0, help="Random seed for the streaming evaluation.")
    add("--streaming-cache-size", type=int, default=4096, help="Cache size for the streaming evaluation.")
    add("--streaming-raw-log-mode", choices=["full", "minimal", "none"], default="minimal",
        help="Raw decision log verbosity for the streaming evaluation.")
    add("--streaming-delta-values", nargs="+", type=float, default=STREAMING_DEFAULT_DELTA_VALUES,
        help="vCache delta values swept in the streaming evaluation.")
    add("--streaming-thresholds", nargs="+", type=float, default=STREAMING_DEFAULT_THRESHOLDS,
        help="Cosine thresholds swept in the streaming evaluation.")
    add("--streaming-target-budgets", nargs="+", type=float, default=STREAMING_DEFAULT_TARGET_BUDGETS,
        help="Target FPR budgets (alpha) for learned methods in the streaming evaluation.")
    add("--streaming-hard-thresholds", nargs="+", type=float, default=STREAMING_DEFAULT_HARD_THRESHOLDS,
        help="Hard-neighbor cosine thresholds for sub-stream analysis.")
    add("--streaming-time", default=None,
        help="SLURM time limit for the streaming job; defaults to --time.")
    add("--streaming-limit", type=int, default=None,
        help="Debug: max examples per dataset in the streaming evaluation (omit for full stream).")
    return parser.parse_args(_rewrite_extra_cli_args(argv))


def _resolve_python(conda_env: str, python_bin: str | None) -> str:
    if python_bin:
        cmd = [python_bin, "-c", "import sys; print(sys.executable)"]
    else:
        cmd = ["conda", "run", "-n", conda_env, "which", "python"]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(f"Could not resolve Python for Conda env {conda_env!r}.\n{stderr.strip()}") from exc
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Conda env {conda_env!r} resolved no Python executable.")
    return lines[-1]


def _best_stdout(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "NA"


def _abs_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _download_inputs(datasets: Sequence[str], embedders: Sequence[str], python_bin: str) -> None:
    errors: list[str] = []
    for dataset in datasets:
        for embedder in embedders:
            try:
                registry.ensure_npz(dataset, embedder, REPO_ROOT, python_bin)
            except (FileNotFoundError, KeyError, subprocess.CalledProcessError) as exc:
                message = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
                errors.append(f"({dataset!r}, {embedder!r}): {message}")
    if errors:
        raise FileNotFoundError("Missing benchmark NPZ inputs:\n  - " + "\n  - ".join(errors))


def _apply_suite(args: argparse.Namespace) -> None:
    if args.suite == "smoke":
        args.seeds, args.alphas = [0], [0.05]
        args.n_trials = 1
        args.n_train = args.n_calib = args.n_eval = 200
        if args.streaming is None:
            args.streaming = False


def _apply_dataset_presets(args: argparse.Namespace) -> None:
    if not getattr(args, "include_vcache_original_datasets", False):
        if args.streaming is None:
            args.streaming = False
        return
    seen = set()
    datasets = []
    for dataset in [*args.datasets, *registry.VCACHE_ORIGINAL_DATASETS]:
        if dataset in seen:
            continue
        seen.add(dataset)
        datasets.append(dataset)
    args.datasets = datasets
    if args.streaming is None:
        args.streaming = True


def _append_extra_cli_arg_once(args: argparse.Namespace, flag: str) -> None:
    current = [str(arg) for arg in getattr(args, "extra_cli_arg", [])]
    if flag not in current:
        current.append(flag)
    args.extra_cli_arg = current


def _apply_competitor_presets(args: argparse.Namespace) -> None:
    if any(dataset in registry.VCACHE_ORIGINAL_DATASETS for dataset in args.datasets):
        _append_extra_cli_arg_once(args, "--include_vcache_baseline")


def _filter_tasks(tasks: list[BenchmarkTask], filter_text: str) -> list[BenchmarkTask]:
    if not filter_text:
        return tasks
    filters = {}
    for raw in filter_text.split(","):
        if "=" not in raw:
            raise ValueError(f"Invalid filter {raw!r}; expected key=value")
        key, value = raw.split("=", 1)
        filters[key.strip()] = value.strip()
    return [task for task in tasks if all(str(getattr(task, key)) == value for key, value in filters.items())]


def _task_key(task: BenchmarkTask) -> tuple[str, str, int, float, str]:
    return (task.dataset, task.embedder, int(task.seed), float(task.alpha), task.tau_mode)


def _task_config_key(task: BenchmarkTask) -> tuple[str, str, float, str, str]:
    return (task.dataset, task.embedder, float(task.alpha), task.tau_mode, task.group)


def _failed_selections(
    path: Path,
) -> tuple[set[tuple[str, str, int, float, str]], set[tuple[str, str, float, str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        task_keys: set[tuple[str, str, int, float, str]] = set()
        config_keys: set[tuple[str, str, float, str, str]] = set()
        for row in rows:
            try:
                dataset = row["dataset"]
                embedder = row["embedder"]
                alpha = float(row["alpha"])
                tau_mode = row["tau_mode"]
                group = row.get("group", "main") or "main"
            except (KeyError, TypeError, ValueError):
                continue
            seed_text = str(row.get("seed", "")).strip()
            try:
                seed = int(seed_text)
            except (TypeError, ValueError):
                config_keys.add((dataset, embedder, alpha, tau_mode, group))
                continue
            task_keys.add((dataset, embedder, seed, alpha, tau_mode))
        return task_keys, config_keys


def _result_csv(task: BenchmarkTask) -> Path:
    return task.output_dir / RESULT_CSV


def _is_ready_for_aggregation(task: BenchmarkTask) -> bool:
    return is_complete(task) and _result_csv(task).exists()


def _row(task: BenchmarkTask, command: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": task.task_id, "group": task.group, "dataset": task.dataset,
        "embedder": task.embedder, "seed": task.seed, "alpha": f"{task.alpha:.4g}",
        "tau_mode": task.tau_mode, "n_train": task.n_train, "n_calib": task.n_calib,
        "n_eval": task.n_eval, "hadamard": int(task.hadamard), "n_trials": task.n_trials,
        "extra_cli_args": " ".join(task.extra_cli_args), "output_dir": str(task.output_dir),
        "npz_path": str(task.npz_path),
    }
    if command is not None:
        row["command"] = command
    return row


def _job_group_key(task: BenchmarkTask) -> tuple[Any, ...]:
    return (
        task.dataset,
        task.embedder,
        float(task.alpha),
        task.tau_mode,
        task.group,
        task.n_train,
        task.n_calib,
        task.n_eval,
        bool(task.hadamard),
        task.n_trials,
        task.extra_cli_args,
        str(task.npz_path),
    )


def _group_tasks_into_jobs(tasks: Sequence[BenchmarkTask], exp_root: Path) -> list[BenchmarkJob]:
    grouped: dict[tuple[Any, ...], list[BenchmarkTask]] = {}
    for task in tasks:
        grouped.setdefault(_job_group_key(task), []).append(task)

    jobs: list[BenchmarkJob] = []
    for job_id, group_tasks in enumerate(grouped.values(), start=1):
        ordered = tuple(sorted(group_tasks, key=lambda task: int(task.seed)))
        jobs.append(
            BenchmarkJob(
                job_id=job_id,
                tasks=ordered,
                output_dir=Path(exp_root) / "jobs" / f"job_{job_id:06d}",
            )
        )
    return jobs


def _job_seed_text(job: BenchmarkJob) -> str:
    return ",".join(str(task.seed) for task in job.tasks)


def _job_task_id_text(job: BenchmarkJob) -> str:
    return ",".join(str(task.task_id) for task in job.tasks)


def _job_row(job: BenchmarkJob, command: str | None = None) -> dict[str, Any]:
    first = job.tasks[0]
    row = _row(first, command=command)
    row["task_id"] = job.job_id
    row["seed"] = _job_seed_text(job)
    row["output_dir"] = str(job.output_dir)
    return row


def _write_job_script(job: BenchmarkJob, python_bin: str) -> Path:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    script = job.output_dir / "job.sh"
    first = job.tasks[0]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "echo " + shlex.quote(
            "Running "
            f"job_id={job.job_id} dataset={first.dataset} embedder={first.embedder} "
            f"alpha={first.alpha:.4g} tau_mode={first.tau_mode} seeds={_job_seed_text(job)}"
        ),
        "",
    ]
    for task in job.tasks:
        lines.append("echo " + shlex.quote(f"Starting seed={task.seed} task_id={task.task_id}"))
        lines.append("bash -lc " + shlex.quote(task_to_command_line(python_bin, task)))
        lines.append("echo " + shlex.quote(f"Finished seed={task.seed} task_id={task.task_id}"))
        lines.append("")
    script.write_text("\n".join(lines), encoding="utf-8")
    script.chmod(0o755)
    return script


def _job_command_line(script: Path) -> str:
    return "bash " + shlex.quote(str(script))


def _job_preview(job: BenchmarkJob, python_bin: str, *, limit: int = 2) -> str:
    lines = [
        f"# job_id={job.job_id} seeds={_job_seed_text(job)} seed_count={len(job.tasks)}",
    ]
    for task in job.tasks[:limit]:
        lines.append(task_to_command_line(python_bin, task))
    if len(job.tasks) > limit:
        lines.append(f"# ... {len(job.tasks) - limit} more seed command(s) in this job")
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_files(
    exp_root: Path,
    tasks: list[BenchmarkTask],
    jobs: list[BenchmarkJob],
    python_bin: str,
    args: argparse.Namespace,
    counts: dict[str, int],
) -> tuple[Path, Path, Path]:
    (exp_root / "slurm").mkdir(parents=True, exist_ok=True)
    (exp_root / "task_logs").mkdir(parents=True, exist_ok=True)
    (exp_root / "jobs").mkdir(parents=True, exist_ok=True)
    git_sha = _best_stdout(["git", "rev-parse", "HEAD"], REPO_ROOT)
    py_version = _best_stdout([python_bin, "-c", "import sys; print(sys.version.replace('\\n', ' '))"])
    commands_file, commands_tsv, failed_tsv = exp_root / "commands.txt", exp_root / "commands.tsv", exp_root / "failed.tsv"
    tasks_tsv = exp_root / "tasks.tsv"
    with commands_file.open("w", encoding="utf-8") as cmd_f, commands_tsv.open("w", encoding="utf-8", newline="") as tsv_f:
        writer = csv.DictWriter(tsv_f, fieldnames=[*HEADERS, "command"], delimiter="\t")
        writer.writeheader()
        for job in jobs:
            script = _write_job_script(job, python_bin)
            command = _job_command_line(script)
            cmd_f.write(command + "\n")
            writer.writerow(_job_row(job, command))

            job_config = {
                "job": _job_row(job),
                "task_ids": _job_task_id_text(job),
                "seeds": [int(task.seed) for task in job.tasks],
                "git_sha": git_sha,
                "python_bin": python_bin,
                "python_version": py_version,
                "script": str(script),
            }
            (job.output_dir / "job_config.json").write_text(
                json.dumps(_jsonable(job_config), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    with tasks_tsv.open("w", encoding="utf-8", newline="") as tsv_f:
        writer = csv.DictWriter(tsv_f, fieldnames=[*HEADERS, "command"], delimiter="\t")
        writer.writeheader()
        for task in tasks:
            task.output_dir.mkdir(parents=True, exist_ok=True)
            command = task_to_command_line(python_bin, task)
            writer.writerow(_row(task, command))
            config = {
                "task": _row(task), "git_sha": git_sha, "python_bin": python_bin,
                "python_version": py_version, "cli_args": task_to_cli_args(task),
            }
            (task.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed_tsv.write_text("\t".join([*HEADERS, "command", "exit_code"]) + "\n", encoding="utf-8")
    manifest = {
        "launcher_args": _jsonable(vars(args)), "git_sha": git_sha,
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_remaining_seed_tasks": len(tasks),
        "total_remaining_jobs": len(jobs),
        "tasks_tsv": str(tasks_tsv),
        **counts,
    }
    (exp_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return commands_file, commands_tsv, failed_tsv


def _print_job_table(jobs: list[BenchmarkJob]) -> None:
    seed_tasks = sum(len(job.tasks) for job in jobs)
    print(f"Remaining seed tasks: {seed_tasks}")
    print(f"Remaining array jobs: {len(jobs)}")
    print("job_id\tdataset\tembedder\tseed_count\tseeds\talpha\ttau_mode\tjob_dir")
    for job in jobs:
        first = job.tasks[0]
        print(
            f"{job.job_id}\t{first.dataset}\t{first.embedder}\t{len(job.tasks)}\t"
            f"{_job_seed_text(job)}\t{first.alpha:.4g}\t{first.tau_mode}\t{job.output_dir}"
        )


def _task_label(task: BenchmarkTask) -> str:
    return (
        f"task_id={task.task_id} dataset={task.dataset} embedder={task.embedder} "
        f"seed={task.seed} alpha={task.alpha:.4g} tau_mode={task.tau_mode}"
    )


def _limited_task_list(tasks: Sequence[BenchmarkTask], limit: int = 12) -> str:
    shown = "\n  - ".join(_task_label(task) for task in tasks[:limit])
    if len(tasks) > limit:
        shown += f"\n  - ... {len(tasks) - limit} more"
    return shown


def _read_failed_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    def has_value(row: dict[str, Any]) -> bool:
        for value in row.values():
            if isinstance(value, list):
                if any(str(item).strip() for item in value):
                    return True
            elif str(value or "").strip():
                return True
        return False

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in rows if has_value(row)]


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return int(number)


def _t_multiplier_95(n: int) -> float:
    if n <= 1:
        return 0.0
    df = n - 1
    if df in T_CRITICAL_975:
        return T_CRITICAL_975[df]
    return 1.96


def _mean_ci95(values: Iterable[Any]) -> dict[str, Any]:
    xs = [value for value in (_float_or_none(value) for value in values) if value is not None]
    n = len(xs)
    if n == 0:
        return {
            "n": 0,
            "mean": "",
            "std": "",
            "ci95_low": "",
            "ci95_high": "",
            "ci95_half_width": "",
            "min": "",
            "max": "",
        }
    mean = statistics.fmean(xs)
    std = statistics.stdev(xs) if n > 1 else 0.0
    half_width = _t_multiplier_95(n) * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": int(n),
        "mean": float(mean),
        "std": float(std),
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
        "ci95_half_width": float(half_width),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def _aggregate_fieldnames() -> list[str]:
    fields = [
        "dataset", "dataset_name", "embedder", "embedder_name", "group",
        "alpha", "tau_mode", "rank", "method", "n", "seed_count",
        "trial_count", "valid_rate",
    ]
    for metric in AGGREGATE_METRICS:
        fields.extend(
            [
                f"{metric}_n",
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_ci95_low",
                f"{metric}_ci95_high",
                f"{metric}_ci95_half_width",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )
    return fields


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _collect_trial_rows(
    tasks: Sequence[BenchmarkTask],
) -> tuple[list[dict[str, Any]], list[BenchmarkTask], list[BenchmarkTask]]:
    rows: list[dict[str, Any]] = []
    incomplete: list[BenchmarkTask] = []
    missing_summary: list[BenchmarkTask] = []

    for task in tasks:
        if not is_complete(task):
            incomplete.append(task)
        path = _result_csv(task)
        if not path.exists():
            missing_summary.append(task)
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("method"):
                    continue
                enriched = dict(row)
                enriched.update(
                    {
                        "_dataset": task.dataset,
                        "_embedder": task.embedder,
                        "_group": task.group,
                        "_alpha": float(task.alpha),
                        "_tau_mode": task.tau_mode,
                        "_task_seed": int(task.seed),
                        "_task_id": int(task.task_id),
                        "_source": str(path),
                    }
                )
                rows.append(enriched)
    return rows, incomplete, missing_summary


def _dataset_name(dataset: str) -> str:
    meta = registry.DATASETS.get(dataset)
    return meta.display_name if meta is not None else dataset


def _embedder_name(embedder: str) -> str:
    meta = registry.EMBEDDERS.get(embedder)
    return meta.display_name if meta is not None else embedder


def _aggregate_trial_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method", ""))
        if not method:
            continue
        key = (
            str(row["_dataset"]),
            str(row["_embedder"]),
            str(row["_group"]),
            float(row["_alpha"]),
            str(row["_tau_mode"]),
            method,
        )
        grouped[key].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (dataset, embedder, group, alpha, tau_mode, method), items in grouped.items():
        seed_values = {
            seed
            for seed in (_int_or_none(row.get("seed", row.get("_task_seed"))) for row in items)
            if seed is not None
        }
        trial_values = {
            (row.get("seed", row.get("_task_seed")), row.get("trial", "0"))
            for row in items
        }
        fprs = [value for value in (_float_or_none(row.get("micro_fpr")) for row in items) if value is not None]
        valid_rate = (
            float(sum(1 for value in fprs if value <= alpha + 1e-12) / len(fprs))
            if fprs
            else ""
        )
        out: dict[str, Any] = {
            "dataset": dataset,
            "dataset_name": _dataset_name(dataset),
            "embedder": embedder,
            "embedder_name": _embedder_name(embedder),
            "group": group,
            "alpha": float(alpha),
            "tau_mode": tau_mode,
            "rank": "",
            "method": method,
            "n": int(len(items)),
            "seed_count": int(len(seed_values)),
            "trial_count": int(len(trial_values)),
            "valid_rate": valid_rate,
        }
        for metric in AGGREGATE_METRICS:
            stats = _mean_ci95(row.get(metric) for row in items)
            for name, value in stats.items():
                out[f"{metric}_{name}"] = value
        aggregate_rows.append(out)

    by_setting: dict[tuple[str, str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_setting[
            (
                str(row["dataset"]),
                str(row["embedder"]),
                str(row["group"]),
                float(row["alpha"]),
                str(row["tau_mode"]),
            )
        ].append(row)

    def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
        alpha = float(row["alpha"])
        valid_rate = _float_or_none(row.get("valid_rate")) or 0.0
        mean_fpr = _float_or_none(row.get("micro_fpr_mean"))
        max_fpr = _float_or_none(row.get("micro_fpr_max"))
        mean_tpr = _float_or_none(row.get("micro_tpr_mean"))
        train_n = _float_or_none(row.get("train_samples_needed_mean"))
        return (
            0 if valid_rate >= 1.0 - 1e-12 else 1,
            0 if mean_fpr is not None and mean_fpr <= alpha + 1e-12 else 1,
            0 if max_fpr is not None and max_fpr <= alpha + 1e-12 else 1,
            -(mean_tpr if mean_tpr is not None else float("-inf")),
            train_n if train_n is not None else float("inf"),
            -valid_rate,
            str(row.get("method", "")),
        )

    for setting_rows in by_setting.values():
        setting_rows.sort(key=rank_key)
        for rank, row in enumerate(setting_rows, start=1):
            row["rank"] = int(rank)

    aggregate_rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["embedder"]),
            float(row["alpha"]),
            str(row["tau_mode"]),
            int(row["rank"]),
            str(row["method"]),
        )
    )
    return aggregate_rows


def _format_float(value: Any, digits: int = 4) -> str:
    number = _float_or_none(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _format_ci(row: dict[str, Any], metric: str, digits: int = 4) -> str:
    mean = _float_or_none(row.get(f"{metric}_mean"))
    half = _float_or_none(row.get(f"{metric}_ci95_half_width"))
    if mean is None or half is None:
        return ""
    return f"{mean:.{digits}f} +/- {half:.{digits}f}"


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _write_markdown_table(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    headers = [
        "embedder", "alpha", "rank", "method", "n", "valid_rate",
        "micro_tpr (95% CI)", "micro_fpr (95% CI)",
        "macro_tpr (95% CI)", "macro_fpr (95% CI)", "train_n",
    ]
    lines = [
        f"# {_md_escape(rows[0]['dataset_name'] if rows else path.stem)}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row.get("embedder", ""),
            _format_float(row.get("alpha"), digits=4),
            row.get("rank", ""),
            _md_escape(row.get("method", "")),
            row.get("n", ""),
            _format_float(row.get("valid_rate"), digits=3),
            _format_ci(row, "micro_tpr"),
            _format_ci(row, "micro_fpr"),
            _format_ci(row, "macro_tpr"),
            _format_ci(row, "macro_fpr"),
            _format_float(row.get("train_samples_needed_mean"), digits=1),
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_aggregate_tables(
    exp_root: Path,
    tasks: Sequence[BenchmarkTask],
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    rows, incomplete, missing_summary = _collect_trial_rows(tasks)
    if incomplete and not allow_partial:
        raise RuntimeError(
            "Some tasks did not finish:\n  - " + _limited_task_list(incomplete)
        )
    if missing_summary and not allow_partial:
        raise RuntimeError(
            f"Some completed task directories are missing {RESULT_CSV}:\n  - "
            + _limited_task_list(missing_summary)
        )
    if not rows:
        raise RuntimeError(f"No {RESULT_CSV} rows were found to aggregate.")

    aggregate_rows = _aggregate_trial_rows(rows)
    fieldnames = _aggregate_fieldnames()
    table_dir = exp_root / "final_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    all_csv = table_dir / "all_datasets.csv"
    _write_csv(all_csv, aggregate_rows, fieldnames)
    written = [all_csv]

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_dataset[str(row["dataset"])].append(row)

    for dataset, dataset_rows in sorted(by_dataset.items()):
        csv_path = table_dir / f"{dataset}.csv"
        md_path = table_dir / f"{dataset}.md"
        _write_csv(csv_path, dataset_rows, fieldnames)
        _write_markdown_table(md_path, dataset_rows)
        written.extend([csv_path, md_path])

    metadata = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "ci_method": "two-sided 95% Student t interval over seed/trial rows; normal 1.96 approximation for df > 30",
        "result_csv": RESULT_CSV,
        "tasks_considered": len(tasks),
        "trial_rows": len(rows),
        "aggregate_rows": len(aggregate_rows),
        "incomplete_tasks": [_row(task) for task in incomplete],
        "missing_summary_tasks": [_row(task) for task in missing_summary],
        "tables": [str(path) for path in written],
    }
    metadata_path = table_dir / "aggregation_metadata.json"
    metadata_path.write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(metadata_path)
    return {
        "paths": written,
        "trial_rows": len(rows),
        "aggregate_rows": len(aggregate_rows),
        "incomplete": incomplete,
        "missing_summary": missing_summary,
    }


def _env_setup(conda_env: str) -> str:
    env = shlex.quote(conda_env)
    return f'set -euo pipefail; CONDA_BASE="$(conda info --base)"; source "${{CONDA_BASE}}/etc/profile.d/conda.sh"; conda activate {env}'


def _sbatch_command(exp_root: Path, commands_file: Path, commands_tsv: Path, failed_tsv: Path, n_tasks: int, args: argparse.Namespace) -> list[str]:
    array_spec = f"1-{n_tasks}"
    if int(args.max_parallel) > 0:
        array_spec += f"%{int(args.max_parallel)}"
    cmd = [
        "sbatch", f"--job-name={args.job_name}", f"--array={array_spec}",
        f"--time={args.time}", f"--cpus-per-task={args.cpus_per_task}", f"--mem={args.mem}",
        f"--output={exp_root}/slurm/%x_%A_%a.out", f"--error={exp_root}/slurm/%x_%A_%a.err",
        "--export="
        f"ALL,COMMANDS_FILE={commands_file},COMMANDS_TSV={commands_tsv},FAILED_TSV={failed_tsv},"
        f"EXP_ROOT={exp_root},ENV_SETUP={_env_setup(args.conda_env)}",
    ]
    for attr, opt in (("partition", "partition"), ("account", "account"), ("qos", "qos")):
        value = getattr(args, attr)
        if value:
            cmd.append(f"--{opt}={value}")
    if args.gpus != "0":
        cmd.append(f"--gres=gpu:{args.gpus}")
    cmd.append(str(SBATCH_FILE))
    return cmd


def _parse_sbatch_job_id(output: str) -> str | None:
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    return match.group(1) if match else None


def _submit_sbatch(sbatch: Sequence[str]) -> str:
    proc = subprocess.run(sbatch, text=True, capture_output=True, check=True)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)
    job_id = _parse_sbatch_job_id(proc.stdout)
    if job_id is None:
        raise RuntimeError(f"Could not parse SLURM job id from sbatch output: {proc.stdout.strip()!r}")
    return job_id


def _wait_for_slurm_job(job_id: str, *, poll_interval: float, timeout: float) -> None:
    poll = max(float(poll_interval), 1.0)
    started = time.monotonic()
    print(f"Waiting for SLURM job {job_id} to finish...", flush=True)
    while True:
        proc = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%i %T"],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"squeue failed while waiting for job {job_id}: {proc.stderr.strip()}")
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            print(f"SLURM job {job_id} is no longer in the queue.", flush=True)
            return
        states = sorted({line.split(maxsplit=1)[1] if " " in line else "UNKNOWN" for line in lines})
        elapsed = time.monotonic() - started
        print(
            f"[wait] job={job_id} elapsed={elapsed / 60.0:.1f}m "
            f"queue_entries={len(lines)} states={','.join(states)}",
            flush=True,
        )
        if timeout > 0 and elapsed >= timeout:
            raise RuntimeError(f"Timed out waiting for SLURM job {job_id} after {timeout:.0f}s.")
        time.sleep(poll)


def _finalize_results(
    exp_root: Path,
    tasks: Sequence[BenchmarkTask],
    failed_tsv: Path,
    args: argparse.Namespace,
) -> int:
    if args.skip_aggregate:
        return 0
    if not tasks:
        print("No tasks selected for aggregation.")
        return 0

    failed_rows = _read_failed_rows(failed_tsv)
    if failed_rows and not args.allow_partial_aggregate:
        print(
            f"ERROR: {len(failed_rows)} task(s) failed; not aggregating. "
            f"See {failed_tsv} or rerun with --allow-partial-aggregate.",
            file=sys.stderr,
        )
        return 1

    try:
        result = _write_aggregate_tables(
            exp_root,
            tasks,
            allow_partial=bool(args.allow_partial_aggregate),
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Aggregated {result['trial_rows']} trial rows into "
        f"{result['aggregate_rows']} method rows."
    )
    print("Final tables:")
    for path in result["paths"]:
        print(f"  {path}")
    return 0


def _build_streaming_cli(python_bin: str, args: argparse.Namespace, output_dir: Path) -> str:
    parts: list[str] = [
        python_bin, "-m", "experiments.compare_vcache_real_datasets",
        "--datasets", *args.streaming_datasets,
        "--methods", *args.streaming_methods,
        "--embedding_model", "GTE",
        "--seed", str(args.streaming_seed),
        "--cache_size", str(args.streaming_cache_size),
        "--eviction_policy", "mru",
        "--delta_values", *[str(v) for v in args.streaming_delta_values],
        "--thresholds", *[str(v) for v in args.streaming_thresholds],
        "--target_budgets", *[str(v) for v in args.streaming_target_budgets],
        "--hard_neighbor_thresholds", *[str(v) for v in args.streaming_hard_thresholds],
        "--allow_pairwise_from_stream_annotations",
        "--vcache_repo_path", str(args.vcache_repo_path),
        "--raw_log_mode", args.streaming_raw_log_mode,
        "--output_dir", str(output_dir),
    ]
    if args.streaming_limit is not None:
        parts += ["--limit", str(args.streaming_limit)]
    return " ".join(shlex.quote(str(p)) for p in parts)


def _write_streaming_script(exp_root: Path, python_bin: str, args: argparse.Namespace) -> Path:
    script_dir = exp_root / "streaming_job"
    script_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_streaming_cli(python_bin, args, exp_root / "streaming")
    script = script_dir / "streaming.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "echo 'Starting vcache streaming evaluation'",
        "",
        cmd,
        "",
        "echo 'Streaming evaluation complete'",
    ]
    script.write_text("\n".join(lines), encoding="utf-8")
    script.chmod(0o755)
    return script


def _sbatch_streaming_command(exp_root: Path, script: Path, args: argparse.Namespace) -> list[str]:
    time_limit = getattr(args, "streaming_time", None) or args.time
    cmd = [
        "sbatch",
        f"--job-name={args.job_name}_stream",
        f"--time={time_limit}",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem={args.mem}",
        f"--output={exp_root}/slurm/%x_%j.out",
        f"--error={exp_root}/slurm/%x_%j.err",
    ]
    for attr, opt in (("partition", "partition"), ("account", "account"), ("qos", "qos")):
        value = getattr(args, attr, "")
        if value:
            cmd.append(f"--{opt}={value}")
    if getattr(args, "gpus", "0") != "0":
        cmd.append(f"--gres=gpu:{args.gpus}")
    cmd.append(str(script))
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_suite(args)
    _apply_dataset_presets(args)
    _apply_competitor_presets(args)
    if args.streaming is None:
        args.streaming = False
    python_bin = _resolve_python(args.conda_env, args.python_bin)
    exp_root = _abs_path(args.exp_root)
    if not args.dry_run:
        _download_inputs(args.datasets, args.embedders, python_bin)
    all_tasks = build_main_suite(
        exp_root, args.datasets, args.embedders, args.seeds, args.alphas,
        args.n_trials, args.n_train, args.n_calib, args.n_eval, REPO_ROOT,
        tau_mode=args.tau_mode, hadamard=args.hadamard, extra_cli_args=args.extra_cli_arg,
        validate_inputs=not args.dry_run,
    )
    before = len(all_tasks)
    filtered = _filter_tasks(all_tasks, args.filter)
    if args.only_failed is not None:
        wanted_tasks, wanted_configs = _failed_selections(args.only_failed)
        filtered = [
            task
            for task in filtered
            if _task_key(task) in wanted_tasks or _task_config_key(task) in wanted_configs
        ]
    after = len(filtered)
    remaining = filtered if args.force else [task for task in filtered if not _is_ready_for_aggregation(task)]
    jobs = _group_tasks_into_jobs(remaining, exp_root)
    counts = {
        "total_tasks_before_filter": before,
        "total_tasks_after_filter": after,
        "total_skipped_complete": after - len(remaining),
        "total_jobs_after_grouping": len(jobs),
    }
    if args.dry_run:
        _print_job_table(jobs)
        if jobs:
            print("\nSample command:")
            print(_job_preview(jobs[0], python_bin))
        if args.streaming:
            streaming_cmd = _build_streaming_cli(python_bin, args, exp_root / "streaming")
            print("\nStreaming evaluation command:")
            print(streaming_cmd)
        return 0
    commands_file, commands_tsv, failed_tsv = _write_files(exp_root, remaining, jobs, python_bin, args, counts)
    streaming_script: Path | None = None
    streaming_sbatch: list[str] = []
    if args.streaming:
        streaming_script = _write_streaming_script(exp_root, python_bin, args)
        streaming_sbatch = _sbatch_streaming_command(exp_root, streaming_script, args)
        print("Streaming sbatch command:")
        print(" ".join(shlex.quote(part) for part in streaming_sbatch))
    if not remaining:
        print("Nothing to do.")
        if args.streaming and args.submit and streaming_script is not None:
            streaming_job_id = _submit_sbatch(streaming_sbatch)
            print(f"Submitted streaming SLURM job {streaming_job_id}.")
        return _finalize_results(exp_root, filtered, failed_tsv, args)
    sbatch = _sbatch_command(exp_root, commands_file, commands_tsv, failed_tsv, len(jobs), args)
    print("Sbatch command:")
    print(" ".join(shlex.quote(part) for part in sbatch))
    if args.submit:
        job_id = _submit_sbatch(sbatch)
        streaming_job_id: str | None = None
        if args.streaming and streaming_script is not None:
            streaming_job_id = _submit_sbatch(streaming_sbatch)
            print(f"Submitted streaming SLURM job {streaming_job_id}.")
        if args.wait:
            _wait_for_slurm_job(job_id, poll_interval=args.poll_interval, timeout=args.wait_timeout)
            return _finalize_results(exp_root, filtered, failed_tsv, args)
        msg = (
            f"Submitted SLURM job {job_id}. Re-run with --exp-root {shlex.quote(str(exp_root))} "
            "after completion to aggregate, or omit --no-wait next time."
        )
        if streaming_job_id:
            msg += f" Streaming job: {streaming_job_id}."
        print(msg)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
