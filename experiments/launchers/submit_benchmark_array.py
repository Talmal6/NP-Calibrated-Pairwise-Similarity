"""Submit NeighborCache benchmark tasks as one SLURM array."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

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
    add("--embedders", nargs="+", default=["default"], help="Embedder registry keys.")
    add("--seeds", nargs="+", type=int, default=list(range(20)), help="Seed values.")
    add("--alphas", nargs="+", type=float, default=[0.01, 0.03, 0.05, 0.10], help="Alpha values.")
    add("--tau-mode", default="global", help="Threshold mode passed to the benchmark CLI.")
    add("--n-train", type=int, default=1200, help="Number of training examples.")
    add("--n-calib", type=int, default=1200, help="Number of calibration examples.")
    add("--n-eval", type=int, default=1200, help="Number of evaluation examples.")
    add("--n-trials", type=int, default=1, help="Repetitions per seed.")
    add("--hadamard", dest="hadamard", action="store_true", default=True, help="Enable Hadamard preprocessing.")
    add("--no-hadamard", dest="hadamard", action="store_false", help="Disable Hadamard preprocessing.")
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
    add("--max-parallel", type=int, default=8, help="Maximum simultaneous array tasks.")
    add("--job-name", default="ncache_bench", help="SLURM job name.")
    add("--dry-run", action="store_true", help="Print the matrix and sample command without writing files.")
    add("--submit", action="store_true", help="Submit with sbatch after writing files.")
    add("--force", action="store_true", help="Re-run tasks even when _COMPLETE exists.")
    add("--only-failed", type=Path, default=None, help="failed.tsv path used to retry matching rows.")
    add("--filter", default="", help="Comma-separated key=value task filter.")
    add("--suite", choices=["smoke", "paper"], default="paper", help="Benchmark suite preset.")
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


def _failed_keys(path: Path) -> set[tuple[str, str, int, float, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        out = set()
        for row in rows:
            try:
                out.add((row["dataset"], row["embedder"], int(row["seed"]), float(row["alpha"]), row["tau_mode"]))
            except (KeyError, TypeError, ValueError):
                continue
        return out


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
    python_bin: str,
    args: argparse.Namespace,
    counts: dict[str, int],
) -> tuple[Path, Path, Path]:
    (exp_root / "slurm").mkdir(parents=True, exist_ok=True)
    (exp_root / "task_logs").mkdir(parents=True, exist_ok=True)
    git_sha = _best_stdout(["git", "rev-parse", "HEAD"], REPO_ROOT)
    py_version = _best_stdout([python_bin, "-c", "import sys; print(sys.version.replace('\\n', ' '))"])
    commands_file, commands_tsv, failed_tsv = exp_root / "commands.txt", exp_root / "commands.tsv", exp_root / "failed.tsv"
    with commands_file.open("w", encoding="utf-8") as cmd_f, commands_tsv.open("w", encoding="utf-8", newline="") as tsv_f:
        writer = csv.DictWriter(tsv_f, fieldnames=[*HEADERS, "command"], delimiter="\t")
        writer.writeheader()
        for task in tasks:
            task.output_dir.mkdir(parents=True, exist_ok=True)
            command = task_to_command_line(python_bin, task)
            cmd_f.write(command + "\n")
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
        "total_remaining": len(tasks), **counts,
    }
    (exp_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return commands_file, commands_tsv, failed_tsv


def _print_table(tasks: list[BenchmarkTask]) -> None:
    print(f"Remaining tasks: {len(tasks)}")
    print("task_id\tdataset\tembedder\tseed\talpha\ttau_mode\toutput_dir")
    for task in tasks:
        print(f"{task.task_id}\t{task.dataset}\t{task.embedder}\t{task.seed}\t{task.alpha:.4g}\t{task.tau_mode}\t{task.output_dir}")


def _env_setup(conda_env: str) -> str:
    env = shlex.quote(conda_env)
    return f'set -euo pipefail; CONDA_BASE="$(conda info --base)"; source "${{CONDA_BASE}}/etc/profile.d/conda.sh"; conda activate {env}'


def _sbatch_command(exp_root: Path, commands_file: Path, commands_tsv: Path, failed_tsv: Path, n_tasks: int, args: argparse.Namespace) -> list[str]:
    cmd = [
        "sbatch", f"--job-name={args.job_name}", f"--array=1-{n_tasks}%{args.max_parallel}",
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_suite(args)
    python_bin = _resolve_python(args.conda_env, args.python_bin)
    exp_root = _abs_path(args.exp_root)
    _download_inputs(args.datasets, args.embedders, python_bin)
    all_tasks = build_main_suite(
        exp_root, args.datasets, args.embedders, args.seeds, args.alphas,
        args.n_trials, args.n_train, args.n_calib, args.n_eval, REPO_ROOT,
        tau_mode=args.tau_mode, hadamard=args.hadamard, extra_cli_args=args.extra_cli_arg,
    )
    before = len(all_tasks)
    filtered = _filter_tasks(all_tasks, args.filter)
    if args.only_failed is not None:
        wanted = _failed_keys(args.only_failed)
        filtered = [task for task in filtered if _task_key(task) in wanted]
    after = len(filtered)
    remaining = filtered if args.force else [task for task in filtered if not is_complete(task)]
    counts = {
        "total_tasks_before_filter": before,
        "total_tasks_after_filter": after,
        "total_skipped_complete": after - len(remaining),
    }
    if args.dry_run:
        _print_table(remaining)
        if remaining:
            print("\nSample command:")
            print(task_to_command_line(python_bin, remaining[0]))
        return 0
    commands_file, commands_tsv, failed_tsv = _write_files(exp_root, remaining, python_bin, args, counts)
    if not remaining:
        print("Nothing to do.")
        return 0
    sbatch = _sbatch_command(exp_root, commands_file, commands_tsv, failed_tsv, len(remaining), args)
    print("Sbatch command:")
    print(" ".join(shlex.quote(part) for part in sbatch))
    if args.submit:
        subprocess.check_call(sbatch)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
