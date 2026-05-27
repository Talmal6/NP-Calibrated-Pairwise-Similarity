#!/usr/bin/env python3
"""
run_paper_slurm.py

Single-file Slurm worker for the normal NeighborCache paper run.

Use with:
  sbatch submit_paper_normal_4090_ec.sbatch

The sbatch file is a Slurm array. Each array task calls this Python file with
--run-task. The Python file reconstructs the same deterministic task list and
runs the task matching SLURM_ARRAY_TASK_ID.

Outputs are written under:
  NeighborCache/results/paper_normal_<slurm-array-job-id>/

Main outputs:
  commands.tsv
  task_status.csv
  failed_commands.tsv
  run_metadata.txt
  task_logs/
  merged_*.csv            if result CSVs are found under the experiment root
  paper_results_flat.csv  flat union of discovered result CSVs

Default paper suite:
  2 datasets
  4 alpha sweep jobs per dataset
  4 train-size ablation jobs per dataset
  4 calib-size ablation jobs per dataset
  = 24 tasks total

This file intentionally does not include the no-whitened ensemble variant.
We will use a second configuration later for that condition.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# =============================================================================
# Experiment defaults
# =============================================================================

DATASETS = {
    "h1h0": "NeighborCache/data/h1h0_final.npz",
    "lmsys": "NeighborCache/data/lmsys_h1h0_by_cluster_h1keep_with_unormalized_embedding_fixed.npz",
}

CLI_MODULE_DEFAULT = "NeighborCache.region_local_threshold.cli"
REGION_KEY_DEFAULT = "global_cluster"

FULL_N_TRAIN = 1200
FULL_N_CALIB = 1200
FULL_N_EVAL = 1200

ALPHAS = [0.01, 0.03, 0.05, 0.10]
TRAIN_SIZES = [100, 300, 700, 1200]
CALIB_SIZES = [100, 300, 700, 1200]

DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 42
DEFAULT_TAU_MODE = "global"

SMOKE_TRIALS = 3
MAIN_TRIALS = 20
ABLATION_TRIALS = 20

BOOKKEEPING_NAMES = {
    "commands.tsv",
    "failed_commands.tsv",
    "task_status.csv",
    "task_status_export.csv",
    "run_metadata.txt",
}


@dataclass(frozen=True)
class Task:
    task_id: int
    group: str
    name: str
    dataset_id: str
    dataset_path: str
    alpha: float
    tau_mode: str
    seed: int
    n_train: int
    n_calib: int
    n_eval: int
    hadamard: bool
    n_trials: int

    def command(self, python_bin: str, cli_module: str, region_key: str, repo_root: Path) -> list[str]:
        data_path = Path(self.dataset_path)
        if not data_path.is_absolute():
            data_path = repo_root / data_path

        cmd = [
            python_bin,
            "-m",
            cli_module,
            "--data",
            str(data_path),
            "--region_key",
            region_key,
            "--alpha",
            f"{self.alpha:.4g}",
            "--tau_mode",
            self.tau_mode,
            "--n_trials",
            str(self.n_trials),
            "--seed",
            str(self.seed),
            "--n_train",
            str(self.n_train),
            "--n_calib",
            str(self.n_calib),
            "--n_eval",
            str(self.n_eval),
        ]

        if self.hadamard:
            cmd.append("--hadamard_preprocess")

        return cmd


def shjoin(items: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in items)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def get_repo_root() -> Path:
    env_root = os.environ.get("REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path.cwd().resolve()


def default_exp_root(repo_root: Path) -> Path:
    explicit = os.environ.get("EXP_ROOT")
    if explicit:
        return Path(explicit).resolve()

    job_id = (
        os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    return (repo_root / "NeighborCache" / "results" / f"paper_normal_{job_id}").resolve()


def build_tasks(suite: str, datasets: list[str] | None, main_trials: int, ablation_trials: int) -> list[Task]:
    selected = DATASETS if not datasets else {k: DATASETS[k] for k in datasets}
    tasks: list[Task] = []

    def add(
        *,
        group: str,
        name: str,
        dataset_id: str,
        dataset_path: str,
        alpha: float = DEFAULT_ALPHA,
        tau_mode: str = DEFAULT_TAU_MODE,
        seed: int = DEFAULT_SEED,
        n_train: int = FULL_N_TRAIN,
        n_calib: int = FULL_N_CALIB,
        n_eval: int = FULL_N_EVAL,
        hadamard: bool = True,
        n_trials: int = MAIN_TRIALS,
    ) -> None:
        tasks.append(
            Task(
                task_id=len(tasks) + 1,
                group=group,
                name=name,
                dataset_id=dataset_id,
                dataset_path=dataset_path,
                alpha=alpha,
                tau_mode=tau_mode,
                seed=seed,
                n_train=n_train,
                n_calib=n_calib,
                n_eval=n_eval,
                hadamard=hadamard,
                n_trials=n_trials,
            )
        )

    include_smoke = suite in {"smoke", "all"}
    include_paper = suite in {"paper", "all"}

    if include_smoke:
        for dataset_id, dataset_path in selected.items():
            add(
                group="smoke",
                name=f"{dataset_id}_smoke",
                dataset_id=dataset_id,
                dataset_path=dataset_path,
                alpha=0.05,
                n_train=300,
                n_calib=300,
                n_eval=300,
                n_trials=SMOKE_TRIALS,
            )

    if include_paper:
        for dataset_id, dataset_path in selected.items():
            for alpha in ALPHAS:
                add(
                    group="main_alpha_sweep",
                    name=f"{dataset_id}_alpha_{alpha:.2f}",
                    dataset_id=dataset_id,
                    dataset_path=dataset_path,
                    alpha=alpha,
                    n_train=FULL_N_TRAIN,
                    n_calib=FULL_N_CALIB,
                    n_eval=FULL_N_EVAL,
                    n_trials=main_trials,
                )

        for dataset_id, dataset_path in selected.items():
            for n_train in TRAIN_SIZES:
                add(
                    group="train_size_ablation",
                    name=f"{dataset_id}_train_{n_train}",
                    dataset_id=dataset_id,
                    dataset_path=dataset_path,
                    alpha=0.05,
                    n_train=n_train,
                    n_calib=FULL_N_CALIB,
                    n_eval=FULL_N_EVAL,
                    n_trials=ablation_trials,
                )

        for dataset_id, dataset_path in selected.items():
            for n_calib in CALIB_SIZES:
                add(
                    group="calib_size_ablation",
                    name=f"{dataset_id}_calib_{n_calib}",
                    dataset_id=dataset_id,
                    dataset_path=dataset_path,
                    alpha=0.05,
                    n_train=FULL_N_TRAIN,
                    n_calib=n_calib,
                    n_eval=FULL_N_EVAL,
                    n_trials=ablation_trials,
                )

    return tasks


def task_as_row(task: Task, command: str) -> dict[str, str]:
    return {
        "task_id": str(task.task_id),
        "group": task.group,
        "name": task.name,
        "dataset_id": task.dataset_id,
        "dataset_path": task.dataset_path,
        "alpha": f"{task.alpha:.4g}",
        "tau_mode": task.tau_mode,
        "seed": str(task.seed),
        "n_train": str(task.n_train),
        "n_calib": str(task.n_calib),
        "n_eval": str(task.n_eval),
        "hadamard": "1" if task.hadamard else "0",
        "n_trials": str(task.n_trials),
        "command": command,
    }


COMMAND_FIELDS = [
    "task_id",
    "group",
    "name",
    "dataset_id",
    "dataset_path",
    "alpha",
    "tau_mode",
    "seed",
    "n_train",
    "n_calib",
    "n_eval",
    "hadamard",
    "n_trials",
    "command",
]

STATUS_FIELDS = COMMAND_FIELDS + [
    "exit_code",
    "slurm_job_id",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "host",
    "started_at",
    "finished_at",
    "duration_sec",
]

FAILED_FIELDS = COMMAND_FIELDS + ["exit_code"]


def write_dict_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]], *, delimiter: str = "\t") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_dict_row_locked(path: Path, lock_path: Path, fieldnames: list[str], row: dict[str, str], *, delimiter: str = "\t") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def ensure_plan_files(exp_root: Path, tasks: list[Task], python_bin: str, cli_module: str, region_key: str, repo_root: Path, args: argparse.Namespace) -> None:
    exp_root.mkdir(parents=True, exist_ok=True)
    (exp_root / "task_logs").mkdir(parents=True, exist_ok=True)
    (exp_root / "locks").mkdir(parents=True, exist_ok=True)

    lock_path = exp_root / "locks" / "plan.lock"

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        commands_tsv = exp_root / "commands.tsv"
        commands_txt = exp_root / "commands.txt"
        metadata = exp_root / "run_metadata.txt"

        if not commands_tsv.exists():
            rows = []
            lines = []
            for task in tasks:
                command = shjoin(task.command(python_bin, cli_module, region_key, repo_root))
                rows.append(task_as_row(task, command))
                lines.append(command)

            write_dict_rows(commands_tsv, COMMAND_FIELDS, rows, delimiter="\t")
            commands_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if not metadata.exists():
            metadata.write_text(
                "\n".join(
                    [
                        f"experiment_root={exp_root}",
                        f"date={now_iso()}",
                        f"repo_root={repo_root}",
                        f"python_bin={python_bin}",
                        f"cli_module={cli_module}",
                        f"region_key={region_key}",
                        f"suite={args.suite}",
                        f"datasets={args.datasets or 'all'}",
                        f"n_tasks={len(tasks)}",
                        f"full_n_train={FULL_N_TRAIN}",
                        f"full_n_calib={FULL_N_CALIB}",
                        f"full_n_eval={FULL_N_EVAL}",
                        f"alphas={ALPHAS}",
                        f"train_sizes={TRAIN_SIZES}",
                        f"calib_sizes={CALIB_SIZES}",
                        f"main_trials={args.main_trials}",
                        f"ablation_trials={args.ablation_trials}",
                        "note=normal run; no without-whitened ensemble variant",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

        fcntl.flock(lock_file, fcntl.LOCK_UN)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return [], []
        return list(reader.fieldnames), [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in reader]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not path.exists():
            path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def candidate_result_csvs(exp_root: Path) -> list[Path]:
    files = []
    for path in exp_root.rglob("*.csv"):
        if path.name in BOOKKEEPING_NAMES:
            continue
        if path.name.startswith("merged_") or path.name == "paper_results_flat.csv":
            continue
        files.append(path)
    return sorted(files)


def collect_csv_outputs(exp_root: Path) -> None:
    files = candidate_result_csvs(exp_root)

    # Always export task_status as a comma CSV too.
    status = exp_root / "task_status.csv"
    if status.exists():
        _, rows = read_csv_rows(status)
        write_csv(exp_root / "task_status_export.csv", rows)

    if not files:
        return

    grouped: dict[str, list[Path]] = {}
    for path in files:
        grouped.setdefault(path.name, []).append(path)

    flat_rows: list[dict[str, str]] = []

    for basename, paths in grouped.items():
        merged: list[dict[str, str]] = []
        for path in paths:
            _, rows = read_csv_rows(path)
            rel = str(path.relative_to(exp_root))
            for row in rows:
                row = dict(row)
                row["source_file"] = rel
                row["source_basename"] = basename
                merged.append(row)
                flat_rows.append(dict(row))

        write_csv(exp_root / f"merged_{basename}", merged)

    write_csv(exp_root / "paper_results_flat.csv", flat_rows)


def run_task(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    exp_root = default_exp_root(repo_root)
    python_bin = sys.executable

    tasks = build_tasks(args.suite, args.datasets, args.main_trials, args.ablation_trials)

    ensure_plan_files(
        exp_root=exp_root,
        tasks=tasks,
        python_bin=python_bin,
        cli_module=args.cli_module,
        region_key=args.region_key,
        repo_root=repo_root,
        args=args,
    )

    task_id_s = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_s is None:
        if args.task_id is None:
            raise RuntimeError("No SLURM_ARRAY_TASK_ID and no --task-id supplied")
        task_id = args.task_id
    else:
        task_id = int(task_id_s)

    if task_id < 1 or task_id > len(tasks):
        print(f"No-op: task_id={task_id} outside 1..{len(tasks)} for suite={args.suite}")
        return 0

    task = tasks[task_id - 1]
    cmd = task.command(python_bin, args.cli_module, args.region_key, repo_root)
    command_str = shjoin(cmd)

    task_log_dir = exp_root / "task_logs"
    task_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = task_log_dir / f"{task.task_id:03d}_{task.name}.out"
    stderr_path = task_log_dir / f"{task.task_id:03d}_{task.name}.err"

    env = os.environ.copy()
    env["EXP_ROOT"] = str(exp_root)
    env["NCACHE_EXP_ROOT"] = str(exp_root)
    env["NCACHE_TASK_ID"] = str(task.task_id)
    env["NCACHE_TASK_NAME"] = task.name
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    started = now_iso()
    t0 = time.perf_counter()

    print(f"Experiment root: {exp_root}")
    print(f"Task {task.task_id}/{len(tasks)}: {task.group}/{task.name}")
    print(f"Command: {command_str}")
    print(f"stdout: {stdout_path}")
    print(f"stderr: {stderr_path}")

    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=out,
            stderr=err,
            text=True,
            check=False,
        )

    duration = time.perf_counter() - t0
    finished = now_iso()

    base = task_as_row(task, command_str)
    status_row = {
        **base,
        "exit_code": str(proc.returncode),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", "NA"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "NA"),
        "host": os.uname().nodename,
        "started_at": started,
        "finished_at": finished,
        "duration_sec": f"{duration:.3f}",
    }

    append_dict_row_locked(
        exp_root / "task_status.csv",
        exp_root / "locks" / "status.lock",
        STATUS_FIELDS,
        status_row,
        delimiter="\t",
    )

    if proc.returncode != 0:
        failed_row = {**base, "exit_code": str(proc.returncode)}
        append_dict_row_locked(
            exp_root / "failed_commands.tsv",
            exp_root / "locks" / "failed.lock",
            FAILED_FIELDS,
            failed_row,
            delimiter="\t",
        )

    # Update merged CSVs opportunistically after every task. Final task completion
    # leaves the merged files in their final state.
    with (exp_root / "locks" / "collect.lock").open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        collect_csv_outputs(exp_root)
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    if proc.returncode != 0:
        print(f"Task failed with exit code {proc.returncode}")
        print(f"See stderr: {stderr_path}")
        return proc.returncode

    print(f"Task completed successfully in {duration:.1f}s")
    return 0


def print_plan(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    tasks = build_tasks(args.suite, args.datasets, args.main_trials, args.ablation_trials)
    print(f"suite={args.suite}")
    print(f"n_tasks={len(tasks)}")
    for task in tasks:
        cmd = task.command(sys.executable, args.cli_module, args.region_key, repo_root)
        print(f"{task.task_id:03d}\t{task.group}\t{task.name}\t{shjoin(cmd)}")
    return 0


def collect_only(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    exp_root = Path(args.exp_root).resolve() if args.exp_root else default_exp_root(repo_root)
    collect_csv_outputs(exp_root)
    print(f"Collected CSV outputs under: {exp_root}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file NeighborCache Slurm experiment worker.")
    parser.add_argument("--run-task", action="store_true", help="Run the task selected by SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--task-id", type=int, default=None, help="Manual task id for local debugging.")
    parser.add_argument("--print-plan", action="store_true", help="Print deterministic task plan and exit.")
    parser.add_argument("--collect", action="store_true", help="Collect/merge CSV outputs and exit.")

    parser.add_argument("--suite", choices=["smoke", "paper", "all"], default=os.environ.get("SUITE", "paper"))
    parser.add_argument("--datasets", nargs="*", choices=sorted(DATASETS.keys()), default=None)

    parser.add_argument("--cli-module", default=os.environ.get("CLI_MODULE", CLI_MODULE_DEFAULT))
    parser.add_argument("--region-key", default=os.environ.get("REGION_KEY", REGION_KEY_DEFAULT))
    parser.add_argument("--exp-root", default=os.environ.get("EXP_ROOT"))

    parser.add_argument("--main-trials", type=int, default=int(os.environ.get("MAIN_TRIALS", str(MAIN_TRIALS))))
    parser.add_argument("--ablation-trials", type=int, default=int(os.environ.get("ABLATION_TRIALS", str(ABLATION_TRIALS))))

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.print_plan:
        return print_plan(args)
    if args.collect:
        return collect_only(args)
    if args.run_task:
        return run_task(args)

    print("Nothing to do. Use --run-task, --print-plan, or --collect.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
