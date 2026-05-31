#!/usr/bin/env python3
"""
DEPRECATED. Use `python -m experiments.launchers.submit_benchmark_array` instead.
Kept for reference; will be removed once the new launcher is validated.

run_paper_experiments_normal.py

Compact Slurm launcher for the FIRST paper experiment run.

Purpose:
  Run the normal configuration on two datasets:
    1. NeighborCache/data/h1h0_final.npz
    2. NeighborCache/data/lmsys_h1h0_by_cluster_h1keep_with_unormalized_embedding_fixed.npz

This script intentionally does NOT run the "without WhitenedCosine" ensemble variant.
We will generate a second launcher for that after the normal run finishes.

Why this version:
  - Uses rounded paper-friendly sizes: 1200 / 1200 / 1200.
  - Avoids huge seed/local-threshold/Hadamard grids.
  - Uses only CLI flags that already appeared in the previous Bash launcher.
  - Produces commands.txt, commands.tsv, failed_commands.tsv, metadata, and an sbatch file.

First:
  python run_paper_experiments_normal.py --suite smoke --dry-run

Then:
  python run_paper_experiments_normal.py --suite smoke --max-parallel 2

If smoke passes:
  python run_paper_experiments_normal.py --suite paper --max-parallel 4

Optional full including smoke:
  python run_paper_experiments_normal.py --suite all --max-parallel 4
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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

    def command(self, python_bin: str, cli_module: str, region_key: str) -> list[str]:
        cmd = [
            python_bin,
            "-m",
            cli_module,
            "--data",
            self.dataset_path,
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


def resolve_python(conda_env: str | None, explicit_python: str | None) -> str:
    if explicit_python:
        return explicit_python

    if conda_env:
        try:
            out = subprocess.check_output(
                [
                    "conda",
                    "run",
                    "-n",
                    conda_env,
                    "python",
                    "-c",
                    "import sys; print(sys.executable)",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                return out.splitlines()[-1].strip()
        except Exception:
            pass

        for conda_sh in [
            Path.home() / "miniconda3/etc/profile.d/conda.sh",
            Path.home() / "anaconda3/etc/profile.d/conda.sh",
            Path("/opt/conda/etc/profile.d/conda.sh"),
        ]:
            if conda_sh.exists():
                cmd = (
                    f"source {shlex.quote(str(conda_sh))} && "
                    f"conda run -n {shlex.quote(conda_env)} "
                    "python -c 'import sys; print(sys.executable)'"
                )
                try:
                    out = subprocess.check_output(
                        ["bash", "-lc", cmd],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    ).strip()
                    if out:
                        return out.splitlines()[-1].strip()
                except Exception:
                    pass

    return sys.executable


def check_python_env(python_bin: str, skip: bool) -> None:
    if skip:
        return

    code = """
import sys
required = ["numpy", "sklearn"]
missing = []
for mod in required:
    try:
        __import__(mod)
    except Exception as e:
        missing.append((mod, repr(e)))
if missing:
    print("ERROR: Missing required Python packages:", file=sys.stderr)
    for mod, err in missing:
        print(f"  {mod}: {err}", file=sys.stderr)
    print(f"Python: {sys.executable}", file=sys.stderr)
    sys.exit(3)
print("Python environment check OK:", sys.executable)
""".strip()

    subprocess.check_call([python_bin, "-c", code])


def git_commit_or_na() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "NA"


def add_task(
    tasks: list[Task],
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


def build_tasks(suite: str, datasets: list[str] | None, main_trials: int, ablation_trials: int) -> list[Task]:
    selected = DATASETS if not datasets else {k: DATASETS[k] for k in datasets}
    tasks: list[Task] = []

    include_smoke = suite in {"smoke", "all"}
    include_paper = suite in {"paper", "all"}

    if include_smoke:
        for dataset_id, dataset_path in selected.items():
            add_task(
                tasks,
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
                add_task(
                    tasks,
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
                add_task(
                    tasks,
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
                add_task(
                    tasks,
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


def write_files(
    *,
    exp_root: Path,
    tasks: list[Task],
    python_bin: str,
    cli_module: str,
    region_key: str,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path]:
    (exp_root / "logs").mkdir(parents=True, exist_ok=True)
    (exp_root / "slurm").mkdir(parents=True, exist_ok=True)

    commands_file = exp_root / "commands.txt"
    commands_tsv = exp_root / "commands.tsv"
    failed_tsv = exp_root / "failed_commands.tsv"
    metadata_file = exp_root / "run_metadata.txt"
    sbatch_file = exp_root / "slurm" / "neighborcache_experiment_array.sbatch"

    with commands_file.open("w", encoding="utf-8") as f_cmd, commands_tsv.open("w", encoding="utf-8") as f_tsv:
        f_tsv.write(
            "\t".join(
                [
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
            )
            + "\n"
        )

        for task in tasks:
            cmd = task.command(python_bin, cli_module, region_key)
            cmd_str = shjoin(cmd)
            f_cmd.write(cmd_str + "\n")
            f_tsv.write(
                "\t".join(
                    [
                        str(task.task_id),
                        task.group,
                        task.name,
                        task.dataset_id,
                        task.dataset_path,
                        f"{task.alpha:.4g}",
                        task.tau_mode,
                        str(task.seed),
                        str(task.n_train),
                        str(task.n_calib),
                        str(task.n_eval),
                        "1" if task.hadamard else "0",
                        str(task.n_trials),
                        cmd_str,
                    ]
                )
                + "\n"
            )

    with failed_tsv.open("w", encoding="utf-8") as f:
        f.write(
            "\t".join(
                [
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
                    "exit_code",
                    "command",
                ]
            )
            + "\n"
        )

    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with metadata_file.open("w", encoding="utf-8") as f:
        f.write(f"experiment_root={exp_root}\n")
        f.write(f"date={now}\n")
        f.write(f"pwd={Path.cwd()}\n")
        f.write(f"python_bin={python_bin}\n")
        f.write(f"cli_module={cli_module}\n")
        f.write(f"region_key={region_key}\n")
        f.write(f"suite={args.suite}\n")
        f.write(f"datasets={args.datasets or 'all'}\n")
        f.write(f"n_tasks={len(tasks)}\n")
        f.write(f"max_parallel={args.max_parallel}\n")
        f.write(f"git_commit={git_commit_or_na()}\n")
        f.write(f"full_n_train={FULL_N_TRAIN}\n")
        f.write(f"full_n_calib={FULL_N_CALIB}\n")
        f.write(f"full_n_eval={FULL_N_EVAL}\n")
        f.write(f"alphas={ALPHAS}\n")
        f.write(f"train_sizes={TRAIN_SIZES}\n")
        f.write(f"calib_sizes={CALIB_SIZES}\n")
        f.write(f"main_trials={args.main_trials}\n")
        f.write(f"ablation_trials={args.ablation_trials}\n")
        f.write("note=normal run; no without-whitened ensemble variant in this script\n")

    sbatch_text = """#!/usr/bin/env bash
set -euo pipefail

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date --iso-8601=seconds)"
echo "EXP_ROOT=${EXP_ROOT}"
echo "COMMANDS_FILE=${COMMANDS_FILE}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
CMD="$(sed -n "${TASK_ID}p" "${COMMANDS_FILE}")"

if [[ -z "${CMD}" ]]; then
  echo "ERROR: No command found for task ${TASK_ID}" >&2
  exit 10
fi

echo "Running command:"
echo "${CMD}"
echo ""

set +e
bash -lc "${CMD}"
EXIT_CODE=$?
set -e

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  echo "Command failed with exit code ${EXIT_CODE}" >&2

  if [[ -n "${COMMANDS_TSV:-}" && -f "${COMMANDS_TSV}" && -n "${FAILED_TSV:-}" ]]; then
    ROW="$(awk -F '\t' -v id="${TASK_ID}" 'NR>1 && $1==id {print; exit}' "${COMMANDS_TSV}")"
    if [[ -n "${ROW}" ]]; then
      echo -e "${ROW}\t${EXIT_CODE}" >> "${FAILED_TSV}"
    else
      echo -e "${TASK_ID}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t${EXIT_CODE}\t${CMD}" >> "${FAILED_TSV}"
    fi
  fi

  exit "${EXIT_CODE}"
fi

echo ""
echo "Task ${TASK_ID} completed successfully."
"""
    sbatch_file.write_text(sbatch_text, encoding="utf-8")
    sbatch_file.chmod(0o755)

    return commands_file, commands_tsv, failed_tsv, metadata_file, sbatch_file


def submit(
    *,
    sbatch_file: Path,
    commands_file: Path,
    commands_tsv: Path,
    failed_tsv: Path,
    exp_root: Path,
    n_tasks: int,
    args: argparse.Namespace,
) -> None:
    if n_tasks <= 0:
        raise RuntimeError("No tasks generated")

    sbatch_args = [
        "sbatch",
        f"--job-name={args.job_name}",
        f"--array=1-{n_tasks}%{args.max_parallel}",
        f"--time={args.time}",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem={args.mem}",
        f"--output={exp_root}/slurm/%x_%A_%a.out",
        f"--error={exp_root}/slurm/%x_%A_%a.err",
        (
            "--export="
            f"ALL,COMMANDS_FILE={commands_file},COMMANDS_TSV={commands_tsv},"
            f"FAILED_TSV={failed_tsv},EXP_ROOT={exp_root}"
        ),
    ]

    if args.partition:
        sbatch_args.append(f"--partition={args.partition}")
    if args.account:
        sbatch_args.append(f"--account={args.account}")
    if args.qos:
        sbatch_args.append(f"--qos={args.qos}")
    if args.gpus != "0":
        sbatch_args.append(f"--gres=gpu:{args.gpus}")

    sbatch_args.append(str(sbatch_file))

    print()
    print("Submit command:")
    print(shjoin(sbatch_args))

    if args.dry_run:
        print()
        print("DRY_RUN enabled; not submitting.")
        return

    subprocess.check_call(sbatch_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate normal NeighborCache paper Slurm experiments.")

    parser.add_argument("--suite", choices=["smoke", "paper", "all"], default="paper")
    parser.add_argument("--datasets", nargs="*", choices=sorted(DATASETS.keys()), default=None)

    parser.add_argument("--conda-env", default=os.environ.get("CONDA_ENV_NAME", "ec"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN"))
    parser.add_argument("--skip-python-check", action="store_true")

    parser.add_argument("--cli-module", default=os.environ.get("CLI_MODULE", CLI_MODULE_DEFAULT))
    parser.add_argument("--region-key", default=os.environ.get("REGION_KEY", REGION_KEY_DEFAULT))

    default_root = (
        Path("NeighborCache/results")
        / f"paper_normal_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    parser.add_argument("--exp-root", default=os.environ.get("EXP_ROOT", str(default_root)))

    parser.add_argument("--main-trials", type=int, default=int(os.environ.get("MAIN_TRIALS", str(MAIN_TRIALS))))
    parser.add_argument("--ablation-trials", type=int, default=int(os.environ.get("ABLATION_TRIALS", str(ABLATION_TRIALS))))

    parser.add_argument("--job-name", default=os.environ.get("JOB_NAME", "ncache_paper"))
    parser.add_argument("--partition", default=os.environ.get("PARTITION", ""))
    parser.add_argument("--account", default=os.environ.get("ACCOUNT", ""))
    parser.add_argument("--qos", default=os.environ.get("QOS", ""))
    parser.add_argument("--time", default=os.environ.get("TIME", "12:00:00"))
    parser.add_argument("--cpus-per-task", default=os.environ.get("CPUS_PER_TASK", "4"))
    parser.add_argument("--mem", default=os.environ.get("MEM", "16G"))
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0"))
    parser.add_argument("--max-parallel", type=int, default=int(os.environ.get("MAX_PARALLEL", "4")))

    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "0") == "1")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    python_bin = resolve_python(args.conda_env, args.python_bin)
    print(f"Using Python: {python_bin}")
    check_python_env(python_bin, args.skip_python_check)

    tasks = build_tasks(
        suite=args.suite,
        datasets=args.datasets,
        main_trials=args.main_trials,
        ablation_trials=args.ablation_trials,
    )

    exp_root = Path(args.exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)

    commands_file, commands_tsv, failed_tsv, metadata_file, sbatch_file = write_files(
        exp_root=exp_root,
        tasks=tasks,
        python_bin=python_bin,
        cli_module=args.cli_module,
        region_key=args.region_key,
        args=args,
    )

    print()
    print(f"Generated {len(tasks)} Slurm array tasks.")
    print(f"Experiment root: {exp_root}")
    print(f"Commands file:   {commands_file}")
    print(f"Commands TSV:    {commands_tsv}")
    print(f"Failed TSV:      {failed_tsv}")
    print(f"Metadata:        {metadata_file}")
    print(f"Sbatch file:     {sbatch_file}")

    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.group] = counts.get(task.group, 0) + 1

    print()
    print("Task groups:")
    for group, count in sorted(counts.items()):
        print(f"  {group:24s} {count:4d}")

    submit(
        sbatch_file=sbatch_file,
        commands_file=commands_file,
        commands_tsv=commands_tsv,
        failed_tsv=failed_tsv,
        exp_root=exp_root,
        n_tasks=len(tasks),
        args=args,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
