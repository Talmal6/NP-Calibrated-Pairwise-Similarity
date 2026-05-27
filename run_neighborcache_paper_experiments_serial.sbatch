#!/usr/bin/env bash
#SBATCH --job-name=ncache_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=neighborcache_serial_%j.out
#SBATCH --error=neighborcache_serial_%j.err

set -euo pipefail

# ============================================================
# Serial Slurm wrapper around the original non-array runner.
# Use this only if you explicitly want one long serial job.
# The array version is usually better.
# ============================================================

ENV_SETUP="${ENV_SETUP:-}"
RUNNER="${RUNNER:-./run_neighborcache_paper_experiments.sh}"

if [[ -n "${ENV_SETUP}" ]]; then
  eval "${ENV_SETUP}"
fi

bash "${RUNNER}"
