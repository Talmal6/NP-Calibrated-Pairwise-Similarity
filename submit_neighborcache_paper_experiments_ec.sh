#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-ec}"

resolve_conda_python() {
  local env_name="$1"

  if command -v conda >/dev/null 2>&1; then
    conda run -n "${env_name}" python -c 'import sys; print(sys.executable)' 2>/dev/null && return 0
  fi

  for conda_sh in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    if [[ -f "${conda_sh}" ]]; then
      source "${conda_sh}"
      conda run -n "${env_name}" python -c 'import sys; print(sys.executable)' 2>/dev/null && return 0
    fi
  done

  return 1
}

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "Resolving Python executable from Conda env: ${CONDA_ENV_NAME}"
  if ! PYTHON_BIN="$(resolve_conda_python "${CONDA_ENV_NAME}")"; then
    echo "ERROR: Could not resolve Python for Conda env '${CONDA_ENV_NAME}'."
    exit 2
  fi
fi

echo "Using PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import sys
required = ["numpy", "optuna", "sklearn"]
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
PY

DATA="${DATA:-NeighborCache/data/h1h0_final.npz}"
CLI_MODULE="${CLI_MODULE:-NeighborCache.region_local_threshold.cli}"
REGION_KEY="${REGION_KEY:-global_cluster}"

EXP_ROOT="${EXP_ROOT:-NeighborCache/results/paper_experiments_slurm_$(date +%Y%m%d_%H%M%S)}"

FULL_N_TRAIN="${FULL_N_TRAIN:-1270}"
FULL_N_CALIB="${FULL_N_CALIB:-1240}"
FULL_N_EVAL="${FULL_N_EVAL:-1270}"

SMOKE_TRIALS="${SMOKE_TRIALS:-5}"
MAIN_TRIALS="${MAIN_TRIALS:-50}"
ABLATION_TRIALS="${ABLATION_TRIALS:-50}"

ALPHAS=(${ALPHAS:-0.01 0.03 0.05 0.10})
SEEDS=(${SEEDS:-42 43 44 45 46})
TRAIN_SIZES=(${TRAIN_SIZES:-100 250 500 750 1000 1270})
CALIB_SIZES=(${CALIB_SIZES:-100 250 500 750 1000 1240})
TAU_MODES=(${TAU_MODES:-global local})

RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_MAIN_ALPHA_SWEEP="${RUN_MAIN_ALPHA_SWEEP:-1}"
RUN_SEED_ROBUSTNESS="${RUN_SEED_ROBUSTNESS:-1}"
RUN_TRAIN_SIZE_ABLATION="${RUN_TRAIN_SIZE_ABLATION:-1}"
RUN_CALIB_SIZE_ABLATION="${RUN_CALIB_SIZE_ABLATION:-1}"
RUN_HADAMARD_ABLATION="${RUN_HADAMARD_ABLATION:-1}"
RUN_TAU_MODE_ABLATION="${RUN_TAU_MODE_ABLATION:-1}"

SBATCH_FILE="${SBATCH_FILE:-./neighborcache_experiment_array.sbatch}"

JOB_NAME="${JOB_NAME:-ncache_exp}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"

TIME="${TIME:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-16G}"
GPUS="${GPUS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
ENV_SETUP="${ENV_SETUP:-}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${EXP_ROOT}/logs" "${EXP_ROOT}/slurm"

COMMANDS_FILE="${EXP_ROOT}/commands.txt"
COMMANDS_TSV="${EXP_ROOT}/commands.tsv"
FAILED_TSV="${EXP_ROOT}/failed_commands.tsv"
METADATA_FILE="${EXP_ROOT}/run_metadata.txt"

: > "${COMMANDS_FILE}"
echo -e "task_id\tgroup\tname\talpha\ttau_mode\tseed\tn_train\tn_calib\tn_eval\thadamard\tn_trials\tcommand" > "${COMMANDS_TSV}"
echo -e "task_id\tgroup\tname\talpha\ttau_mode\tseed\tn_train\tn_calib\tn_eval\thadamard\tn_trials\texit_code\tcommand" > "${FAILED_TSV}"

TASK_ID=0

append_exp() {
  local group="$1"
  local name="$2"
  local alpha="$3"
  local tau_mode="$4"
  local seed="$5"
  local n_train="$6"
  local n_calib="$7"
  local n_eval="$8"
  local hadamard="$9"
  local n_trials="${10}"

  local had_flag=""
  if [[ "${hadamard}" == "1" ]]; then
    had_flag="--hadamard_preprocess"
  fi

  local cmd
  cmd="${PYTHON_BIN} -m ${CLI_MODULE} --data ${DATA} --region_key ${REGION_KEY} --alpha ${alpha} --tau_mode ${tau_mode} --n_trials ${n_trials} --seed ${seed} --n_train ${n_train} --n_calib ${n_calib} --n_eval ${n_eval} ${had_flag}"

  TASK_ID=$((TASK_ID + 1))
  echo "${cmd}" >> "${COMMANDS_FILE}"
  echo -e "${TASK_ID}\t${group}\t${name}\t${alpha}\t${tau_mode}\t${seed}\t${n_train}\t${n_calib}\t${n_eval}\t${hadamard}\t${n_trials}\t${cmd}" >> "${COMMANDS_TSV}"
}

if [[ "${RUN_SMOKE}" == "1" ]]; then
  append_exp "smoke" "smoke_small_data" "0.05" "global" "42" "300" "300" "300" "1" "${SMOKE_TRIALS}"
fi

if [[ "${RUN_MAIN_ALPHA_SWEEP}" == "1" ]]; then
  for alpha in "${ALPHAS[@]}"; do
    append_exp "main_alpha_sweep" "full_data_alpha_${alpha}" "${alpha}" "global" "42" "${FULL_N_TRAIN}" "${FULL_N_CALIB}" "${FULL_N_EVAL}" "1" "${MAIN_TRIALS}"
  done
fi

if [[ "${RUN_SEED_ROBUSTNESS}" == "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    append_exp "seed_robustness" "alpha_0.05_seed_${seed}" "0.05" "global" "${seed}" "${FULL_N_TRAIN}" "${FULL_N_CALIB}" "${FULL_N_EVAL}" "1" "${MAIN_TRIALS}"
  done
fi

if [[ "${RUN_TRAIN_SIZE_ABLATION}" == "1" ]]; then
  for n_train in "${TRAIN_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      append_exp "train_size_ablation" "n_train_${n_train}_seed_${seed}" "0.05" "global" "${seed}" "${n_train}" "${FULL_N_CALIB}" "${FULL_N_EVAL}" "1" "${ABLATION_TRIALS}"
    done
  done
fi

if [[ "${RUN_CALIB_SIZE_ABLATION}" == "1" ]]; then
  for n_calib in "${CALIB_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      append_exp "calib_size_ablation" "n_calib_${n_calib}_seed_${seed}" "0.05" "global" "${seed}" "${FULL_N_TRAIN}" "${n_calib}" "${FULL_N_EVAL}" "1" "${ABLATION_TRIALS}"
    done
  done
fi

if [[ "${RUN_HADAMARD_ABLATION}" == "1" ]]; then
  for hadamard in 0 1; do
    for seed in "${SEEDS[@]}"; do
      append_exp "hadamard_ablation" "hadamard_${hadamard}_seed_${seed}" "0.05" "global" "${seed}" "${FULL_N_TRAIN}" "${FULL_N_CALIB}" "${FULL_N_EVAL}" "${hadamard}" "${ABLATION_TRIALS}"
    done
  done
fi

if [[ "${RUN_TAU_MODE_ABLATION}" == "1" ]]; then
  for tau_mode in "${TAU_MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      append_exp "tau_mode_ablation" "tau_${tau_mode}_seed_${seed}" "0.05" "${tau_mode}" "${seed}" "${FULL_N_TRAIN}" "${FULL_N_CALIB}" "${FULL_N_EVAL}" "1" "${ABLATION_TRIALS}"
    done
  done
fi

N_TASKS="$(wc -l < "${COMMANDS_FILE}" | tr -d ' ')"

{
  echo "experiment_root=${EXP_ROOT}"
  echo "date=$(date --iso-8601=seconds)"
  echo "pwd=$(pwd)"
  echo "conda_env_name=${CONDA_ENV_NAME}"
  echo "python_bin=${PYTHON_BIN}"
  echo "n_tasks=${N_TASKS}"
  echo "max_parallel=${MAX_PARALLEL}"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo NA)"
} > "${METADATA_FILE}"

echo ""
echo "Generated ${N_TASKS} Slurm array tasks."
echo "Experiment root: ${EXP_ROOT}"
echo "Commands file: ${COMMANDS_FILE}"
echo "Commands TSV: ${COMMANDS_TSV}"
echo "Metadata: ${METADATA_FILE}"

SBATCH_ARGS=()
SBATCH_ARGS+=(--job-name="${JOB_NAME}")
SBATCH_ARGS+=(--array="1-${N_TASKS}%${MAX_PARALLEL}")
SBATCH_ARGS+=(--time="${TIME}")
SBATCH_ARGS+=(--cpus-per-task="${CPUS_PER_TASK}")
SBATCH_ARGS+=(--mem="${MEM}")
SBATCH_ARGS+=(--output="${EXP_ROOT}/slurm/%x_%A_%a.out")
SBATCH_ARGS+=(--error="${EXP_ROOT}/slurm/%x_%A_%a.err")
SBATCH_ARGS+=(--export="ALL,COMMANDS_FILE=${COMMANDS_FILE},COMMANDS_TSV=${COMMANDS_TSV},FAILED_TSV=${FAILED_TSV},EXP_ROOT=${EXP_ROOT},ENV_SETUP=${ENV_SETUP}")

if [[ -n "${PARTITION}" ]]; then SBATCH_ARGS+=(--partition="${PARTITION}"); fi
if [[ -n "${ACCOUNT}" ]]; then SBATCH_ARGS+=(--account="${ACCOUNT}"); fi
if [[ -n "${QOS}" ]]; then SBATCH_ARGS+=(--qos="${QOS}"); fi
if [[ "${GPUS}" != "0" ]]; then SBATCH_ARGS+=(--gres="gpu:${GPUS}"); fi

echo ""
echo "sbatch ${SBATCH_ARGS[*]} ${SBATCH_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, not submitting."
  exit 0
fi

sbatch "${SBATCH_ARGS[@]}" "${SBATCH_FILE}"
