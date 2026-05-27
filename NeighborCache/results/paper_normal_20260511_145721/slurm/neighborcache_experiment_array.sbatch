#!/usr/bin/env bash
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
    ROW="$(awk -F '	' -v id="${TASK_ID}" 'NR>1 && $1==id {print; exit}' "${COMMANDS_TSV}")"
    if [[ -n "${ROW}" ]]; then
      echo -e "${ROW}	${EXIT_CODE}" >> "${FAILED_TSV}"
    else
      echo -e "${TASK_ID}	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	${EXIT_CODE}	${CMD}" >> "${FAILED_TSV}"
    fi
  fi

  exit "${EXIT_CODE}"
fi

echo ""
echo "Task ${TASK_ID} completed successfully."
