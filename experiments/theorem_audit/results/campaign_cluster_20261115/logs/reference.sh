#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ntk-mpl-${SLURM_JOB_ID}"
.venv/bin/python experiments/theorem_audit/results/campaign_cluster_20261115/experiments/theorem_audit/campaign.py worker --directory experiments/theorem_audit/results/campaign_cluster_20261115 --kind reference --batch-size 1 --task-index "${SLURM_ARRAY_TASK_ID:-0}"
