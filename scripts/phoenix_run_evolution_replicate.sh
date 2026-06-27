#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="${LAB_ROOT:-/cs/labs/mornitzan/hrappeport}"
REPO_DIR="${REPO_DIR:-${LAB_ROOT}/RCCN_Evolution}"
VENV="${VENV:-${LAB_ROOT}/venvs/rccn_evolution_gpu}"

ANCESTOR_J="${ANCESTOR_J:-outputs/ancestor_aging_additive_update/J_matrices/ancestor_J_00.npy}"
TOPOLOGY="${TOPOLOGY:-data/topology_1.npy}"
REPLICATE_ID="${REPLICATE_ID:-0}"
SEED="${SEED:-2026062500}"
N_CYCLES="${N_CYCLES:-200}"
T_A="${T_A:-0}"
T_2="${T_2:-200}"
BACKEND="${BACKEND:-torch}"
DEVICE="${DEVICE:-cuda}"
RUN_LABEL="${RUN_LABEL:-evolution_j0_missing_rep0_batched_${SLURM_JOB_ID:-manual}}"
OUT_DIR="${OUT_DIR:-outputs/${RUN_LABEL}/replicate_${REPLICATE_ID}}"

cd "${REPO_DIR}"
git pull --ff-only

module load cuda/12.4.1 || true
source "${VENV}/bin/activate"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
python - <<'PY'
import rccn_evolution
import torch

print("rccn_evolution", getattr(rccn_evolution, "__file__", None), flush=True)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
print(
    "device",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    flush=True,
)
PY

python scripts/run_evolution_replicate.py \
  --ancestor-j "${ANCESTOR_J}" \
  --topology "${TOPOLOGY}" \
  --out-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --n-cycles "${N_CYCLES}" \
  --T-a "${T_A}" \
  --T-2 "${T_2}" \
  --backend "${BACKEND}" \
  --device "${DEVICE}"
