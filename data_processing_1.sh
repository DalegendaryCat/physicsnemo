#!/usr/bin/env bash
# pipeline.sh
# Full pipeline: reconstruct train/val/test split → ETL
# IMPORTANT: change data_sources.py StructGrid to XMLUnstrcGrid AND path.py HDBPath2 to HDBPath
# Usage: bash pipeline.sh

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
# change directories before running
VENV="/home/nguye/physicsnemo/venv-wsl"
RECONSTRUCT_SRC="/home/nguye/physicsnemo/"
CURATOR_SRC="/home/nguye/physicsnemo/external_aerodynamics_curator/"
CASES_DIR="/home/nguye/physicsnemo/simulation_data/simulation_data"
CASES_OUTPUT="/home/nguye/physicsnemo/Dataset/HDB_1_input"
ETL_OUTPUT="/home/nguye/physicsnemo/Dataset/HDB_1_processed"

# Split ratios (must sum to 1.0)
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1

RANDOM_SEED=42

# ─── Helpers ─────────────────────────────────────────────────────────────────

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }
ok()  { echo -e "\033[1;32m  ✓ $*\033[0m"; }
err() { echo -e "\033[1;31m  ✗ $*\033[0m"; exit 1; }

# ─── Activate venv ───────────────────────────────────────────────────────────

log "Activating virtual environment..."
source "${VENV}/bin/activate"
ok "venv: $(python3 --version)"

# ─── Step 1: Reconstruct + Rotate + STL + Split cases ────────────────────────

log "Step 1 — Reconstruct cases + rotate to North + generate STL"
cd "${RECONSTRUCT_SRC}"

python3 reconstruct_cases_with_rotation.py "${CASES_DIR}" "${CASES_OUTPUT}"
ok "Reconstruction complete"

# ─── Step 2: ETL ─────────────────────────────────────────────────────────────

log "Step 2 — Run ETL pipeline"
cd "${CURATOR_SRC}"

mkdir -p "${ETL_OUTPUT}"

python3 run_etl.py \
    --config-dir=config \
    --config-name=external_aero_etl_hdb_non_dir \
    etl.source.input_dir="${CASES_OUTPUT}/train" \
    etl.sink.output_dir="${ETL_OUTPUT}/train" \
    etl.common.model_type=combined \
    serialization_format=numpy

python3 run_etl.py \
    --config-dir=config \
    --config-name=external_aero_etl_hdb_non_dir \
    etl.source.input_dir="${CASES_OUTPUT}/test" \
    etl.sink.output_dir="${ETL_OUTPUT}/test" \
    etl.common.model_type=combined \
    serialization_format=numpy

python3 run_etl.py \
    --config-dir=config \
    --config-name=external_aero_etl_hdb_non_dir \
    etl.source.input_dir="${CASES_OUTPUT}/validation" \
    etl.sink.output_dir="${ETL_OUTPUT}/validation" \
    etl.common.model_type=combined \
    serialization_format=numpy

ok "ETL complete — NPZ files in ${ETL_OUTPUT}"


# ─── Done ────────────────────────────────────────────────────────────────────

log "Pipeline finished successfully"
echo ""
echo "  Output:"
echo "    Train:      ${ETL_OUTPUT}/train/"
echo "    Validation: ${ETL_OUTPUT}/validation/"
echo "    Test:       ${ETL_OUTPUT}/test/"
echo ""