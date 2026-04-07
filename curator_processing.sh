#!/usr/bin/env bash
# pipeline.sh
# Full pipeline: reconstruct train/val/test split → ETL
# IMPORTANT: change data_sources.py StructGrid to XMLUnstrcGrid AND path.py HDBPath2 to HDBPath
# Usage: bash pipeline.sh
set -euo pipefail
# ─── Config ──────────────────────────────────────────────────────────────────
# change directories before running
VENV="/home/nguye/physicsnemo/venv-wsl"
RECONSTRUCT_SRC="/home/nguye/physicsnemo/external_aerodynamics_curator"
CURATOR_SRC="/home/nguye/physicsnemo/external_aerodynamics_curator"
CASES_DIR="/home/nguye/physicsnemo/simulation_data/"
CASES_OUTPUT="/home/nguye/physicsnemo/Dataset/testing_output"
ETL_OUTPUT="/home/nguye/physicsnemo/Dataset/testing_output_processed"
# Split ratios (must sum to 1.0)
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
RANDOM_SEED=42
# ─── Helpers ─────────────────────────────────────────────────────────────────
log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }
ok()  { echo -e "\033[1;32m  ✓ $*\033[0m"; }
err() { echo -e "\033[1;31m  ✗ $*\033[0m"; exit 1; }
# ─── Activate conda env ──────────────────────────────────────────────────────
log "Activating conda environment..."
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "${VENV}"
source "${VENV}/bin/activate"
ok "venv: $(python3 --version)"
# ─── Step 1: Reconstruct + foamToVTK + Rotate + Split cases───────────────────
log "Step 1 — Reconstruct cases + generate files + Rotate to North"
cd "${RECONSTRUCT_SRC}"
python3 reconstruct_op.py "${CASES_DIR}"
ok "Reconstruction complete"
python3 get_vtk_op.py "${CASES_DIR}"
ok "foamToVTK complete"
python3 extract_cases_op.py "${CASES_DIR}" "${CASES_OUTPUT}"\
ok "extract cases complete"
# ─── Step 2: ETL ─────────────────────────────────────────────────────────────
log "Step 2 — Run ETL pipeline"
cd "${CURATOR_SRC}"
mkdir -p "${ETL_OUTPUT}"
ETL_START=$SECONDS
for split in train validation; do
    split_dir="${CASES_OUTPUT}/${split}"
    count=$(find "${split_dir}" -name "*.vtu" -o -name "*.vtp" 2>/dev/null | wc -l)
    if [ "${count}" -eq 0 ]; then
        log "Skipping ETL for ${split} — no VTU/VTP files found in ${split_dir}"
        continue
    fi
    ok "${split}: ${count} files found — running ETL"
    SPLIT_START=$SECONDS
    python run_etl.py \
        --config-dir=config \
        --config-name=external_aero_etl_hdb_non_dir \
        etl.source.input_dir="${split_dir}" \
        etl.sink.output_dir="${ETL_OUTPUT}/${split}" \
        etl.common.model_type=combined \
        serialization_format=numpy
    SPLIT_ELAPSED=$(( SECONDS - SPLIT_START ))
    ok "${split} ETL done in ${SPLIT_ELAPSED}s ($(( SPLIT_ELAPSED / 60 ))m $(( SPLIT_ELAPSED % 60 ))s)"
done
ETL_ELAPSED=$(( SECONDS - ETL_START ))
ok "ETL complete in ${ETL_ELAPSED}s ($(( ETL_ELAPSED / 60 ))m $(( ETL_ELAPSED % 60 ))s) — NPZ files in ${ETL_OUTPUT}"
# ─── Done ────────────────────────────────────────────────────────────────────
log "Pipeline finished successfully"
echo ""
echo "  Output:"
echo "    Train:      ${ETL_OUTPUT}/train/"
echo "    Validation: ${ETL_OUTPUT}/validation/"
echo ""

