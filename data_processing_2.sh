#!/usr/bin/env bash
# pipeline.sh
# Full pipeline: reconstruct → ETL → train/val/test split
# IMPORTANT: change data_sources.py XMLUnstrcGrid to StructGrid AND path.py HDBPath to HDBPath2
# Usage: bash pipeline.sh

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────

VENV="/home/nguye/physicsnemo/venv-wsl"
RECONSTRUCT_SRC="/home/nguye/physicsnemo/"
CURATOR_SRC="/home/nguye/physicsnemo/external_aerodynamics_curator/"
CASES_DIR="/home/nguye/physicsnemo/simulation_data/simulation_data"
ETL_OUTPUT="/home/nguye/physicsnemo/Dataset/test_vtk_curator"

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

# ─── Step 1: Reconstruct + Rotate + STL ──────────────────────────────────────

log "Step 1 — Reconstruct cases + rotate to North + generate STL"
cd "${RECONSTRUCT_SRC}"

python3 reconstruct_cases_with_rotation_2.py "${CASES_DIR}"
ok "Reconstruction complete"

# ─── Step 2: ETL ─────────────────────────────────────────────────────────────

log "Step 2 — Run ETL pipeline"
cd "${CURATOR_SRC}"

mkdir -p "${ETL_OUTPUT}"

python3 run_etl.py \
    --config-dir=config \
    --config-name=external_aero_etl_hdb_non_dir \
    etl.source.input_dir="${CASES_DIR}" \
    etl.sink.output_dir="${ETL_OUTPUT}" \
    etl.common.model_type=combined \
    serialization_format=numpy

ok "ETL complete — NPZ files in ${ETL_OUTPUT}"

# ─── Step 3: Split into train / validation / test ─────────────────────────────

log "Step 3 — Splitting into train/validation/test"

python3 - <<PYEOF
import os
import shutil
import random

output_dir  = "${ETL_OUTPUT}"
train_ratio = float("${TRAIN_RATIO}")
val_ratio   = float("${VAL_RATIO}")
test_ratio  = float("${TEST_RATIO}")
seed        = int("${RANDOM_SEED}")

# Find all NPZ files in the flat ETL output
all_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".npz")])

if not all_files:
    print("  No NPZ files found in ETL output — check ETL step.")
    exit(1)

print(f"  Total NPZ files: {len(all_files)}")

# Shuffle with fixed seed
random.seed(seed)
random.shuffle(all_files)

n       = len(all_files)
n_train = round(n * train_ratio)
n_val   = round(n * val_ratio)
n_test  = n - n_train - n_val

splits = {
    "train":      all_files[:n_train],
    "validation": all_files[n_train:n_train + n_val],
    "test":       all_files[n_train + n_val:],
}

for split_name, files in splits.items():
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    for f in files:
        src = os.path.join(output_dir, f)
        dst = os.path.join(split_dir, f)
        shutil.move(src, dst)
    print(f"  {split_name:12s}: {len(files)} files")

print(f"\n  Split complete  (seed={seed})")
print(f"  train={n_train}  val={n_val}  test={n_test}")
PYEOF

ok "Split complete"

# ─── Done ────────────────────────────────────────────────────────────────────

log "Pipeline finished successfully"
echo ""
echo "  Output:"
echo "    Train:      ${ETL_OUTPUT}/train/"
echo "    Validation: ${ETL_OUTPUT}/validation/"
echo "    Test:       ${ETL_OUTPUT}/test/"
echo ""