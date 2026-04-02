#!/usr/bin/env bash

#BASE_DIR="/scratch/phong/testing-cases"
#PY_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/verify.py"

BASE_DIR="/Users/nguyentuanphong/Desktop/ASTAR_Internship/HDB_dataset/simulation_data"
PY_SCRIPT="/Users/nguyentuanphong/Desktop/verify.py"
JOBS=2
OFFSETS="0.5 2 3"

START=$(date +%s)

for case_path in "$BASE_DIR"/*/; do
    case_name="$(basename "$case_path")"
    #output_dir="/scratch/phong/yh/verify_shell/${case_name}"
    output_dir="/Users/nguyentuanphong/Desktop/ASTAR_Internship/HDB_dataset/yh/${case_name}"
    mkdir -p "$output_dir"

    echo "Running: $case_name"
    CASE_PATH="$case_path" OUTPUT_DIR="$output_dir" python3 "$PY_SCRIPT" $OFFSETS &

    # Limit parallel jobs
    while [[ $(jobs -r | wc -l) -ge $JOBS ]]; do
        sleep 1
    done
done

wait
END=$(date +%s)
echo "All cases done. Total time: $((END - START))s"