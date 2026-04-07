#!/usr/bin/env python3
# 01_reconstruct.py
#
# Phase 1: Run reconstructPar on all valid OpenFOAM cases in parallel.

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Config ──────────────────────────────────────────────────────────────────

TIME = "1250"
RECONSTRUCT_OPTS = ["-time", TIME]
MAX_WORKERS = 4

DEFAULT_CASES_DIR = "/home/nguye/physicsnemo/simulation_data/simulation_data"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def is_openfoam_case(path):
    has_system = os.path.isdir(os.path.join(path, "system"))
    has_processors = any(
        d.startswith("processor") for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    )
    return has_system, has_processors


def run_command(cmd, log_path):
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode == 0


def process_case(args):
    cases_dir, case_name = args
    case_path = os.path.join(cases_dir, case_name)

    status = {"case": case_name, "success": False, "error": None}

    # Rename existing TIME folder to avoid stale data
    time_folder = os.path.join(case_path, TIME)
    time_folder_old = os.path.join(case_path, f"{TIME}.old")
    if os.path.isdir(time_folder):
        if os.path.isdir(time_folder_old):
            shutil.rmtree(time_folder_old)
        shutil.move(time_folder, time_folder_old)
        print(f"  [pre-check] {case_name}/{TIME} → {TIME}.old")

    log_path = os.path.join(case_path, "reconstructPar.log")
    cmd = ["reconstructPar"] + RECONSTRUCT_OPTS + ["-case", case_path]

    if not run_command(cmd, log_path):
        status["error"] = "reconstructPar failed"
        return status

    status["success"] = True
    return status


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        cases_dir = DEFAULT_CASES_DIR
        print(f"No cases dir provided — using default: {cases_dir}")
    else:
        cases_dir = os.path.abspath(sys.argv[1])

    if not os.path.isdir(cases_dir):
        print(f"Error: Cases directory '{cases_dir}' not found.")
        sys.exit(1)

    print("============================================")
    print(" Phase 1: reconstructPar")
    print(f" Cases dir: {cases_dir}")
    print(f" Time:      {TIME}")
    print(f" Workers:   {MAX_WORKERS}")
    print("============================================\n")

    all_cases = [
        d for d in os.listdir(cases_dir)
        if os.path.isdir(os.path.join(cases_dir, d))
    ]

    valid_cases = []
    skipped = []
    for case_name in all_cases:
        case_path = os.path.join(cases_dir, case_name)
        has_system, has_processors = is_openfoam_case(case_path)
        if not has_system or not has_processors:
            reason = "no system/ folder" if not has_system else "no processor* dirs"
            print(f"  Skipping '{case_name}' — {reason}")
            skipped.append(case_name)
        else:
            valid_cases.append(case_name)

    print(f"\nFound {len(valid_cases)} valid cases — reconstructing with {MAX_WORKERS} workers...\n")

    start = time.perf_counter()
    success = []
    failed = []

    args_list = [(cases_dir, c) for c in valid_cases]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_case, args): args[1] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            case_name = result["case"]
            if result["success"]:
                print(f"  + {case_name}")
                success.append(case_name)
            else:
                print(f"  x {case_name} — {result['error']}")
                failed.append(case_name)

    elapsed = time.perf_counter() - start

    print("\n============================================")
    print(" Summary")
    print("============================================")
    print(f" Succeeded: {len(success)}")
    for c in success:
        print(f"   + {c}")
    print(f" Failed:    {len(failed)}")
    for c in failed:
        print(f"   x {c}")
    print(f" Skipped:   {len(skipped)}")
    for c in skipped:
        print(f"   - {c}")
    print(f"\n Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("============================================")


if __name__ == "__main__":
    main()