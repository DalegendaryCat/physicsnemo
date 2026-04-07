#!/usr/bin/env python3
# 02_foam_to_vtk.py
#
# Phase 2: Run foamToVTK on all valid OpenFOAM cases in parallel.
# Run this after 01_reconstruct.py has completed successfully.

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Config ──────────────────────────────────────────────────────────────────

TIME = "1250"
VTK_OPTS = ["-time", TIME, "-fields", "(U p nut)"]
MAX_WORKERS = 4

DEFAULT_CASES_DIR = "/home/nguye/physicsnemo/simulation_data/simulation_data"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def is_openfoam_case(path):
    has_system = os.path.isdir(os.path.join(path, "system"))
    has_time_dir = os.path.isdir(os.path.join(path, TIME))
    return has_system, has_time_dir


def run_command(cmd, log_path):
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode == 0


def process_case(args):
    cases_dir, case_name = args
    case_path = os.path.join(cases_dir, case_name)

    status = {"case": case_name, "success": False, "error": None}

    log_path = os.path.join(case_path, "foamToVTK.log")
    cmd = ["foamToVTK"] + VTK_OPTS + ["-case", case_path]

    if not run_command(cmd, log_path):
        status["error"] = "foamToVTK failed (check foamToVTK.log)"
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
    print(" Phase 2: foamToVTK")
    print(f" Cases dir: {cases_dir}")
    print(f" Time:      {TIME}")
    print(f" Fields:    {VTK_OPTS}")
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
        has_system, has_time_dir = is_openfoam_case(case_path)
        if not has_system:
            print(f"  Skipping '{case_name}' — no system/ folder")
            skipped.append(case_name)
        elif not has_time_dir:
            print(f"  Skipping '{case_name}' — no {TIME}/ folder (run 01_reconstruct.py first?)")
            skipped.append(case_name)
        else:
            valid_cases.append(case_name)

    print(f"\nFound {len(valid_cases)} valid cases — converting with {MAX_WORKERS} workers...\n")

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