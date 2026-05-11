#!/usr/bin/env python3
# reconstruct_cases.py
# For each OpenFOAM case:
#   1. Check if time 1250 folder exists with all required variables — skip if complete
#   2. Run reconstructPar if needed
#   3. Rotate case to North using transformPoints
#   4. Generate building STL from VTK/buildings/*.vtk
#
# Usage: python3 reconstruct_cases.py [/path/to/cases/directory]

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import vtk

# ─── Config ──────────────────────────────────────────────────────────────────

TIME = "1250"
MAX_WORKERS = 4

# Required field files that must exist in the TIME folder for a case to be
# considered complete. Add or remove fields as needed.
REQUIRED_FIELDS = ["U", "p", "nut", "p_rgh", "T"]

# Rotation angles to align all cases to North
ROTATION_MAP = {
    "N": 0,    # no rotation needed
    "S": 180,
    "E": 90,
    "W": 270,
}

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


def check_time_folder(case_path):
    """
    Check if the TIME folder exists and contains all required fields.

    Returns:
        (exists, missing_fields)
        exists        — True if TIME folder exists
        missing_fields — list of field names not found in the TIME folder
    """
    time_dir = os.path.join(case_path, TIME)
    if not os.path.isdir(time_dir):
        return False, REQUIRED_FIELDS

    # Fields may be plain files or .gz compressed
    present = set()
    for f in os.listdir(time_dir):
        name = f.replace(".gz", "").strip()
        present.add(name)

    missing = [f for f in REQUIRED_FIELDS if f not in present]
    return True, missing


def find_buildings_vtk(case_path):
    """Find the buildings patch VTK file in VTK/buildings/."""
    buildings_dir = os.path.join(case_path, "VTK", "buildings")
    if not os.path.isdir(buildings_dir):
        return None
    for f in os.listdir(buildings_dir):
        if f.endswith(".vtk"):
            return os.path.join(buildings_dir, f)
    return None


def generate_stl(buildings_vtk_path, stl_path):
    """Convert buildings patch VTK (PolyData) to STL."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(buildings_vtk_path)
    reader.Update()
    poly = reader.GetOutput()
    if not poly or poly.GetNumberOfPoints() == 0:
        print(f"  [ERROR] Empty or unreadable: {buildings_vtk_path}")
        return False

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(reader.GetOutputPort())
    triangulate.Update()

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(stl_path)
    writer.SetInputConnection(triangulate.GetOutputPort())
    writer.Write()

    del reader, writer
    return True


# ─── Per-case processing ─────────────────────────────────────────────────────

def process_case(args):
    cases_dir, case_name = args
    case_path = os.path.join(cases_dir, case_name)
    status = {"case": case_name, "success": False, "skipped": False, "error": None, "notes": []}

    # ── 1. Check if already complete ─────────────────────────────────────────
    exists, missing = check_time_folder(case_path)

    if exists and not missing:
        status["success"] = True
        status["skipped"] = True
        status["notes"].append(f"Time {TIME} folder complete — skipping reconstructPar")
    else:
        if not exists:
            status["notes"].append(f"Time {TIME} folder missing — running reconstructPar")
        else:
            status["notes"].append(f"Time {TIME} folder incomplete — missing: {missing} — running reconstructPar")

        # ── 2. Reconstruct ────────────────────────────────────────────────────
        log_path = os.path.join(case_path, "reconstructPar.log")
        cmd = ["reconstructPar", "-time", TIME, "-case", case_path]
        if not run_command(cmd, log_path):
            status["error"] = "reconstructPar failed"
            return status
        status["notes"].append("reconstructPar done")

        # ── 3. Rotate to North ────────────────────────────────────────────────
        direction = case_name.split("_")[-1].upper()
        rotation_angle = ROTATION_MAP.get(direction, 0)
        if rotation_angle != 0:
            rotate_log = os.path.join(case_path, "transformPoints.log")
            rotate_cmd = [
                "transformPoints",
                "-rotateFields",
                "-case", case_path,
                f"Rz={rotation_angle}",
            ]
            if not run_command(rotate_cmd, rotate_log):
                status["error"] = f"transformPoints failed (Rz={rotation_angle})"
                return status
            status["notes"].append(f"Rotated {direction} → N (Rz={rotation_angle}°)")
        else:
            status["notes"].append("No rotation needed (already North)")

    # ── 4. Generate STL from buildings VTK ───────────────────────────────────
    # Run foamToVTK first if VTK/buildings/ doesn't exist yet
    buildings_vtk = find_buildings_vtk(case_path)
    if buildings_vtk is None:
        vtk_log = os.path.join(case_path, "foamToVTK.log")
        vtk_cmd = ["foamToVTK", "-time", TIME, "-case", case_path]
        if not run_command(vtk_cmd, vtk_log):
            status["error"] = "foamToVTK failed"
            return status
        status["notes"].append("foamToVTK done")
        buildings_vtk = find_buildings_vtk(case_path)

    if buildings_vtk:
        stl_path = os.path.join(case_path, "VTK", "buildings", f"{case_name}.stl")
        ok = generate_stl(buildings_vtk, stl_path)
        if ok:
            status["notes"].append(f"STL generated: {stl_path}")
        else:
            status["notes"].append("STL generation failed")
    else:
        status["notes"].append("No buildings VTK found — STL not generated")

    status["success"] = True
    return status


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    DEFAULT_CASES_DIR = "/home/nguye/physicsnemo/simulation_data/simulation_data"

    if len(sys.argv) < 2:
        cases_dir = DEFAULT_CASES_DIR
        print(f"No path provided — using default: {cases_dir}")
    else:
        cases_dir = os.path.abspath(sys.argv[1])

    total_start = time.perf_counter()

    print("============================================")
    print(" OpenFOAM Batch Reconstruct + Rotate + STL")
    print(f" Cases dir: {cases_dir}")
    print(f" Time:      {TIME}")
    print(f" Workers:   {MAX_WORKERS}")
    print(f" Required fields: {REQUIRED_FIELDS}")
    print("============================================\n")

    if not os.path.isdir(cases_dir):
        print(f"Error: '{cases_dir}' not found.")
        sys.exit(1)

    # Find valid OpenFOAM cases
    all_subdirs = sorted([
        d for d in os.listdir(cases_dir)
        if os.path.isdir(os.path.join(cases_dir, d))
    ])

    valid_cases, skipped_dirs = [], []
    for case_name in all_subdirs:
        case_path = os.path.join(cases_dir, case_name)
        has_system, has_processors = is_openfoam_case(case_path)
        if not has_system or not has_processors:
            reason = "no system/" if not has_system else "no processor* dirs"
            print(f"  Skipping '{case_name}' — {reason}")
            skipped_dirs.append(case_name)
        else:
            valid_cases.append(case_name)

    print(f"\nFound {len(valid_cases)} valid cases — processing with {MAX_WORKERS} workers...\n")

    # Process in parallel
    success, skipped, failed = [], [], []
    args_list = [(cases_dir, case_name) for case_name in valid_cases]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_case, args): args[1] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            case_name = result["case"]
            for note in result["notes"]:
                print(f"  [{case_name}] {note}")
            if result["success"]:
                if result["skipped"]:
                    print(f"  → {case_name}: SKIPPED (already complete)")
                    skipped.append(case_name)
                else:
                    print(f"  → {case_name}: DONE")
                    success.append(case_name)
            else:
                print(f"  → {case_name}: FAILED — {result['error']}")
                failed.append(case_name)
            print()

    total_elapsed = time.perf_counter() - total_start

    print("============================================")
    print(" Summary")
    print("============================================")
    print(f" Processed: {len(success)}")
    print(f" Skipped:   {len(skipped)} (already complete)")
    print(f" Failed:    {len(failed)}")
    for c in failed:
        print(f"   x {c}")
    print(f" Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("============================================")


if __name__ == "__main__":
    main()