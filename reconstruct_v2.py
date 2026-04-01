#!/usr/bin/env python3
# reconstruct_cases_vtk_rotation.py
#
# Reconstructs OpenFOAM cases, converts to VTK, and applies rotation ONLY
# during VTK → VTU/VTP/STL conversion. Original cases are never modified.

import os
import subprocess
import sys
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import vtk
from physicsnemo.utils.domino.vtk_file_utils import write_to_vtp, write_to_vtu

# ─── Config ──────────────────────────────────────────────────────────────────

TIME = "1250"
RECONSTRUCT_OPTS = ["-time", TIME]
VTK_OPTS = ["-time", TIME]
OUTPUT_DIR = "/home/nguye/physicsnemo/Dataset/hdb_input_rotated"
SPLIT = {"train": 0.8, "test": 0.1, "validation": 0.1}
RANDOM_SEED = 42
MAX_WORKERS = 4

REQUIRED_FIELDS = ["U", "p", "nut", "p_rgh", "T"]

ROTATION_MAP = {
    "N": 0,
    "S": 180,
    "E": 90,
    "W": 270,
}

# ─── OpenFOAM helpers ────────────────────────────────────────────────────────

def is_openfoam_case(path):
    has_system = os.path.isdir(os.path.join(path, "system"))
    has_processors = any(
        d.startswith("processor") for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    )
    return has_system, has_processors


def check_time_folder(case_path):
    time_dir = os.path.join(case_path, TIME)
    if not os.path.isdir(time_dir):
        return False, REQUIRED_FIELDS

    present = set()
    for f in os.listdir(time_dir):
        name = f.replace(".gz", "").strip()
        present.add(name)

    missing = [f for f in REQUIRED_FIELDS if f not in present]
    return len(missing) == 0, missing


def run_command(cmd, log_path):
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode == 0


def find_vtk_file(case_path, case_name):
    vtk_dir = os.path.join(case_path, "VTK")
    if not os.path.isdir(vtk_dir):
        return None
    for f in os.listdir(vtk_dir):
        if f.endswith(".vtk") and f.startswith(case_name):
            return os.path.join(vtk_dir, f)
    return None


def find_buildings_vtk_file(case_path):
    buildings_dir = os.path.join(case_path, "VTK", "buildings")
    if not os.path.isdir(buildings_dir):
        return None
    for f in os.listdir(buildings_dir):
        if f.endswith(".vtk"):
            return os.path.join(buildings_dir, f)
    return None


# ─── Rotation helper ──────────────────────────────────────────────────────────

def apply_rotation(data, angle_deg):
    transform = vtk.vtkTransform()
    transform.RotateZ(angle_deg)

    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(data)
    transform_filter.Update()

    return transform_filter.GetOutput()


# ─── File format conversion ───────────────────────────────────────────────────

def convert_vtk_to_vtu(vtk_filename: str, vtu_filename: str, angle_deg: float) -> None:
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_filename)
    reader.Update()

    data = reader.GetOutput()
    if not data:
        print(f"[ERROR] Failed to read {vtk_filename}")
        return

    rotated = apply_rotation(data, angle_deg)
    write_to_vtu(rotated, vtu_filename)


def convert_buildings_vtk_to_vtp(buildings_vtk: str, vtp_filename: str, angle_deg: float) -> None:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(buildings_vtk)
    reader.Update()

    data = reader.GetOutput()
    if not data or data.GetNumberOfPoints() == 0:
        print(f"[ERROR] Failed to read {buildings_vtk}")
        return

    rotated = apply_rotation(data, angle_deg)

    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputData(rotated)
    clean_filter.Update()

    write_to_vtp(clean_filter.GetOutput(), vtp_filename)


def convert_vtp_to_stl(vtp_filename: str, stl_filename: str) -> None:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_filename)
    reader.Update()

    data = reader.GetOutput()
    if not data or data.GetNumberOfPoints() == 0:
        print(f"[ERROR] Failed to read {vtp_filename}")
        return

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(reader.GetOutputPort())
    triangulate.Update()

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(stl_filename)
    writer.SetInputConnection(triangulate.GetOutputPort())
    writer.Write()


# ─── Per-case processing ─────────────────────────────────────────────────────

def process_case(args):
    cases_dir, case_name = args
    case_path = os.path.join(cases_dir, case_name)

    status = {"case": case_name, "success": False, "error": None, "skipped_recon": False}

    is_complete, missing = check_time_folder(case_path)

    if is_complete:
        print(f"  [SKIP] {case_name} — already reconstructed")
        status["skipped_recon"] = True
    else:
        log_path = os.path.join(case_path, "reconstructPar.log")
        cmd = ["reconstructPar"] + RECONSTRUCT_OPTS + ["-case", case_path]

        if not run_command(cmd, log_path):
            status["error"] = "reconstructPar failed"
            return status

    # ONLY run foamToVTK (no rotation!)
    vtk_log_path = os.path.join(case_path, "foamToVTK.log")
    vtk_cmd = ["foamToVTK"] + VTK_OPTS + ["-case", case_path]

    if not run_command(vtk_cmd, vtk_log_path):
        status["error"] = "foamToVTK failed"
        return status

    status["success"] = True
    return status


# ─── Split helper ────────────────────────────────────────────────────────────

def split_cases(case_names):
    cases = list(case_names)
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    random.shuffle(cases)

    n = len(cases)
    n_train = round(n * SPLIT["train"])
    n_test = round(n * SPLIT["test"])

    return {
        "train": cases[:n_train],
        "test": cases[n_train:n_train + n_test],
        "validation": cases[n_train + n_test:]
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    DEFAULT_CASES_DIR = "/home/nguye/physicsnemo/simulation_data/simulation_data"

    cases_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CASES_DIR
    output_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR

    total_start = time.perf_counter()

    print("============================================")
    print(" OpenFOAM → VTK → Rotated Dataset Pipeline")
    print("============================================\n")

    for split in SPLIT:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    all_cases = [
        d for d in os.listdir(cases_dir)
        if os.path.isdir(os.path.join(cases_dir, d))
    ]

    valid_cases = []
    for case_name in all_cases:
        case_path = os.path.join(cases_dir, case_name)
        has_system, has_processors = is_openfoam_case(case_path)
        if has_system and has_processors:
            valid_cases.append(case_name)

    print(f"Found {len(valid_cases)} valid cases\n")

    # ── Parallel processing ─────────────────────────────────────────────
    success = []
    args_list = [(cases_dir, c) for c in valid_cases]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_case, args): args[1] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                success.append(result["case"])
                print(f"  + {result['case']}")
            else:
                print(f"  x {result['case']} — {result['error']}")

    # ── Split dataset ───────────────────────────────────────────────────
    splits = split_cases(success)

    for split_name, case_list in splits.items():
        print(f"\n{split_name.upper()} ({len(case_list)})")

        for case_name in case_list:
            case_path = os.path.join(cases_dir, case_name)
            dest_case_dir = os.path.join(output_dir, split_name, case_name)
            os.makedirs(dest_case_dir, exist_ok=True)

            direction = case_name.split("_")[-1].upper()
            angle = ROTATION_MAP.get(direction, 0)

            # Buildings
            buildings_vtk = find_buildings_vtk_file(case_path)
            if buildings_vtk:
                vtp_dest = os.path.join(dest_case_dir, f"{case_name}.vtp")
                convert_buildings_vtk_to_vtp(buildings_vtk, vtp_dest, angle)

                stl_dest = os.path.join(dest_case_dir, f"{case_name}.stl")
                convert_vtp_to_stl(vtp_dest, stl_dest)

            # Domain
            vtk_file = find_vtk_file(case_path, case_name)
            if vtk_file:
                vtu_dest = os.path.join(dest_case_dir, f"{case_name}.vtu")
                convert_vtk_to_vtu(vtk_file, vtu_dest, angle)

    total_elapsed = time.perf_counter() - total_start

    print("\n============================================")
    print(" DONE")
    print("============================================")
    print(f"Processed cases: {len(success)}")
    print(f"Output: {output_dir}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
