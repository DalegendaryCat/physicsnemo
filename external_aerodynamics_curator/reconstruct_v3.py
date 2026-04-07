#!/usr/bin/env python3
# reconstruct_cases_vtk_rotation.py
#
# Reconstructs OpenFOAM cases, converts to VTK, and applies rotation ONLY
# during VTK → VTU/VTP/STL conversion. Original cases are never modified.

import os
import shutil
import subprocess
import sys
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import vtk
from physicsnemo.models.domino.utils.vtk_file_utils import write_to_vtp, write_to_vtu

# ─── Config ──────────────────────────────────────────────────────────────────

TIME = "1250"
RECONSTRUCT_OPTS = ["-time", TIME]
VTK_OPTS = ["-time", TIME, "-fields", "(U p nut)"]
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

# FOR OPENFOAM 8/13 file structure

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

# FOR OPENFOAM v2412 file structure
def find_internal_vtu(case_path, case_name):
    vtk_dir = os.path.join(case_path, "VTK")
    if not os.path.isdir(vtk_dir):
        return None
    for f in os.listdir(vtk_dir):
        if f == f"{case_name}_1":
            internal = os.path.join(vtk_dir, f, "internal.vtu")
            if os.path.exists(internal):
                return internal
    return None

def find_buildings_vtp(case_path, case_name):
    vtk_dir = os.path.join(case_path, "VTK")
    if not os.path.isdir(vtk_dir):
        return None
    for f in os.listdir(vtk_dir):
        if f == f"{case_name}_1":
            buildings = os.path.join(vtk_dir, f, "boundary", "buildings.vtp")
            if os.path.exists(buildings):
                return buildings

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
    if vtk_filename[-1] == "k":
        reader = vtk.vtkUnstructuredGridReader()
    elif vtk_filename[-1] == "u":
        reader = vtk.vtkXMLUnstructuredGridReader()

    reader.SetFileName(vtk_filename)
    reader.Update()

    data = reader.GetOutput()
    if not data:
        print(f"[ERROR] Failed to read {vtk_filename}")
        return

    rotated = apply_rotation(data, angle_deg)
    write_to_vtu(rotated, vtu_filename)

def convert_buildings_vtk_to_vtp(buildings_vtk: str, vtp_filename: str, angle_deg: float) -> None:
    if buildings_vtk[-1] == "k":
        reader = vtk.vtkPolyDataReader()
    elif buildings_vtk[-1] == "p":
        reader = vtk.vtkXMLPolyDataReader()

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

    # ── Pre-check: rename existing TIME folder if it exists ──────────────────
    time_folder = os.path.join(case_path, TIME)
    time_folder_old = os.path.join(case_path, f"{TIME}.old")
    if os.path.isdir(time_folder):
        if os.path.isdir(time_folder_old):
            shutil.rmtree(time_folder_old)   # remove old backup first if it exists
        shutil.move(time_folder, time_folder_old)
        print(f"  [pre-check] {case_name}/{TIME} → {TIME}.old")

    # reconstruct everything
    log_path = os.path.join(case_path, "reconstructPar.log")
    cmd = ["reconstructPar"] + RECONSTRUCT_OPTS + ["-case", case_path]

    if not run_command(cmd, log_path):
        status["error"] = "reconstructPar failed"
        return status

    # run foamToVTK
    vtk_log_path = os.path.join(case_path, "foamToVTK.log")
    vtk_cmd = ["foamToVTK"] + VTK_OPTS + ["-case", case_path]

    if not run_command(vtk_cmd, vtk_log_path):
        status["error"] = "foamToVTK failed"
        return status

    status["success"] = True
    return status


# ─── Per-case conversion (Phase 2) ───────────────────────────────────────────

def convert_case(args):
    cases_dir, case_name, split_name, output_dir = args
    case_start = time.perf_counter()
    case_path = os.path.join(cases_dir, case_name)
    dest_case_dir = os.path.join(output_dir, split_name, case_name)
    os.makedirs(dest_case_dir, exist_ok=True)

    status = {"case": case_name, "split": split_name, "success": False, "error": None}

    direction = case_name.split("_")[-1].upper()
    angle = ROTATION_MAP.get(direction, 0)

    try:
        # 1. Processing Buildings Surface
        buildings_vtk = find_buildings_vtp(case_path, case_name)
        if buildings_vtk:
            vtp_dest = os.path.join(dest_case_dir, f"{case_name}.vtp")
            convert_buildings_vtk_to_vtp(buildings_vtk, vtp_dest, angle)

            stl_dest = os.path.join(dest_case_dir, f"{case_name}.stl")
            convert_vtp_to_stl(vtp_dest, stl_dest)

        # 2. Processing Domain Volume
        vtk_file = find_internal_vtu(case_path, case_name)
        if vtk_file:
            vtu_dest = os.path.join(dest_case_dir, f"{case_name}.vtu")
            convert_vtk_to_vtu(vtk_file, vtu_dest, angle)

        status["success"] = True

    except Exception as e:
        status["error"] = str(e)

    case_elapsed = time.perf_counter() - case_start
    status["elapsed"] = case_elapsed
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

    if len(sys.argv) < 2:
        cases_dir = DEFAULT_CASES_DIR
        print(f"No cases dir provided — using default: {cases_dir}")
    else:
        cases_dir = os.path.abspath(sys.argv[1])

    if len(sys.argv) < 3:
        output_dir = OUTPUT_DIR
        print(f"No output dir provided — using default: {output_dir}")
    else:
        output_dir = os.path.abspath(sys.argv[2])

    total_start = time.perf_counter()

    print("============================================")
    print(" OpenFOAM → VTK → Rotated Dataset Pipeline")
    print(f" Cases dir:    {cases_dir}")
    print(f" Output dir:   {output_dir}")
    print(f" Time:         {TIME}")
    print(f" Workers:      {MAX_WORKERS}")
    print(f" Required fields: {REQUIRED_FIELDS}")
    print(f" Split:        train={int(SPLIT['train']*100)}% / test={int(SPLIT['test']*100)}% / validation={int(SPLIT['validation']*100)}%")
    print("============================================\n")

    if not os.path.isdir(cases_dir):
        print(f"Error: Cases directory '{cases_dir}' not found.")
        sys.exit(1)

    # Create output directories
    for split in SPLIT:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Find valid cases
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
            print(f"Skipping '{case_name}' — {reason}")
            skipped.append(case_name)
        else:
            valid_cases.append(case_name)

    print(f"\nFound {len(valid_cases)} valid cases — processing with {MAX_WORKERS} workers...\n")

    # ── Phase 1: Parallel reconstruction + foamToVTK ───────────────────
    recon_start = time.perf_counter()
    success = []
    failed = []

    args_list = [(cases_dir, c) for c in valid_cases]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_case, args): args[1] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            case_name = result["case"]
            if result["success"]:
                tag = "(recon skipped)" if result["skipped_recon"] else ""
                print(f"  + {case_name} {tag}")
                success.append(case_name)
            else:
                print(f"  x {case_name} — {result['error']}")
                failed.append(case_name)

    if not success:
        print("\nNo cases succeeded — skipping dataset organisation.")
        sys.exit(1)

    recon_elapsed = time.perf_counter() - recon_start
    print(f"\n  Reconstruction completed in {recon_elapsed:.1f}s ({recon_elapsed/60:.1f} min)")

    # ── Phase 2: Parallel VTK → VTU/VTP/STL conversion ─────────────────
    print("\n============================================")
    print(" Organising into train/test/validation")
    print("============================================\n")

    organise_start = time.perf_counter()
    splits = split_cases(success)

    convert_args = [
        (cases_dir, case_name, split_name, output_dir)
        for split_name, case_list in splits.items()
        for case_name in case_list
    ]

    convert_success = []
    convert_failed = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(convert_case, args): args[1] for args in convert_args}
        for future in as_completed(futures):
            result = future.result()
            case_name = result["case"]
            split_name = result["split"]
            elapsed = result.get("elapsed", 0)
            if result["success"]:
                print(f"  → [{split_name}] {case_name} done in {elapsed:.1f}s")
                convert_success.append(case_name)
            else:
                print(f"  x [{split_name}] {case_name} — {result['error']}")
                convert_failed.append(case_name)

    organise_elapsed = time.perf_counter() - organise_start
    total_elapsed = time.perf_counter() - total_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print("============================================")
    print(" Summary")
    print("============================================")
    print(f" Reconstructed:   {len(success)}")
    for c in success:
        print(f"   + {c}")
    print(f" Recon failed:    {len(failed)}")
    for c in failed:
        print(f"   x {c}")
    print(f" Converted:       {len(convert_success)}")
    print(f" Convert failed:  {len(convert_failed)}")
    for c in convert_failed:
        print(f"   x {c}")
    print(f" Skipped:         {len(skipped)}")
    for c in skipped:
        print(f"   - {c}")
    print("============================================")
    print(f"\n Timing breakdown:")
    print(f"   Reconstruction + foamToVTK : {recon_elapsed:.1f}s ({recon_elapsed/60:.1f} min)")
    print(f"   Organise + convert         : {organise_elapsed:.1f}s ({organise_elapsed/60:.1f} min)")
    print(f"   Total                      : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("============================================")
    print(f"\nOutput structure per case:")
    print(f"  {output_dir}/<split>/<case_name>/")
    print(f"    <case_name>.stl   (from VTK/buildings/<>.vtk via VTP)")
    print(f"    <case_name>.vtp   (from VTK/buildings/<>.vtk)")
    print(f"    <case_name>.vtu   (from VTK/<case_name>.vtk)")
    print(f"\nDataset saved to: {output_dir}")


if __name__ == "__main__":
    main()