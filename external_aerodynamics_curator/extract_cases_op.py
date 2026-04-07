#!/usr/bin/env python3
# 03_extract_and_organise.py
#
# Phase 3: Read VTK/VTU/VTP files produced by foamToVTK, apply rotation to
# both geometry AND vector fields, then write out the train/test/validation
# dataset as VTU + VTP + STL files.
# Run this after 02_foam_to_vtk.py has completed successfully.

import os
import sys
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from physicsnemo.models.domino.utils.vtk_file_utils import write_to_vtp, write_to_vtu

# ─── Config ──────────────────────────────────────────────────────────────────

TIME = "1250"
OUTPUT_DIR = "/home/nguye/physicsnemo/Dataset/hdb_input_rotated"
SPLIT = {"train": 0.8, "test": 0.1, "validation": 0.1}
RANDOM_SEED = 42
MAX_WORKERS = 4

ROTATION_MAP = {
    "N": 0,
    "S": 180,
    "E": 90,
    "W": 270,
}

DEFAULT_CASES_DIR = "/home/nguye/physicsnemo/simulation_data/simulation_data"

# ─── VTK file finders ────────────────────────────────────────────────────────

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
    return None


# ─── Rotation ────────────────────────────────────────────────────────────────

def apply_rotation(data, angle_deg):
    # 1. Rotate geometry (point coordinates)
    transform = vtk.vtkTransform()
    transform.RotateZ(angle_deg)

    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(data)
    transform_filter.Update()

    output = transform_filter.GetOutput()

    # 2. Build 3x3 rotation matrix around Z
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ], dtype=np.float64)

    # 3. Rotate all 3-component point data arrays (e.g. U velocity)
    pd = output.GetPointData()
    for i in range(pd.GetNumberOfArrays()):
        arr = pd.GetArray(i)
        if arr is None:
            continue
        if arr.GetNumberOfComponents() == 3:
            np_arr = vtk_to_numpy(arr).copy()
            rotated = (R @ np_arr.T).T
            new_vtk = numpy_to_vtk(rotated, deep=True)
            new_vtk.SetName(arr.GetName())
            pd.RemoveArray(arr.GetName())
            pd.AddArray(new_vtk)

    # 4. Rotate all 3-component cell data arrays
    cd = output.GetCellData()
    for i in range(cd.GetNumberOfArrays()):
        arr = cd.GetArray(i)
        if arr is None:
            continue
        if arr.GetNumberOfComponents() == 3:
            np_arr = vtk_to_numpy(arr).copy()
            rotated = (R @ np_arr.T).T
            new_vtk = numpy_to_vtk(rotated, deep=True)
            new_vtk.SetName(arr.GetName())
            cd.RemoveArray(arr.GetName())
            cd.AddArray(new_vtk)

    return output


# ─── Conversion helpers ───────────────────────────────────────────────────────

def convert_vtk_to_vtu(vtk_filename: str, vtu_filename: str, angle_deg: float) -> None:
    if vtk_filename.endswith(".vtk"):
        reader = vtk.vtkUnstructuredGridReader()
    elif vtk_filename.endswith(".vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        print(f"[ERROR] Unknown extension for {vtk_filename}")
        return

    reader.SetFileName(vtk_filename)
    reader.Update()

    data = reader.GetOutput()
    if not data:
        print(f"[ERROR] Failed to read {vtk_filename}")
        return

    rotated = apply_rotation(data, angle_deg)
    write_to_vtu(rotated, vtu_filename)


def convert_buildings_to_vtp(buildings_src: str, vtp_filename: str, angle_deg: float) -> None:
    if buildings_src.endswith(".vtk"):
        reader = vtk.vtkPolyDataReader()
    elif buildings_src.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        print(f"[ERROR] Unknown extension for {buildings_src}")
        return

    reader.SetFileName(buildings_src)
    reader.Update()

    data = reader.GetOutput()
    if not data or data.GetNumberOfPoints() == 0:
        print(f"[ERROR] Failed to read {buildings_src}")
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


# ─── Per-case conversion ──────────────────────────────────────────────────────

def convert_case(args):
    cases_dir, case_name, split_name, output_dir = args
    case_start = time.perf_counter()
    case_path = os.path.join(cases_dir, case_name)
    dest_case_dir = os.path.join(output_dir, split_name, case_name)
    os.makedirs(dest_case_dir, exist_ok=True)

    status = {"case": case_name, "split": split_name, "success": False, "error": None, "elapsed": 0}

    direction = case_name.split("_")[-1].upper()
    angle = ROTATION_MAP.get(direction, 0)

    try:
        # 1. Buildings surface → VTP + STL
        buildings_src = find_buildings_vtp(case_path, case_name)
        if not buildings_src:
            buildings_src = find_buildings_vtk_file(case_path)

        if buildings_src:
            vtp_dest = os.path.join(dest_case_dir, f"{case_name}.vtp")
            convert_buildings_to_vtp(buildings_src, vtp_dest, angle)

            stl_dest = os.path.join(dest_case_dir, f"{case_name}.stl")
            convert_vtp_to_stl(vtp_dest, stl_dest)
        else:
            print(f"  [WARN] {case_name}: no buildings VTP/VTK found")

        # 2. Domain volume → VTU
        vtk_file = find_internal_vtu(case_path, case_name)
        if not vtk_file:
            vtk_file = find_vtk_file(case_path, case_name)

        if vtk_file:
            vtu_dest = os.path.join(dest_case_dir, f"{case_name}.vtu")
            convert_vtk_to_vtu(vtk_file, vtu_dest, angle)
        else:
            print(f"  [WARN] {case_name}: no internal VTU/VTK found")

        status["success"] = True

    except Exception as e:
        status["error"] = str(e)

    status["elapsed"] = time.perf_counter() - case_start
    return status


# ─── Split helper ─────────────────────────────────────────────────────────────

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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
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

    if not os.path.isdir(cases_dir):
        print(f"Error: Cases directory '{cases_dir}' not found.")
        sys.exit(1)

    print("============================================")
    print(" Phase 3: Extract, Rotate & Organise")
    print(f" Cases dir:  {cases_dir}")
    print(f" Output dir: {output_dir}")
    print(f" Workers:    {MAX_WORKERS}")
    print(f" Split:      train={int(SPLIT['train']*100)}% / test={int(SPLIT['test']*100)}% / val={int(SPLIT['validation']*100)}%")
    print("============================================\n")

    for split in SPLIT:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Find cases that have a VTK folder (i.e. foamToVTK has been run)
    all_cases = [
        d for d in os.listdir(cases_dir)
        if os.path.isdir(os.path.join(cases_dir, d))
    ]

    valid_cases = []
    skipped = []
    for case_name in all_cases:
        case_path = os.path.join(cases_dir, case_name)
        vtk_dir = os.path.join(case_path, "VTK")
        if not os.path.isdir(vtk_dir):
            print(f"  Skipping '{case_name}' — no VTK/ folder (run 02_foam_to_vtk.py first?)")
            skipped.append(case_name)
        else:
            valid_cases.append(case_name)

    print(f"Found {len(valid_cases)} cases with VTK output — organising with {MAX_WORKERS} workers...\n")

    splits = split_cases(valid_cases)

    convert_args = [
        (cases_dir, case_name, split_name, output_dir)
        for split_name, case_list in splits.items()
        for case_name in case_list
    ]

    total_start = time.perf_counter()
    success = []
    failed = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(convert_case, args): args[1] for args in convert_args}
        for future in as_completed(futures):
            result = future.result()
            case_name = result["case"]
            split_name = result["split"]
            elapsed = result["elapsed"]
            if result["success"]:
                print(f"  → [{split_name}] {case_name}  ({elapsed:.1f}s)")
                success.append(case_name)
            else:
                print(f"  x [{split_name}] {case_name} — {result['error']}")
                failed.append(case_name)

    total_elapsed = time.perf_counter() - total_start

    print("\n============================================")
    print(" Summary")
    print("============================================")
    print(f" Converted:  {len(success)}")
    print(f" Failed:     {len(failed)}")
    for c in failed:
        print(f"   x {c}")
    print(f" Skipped:    {len(skipped)}")
    for c in skipped:
        print(f"   - {c}")
    print(f"\n Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("============================================")
    print(f"\nOutput structure per case:")
    print(f"  {output_dir}/<split>/<case_name>/")
    print(f"    <case_name>.vtu")
    print(f"    <case_name>.vtp")
    print(f"    <case_name>.stl")
    print(f"\nDataset saved to: {output_dir}")


if __name__ == "__main__":
    main()