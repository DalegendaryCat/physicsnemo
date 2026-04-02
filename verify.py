"""
Verify whether pressure at offset shell differs from pressure at building surface.
Usage: python verify.py 0.5 2 3
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import numpy as np
import trimesh
import pyvista as pv
from pathlib import Path
from scipy.spatial import cKDTree
from fluidfoam import readmesh, readfield

# ============================================================
# CONFIG
# ============================================================
CASE_PATH = Path(os.environ["CASE_PATH"])
TIMESTEP = "1250"
SPACING = 2.0
IDW_K = 4
FILTER_MARGIN = 40.0

OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])

# ============================================================
# FUNCTIONS
# ============================================================
def load_stl(case_path):
    stl_path = case_path / "constant" / "triSurface" / "buildings.stl"
    mesh = trimesh.load(str(stl_path), force="mesh")
    mesh.merge_vertices()
    mesh.fix_normals()
    mesh.fill_holes()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    return mesh


def sdf_offset(stl_mesh, points, normals, distance):
    overshoot = distance * 1.3
    initial = points + normals * overshoot
    closest, _, tri_ids = trimesh.proximity.closest_point(stl_mesh, initial)

    dirs = initial - closest
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    degen = (norms < 1e-8).flatten()
    if degen.any():
        dirs[degen] = stl_mesh.face_normals[tri_ids[degen]]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    dirs /= norms

    return closest + dirs * distance


def read_cfd(case_path, stl_mesh):
    x, y, z = readmesh(str(case_path), structured=False)
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    n_total = len(x)

    bb_min = stl_mesh.bounds[0] - FILTER_MARGIN
    bb_max = stl_mesh.bounds[1] + FILTER_MARGIN
    mask = (
        (x >= bb_min[0]) & (x <= bb_max[0]) &
        (y >= bb_min[1]) & (y <= bb_max[1]) &
        (z >= bb_min[2]) & (z <= bb_max[2])
    )

    centers = np.column_stack([x[mask], y[mask], z[mask]])
    return centers, mask, n_total


def read_pressure(case_path, mask, n_total):
    vals = np.asarray(readfield(str(case_path), TIMESTEP, "p")).reshape(-1)
    if vals.size == 1:
        return np.full(int(mask.sum()), float(vals[0]))
    if len(vals) > n_total:
        vals = vals[:n_total]
    return vals[mask]


def idw_interpolate(tree, p_values, target_points, k=4):
    dists, idxs = tree.query(target_points, k=k, workers=-1)
    if k == 1:
        dists, idxs = dists[:, None], idxs[:, None]
    dists = np.maximum(dists, 1e-10)
    w = 1.0 / dists
    w /= w.sum(axis=1, keepdims=True)
    return (p_values[idxs] * w).sum(axis=1)


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python verify.py <offset1> [offset2] [offset3] ...")
        print("Example: python verify.py 0.5 2 3")
        sys.exit(1)

    offsets = [float(x) for x in sys.argv[1:]]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load geometry
    stl_mesh = load_stl(CASE_PATH)
    n_target = max(int(stl_mesh.area / (SPACING ** 2)), 100)
    surface_pts, face_ids = trimesh.sample.sample_surface_even(stl_mesh, count=n_target)
    normals = stl_mesh.face_normals[face_ids]

    # Read CFD once
    centers, mask, n_total = read_cfd(CASE_PATH, stl_mesh)
    p_filtered = read_pressure(CASE_PATH, mask, n_total)
    tree = cKDTree(centers)

    # Interpolate surface pressure once
    p_surface = idw_interpolate(tree, p_filtered, surface_pts, k=IDW_K)

    # Save building geometry once
    pv.wrap(stl_mesh).save(str(OUTPUT_DIR / "building.vtp"))

    # Loop over offsets
    for offset in offsets:
        tag = f"{offset}m"

        shell_pts = sdf_offset(stl_mesh, surface_pts, normals, offset)
        p_shell = idw_interpolate(tree, p_filtered, shell_pts, k=IDW_K)
        delta_p = p_shell - p_surface

        # CSV
        results = np.column_stack([p_surface, p_shell, delta_p])
        np.savetxt(OUTPUT_DIR / f"pressure_comparison_{tag}.csv", results,
                   delimiter=",", header="p_surface,p_shell,delta_p",
                   comments="", fmt="%.8f")

        # Shell points VTP
        shell_pv = pv.PolyData(shell_pts)
        shell_pv["p_shell"] = p_shell
        shell_pv.save(str(OUTPUT_DIR / f"shell_points_{tag}.vtp"))

        # Surface comparison VTP
        surf_pv = pv.PolyData(surface_pts)
        surf_pv["p_surface"] = p_surface
        surf_pv["p_shell"] = p_shell
        surf_pv["delta_p"] = delta_p
        surf_pv.save(str(OUTPUT_DIR / f"surface_comparison_{tag}.vtp"))

        print(f"  [{tag}] delta_p: mean={delta_p.mean():.4f}, std={delta_p.std():.4f}, max={np.abs(delta_p).max():.4f}")

    print(f"Done. Output -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
