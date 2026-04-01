import pandas as pd
import random

for split, path in surface_paths.items():
    files = list(Path(path).glob("*.npy"))

    if not files:
        print(f"\nNo processed files found for {split} split")
        continue

    rows = []
    for f in random.sample(files, min(4, len(files))):
        try:
            d = np.load(f, allow_pickle=True).item()
            row = {
                "file": f.stem,
                "velocity_x (m/s)": d["global_params_values"][1],
                "velocity_y (m/s)": d["global_params_values"][2],
                "density (kg/m³)": d["global_params_values"][0],
                "surface_points": d["surface_mesh_centers"].shape[0],
                "stl_faces": d["stl_faces"].shape[0] // 3,
                "volume_points": d["volume_mesh_centers"].shape[0],
            }

            # Surface fields
            surface_fields = d["surface_fields"]
            surface_names = ["pressure", "U_x", "U_y", "U_z"]
            for j, name in enumerate(surface_names[:surface_fields.shape[1] if surface_fields.ndim > 1 else 1]):
                col = surface_fields[:, j] if surface_fields.ndim > 1 else surface_fields
                row[f"surf_{name}_min"], row[f"surf_{name}_max"] = col.min(), col.max()

            # Volume fields
            volume_fields = d["volume_fields"]
            volume_names = ["p", "U_x", "U_y", "U_z"]
            for j, name in enumerate(volume_names[:volume_fields.shape[1] if volume_fields.ndim > 1 else 1]):
                col = volume_fields[:, j] if volume_fields.ndim > 1 else volume_fields
                row[f"vol_{name}_min"], row[f"vol_{name}_max"] = col.min(), col.max()

            rows.append(row)
        except Exception as e:
            rows.append({"file": f.name, "error": str(e)})

    print(f"\n{split.upper()} Split — {len(files)} files total:")
    display(pd.DataFrame(rows))

    # Examine the data fields available
    print(f"\nAvailable fields:")
    for name in d.keys():
        val = d[name]
        if hasattr(val, "shape"):
            print(f"  {name:24s} | shape = {str(getattr(val, 'shape', ())):12s} | range=[{np.min(val):8.4f}, {np.max(val):11.4f}]")
        else:
            print(f"  {name:24s} | value = {val}")
