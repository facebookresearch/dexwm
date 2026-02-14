import h5py
import os
import shutil

# --- Configuration ---
input_path = "./projects/PnPSucc/_0.h5"
output_path = "./projects/PnPSucc/_0_cleaned.h5"
demos_to_delete = {4, 253, 258, 153, 115, 281, 286, 565, 553, 363, 274,215,79,555}
demos_to_delete = {f"demo_{i}" for i in demos_to_delete}

# --- Processing ---
with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
    # Copy top-level attributes
    for attr_key in f_in.attrs:
        f_out.attrs[attr_key] = f_in.attrs[attr_key]

    # Create the 'data' group
    f_out.create_group("data")

    # Copy 'data' attributes (like env_args, etc.)
    for attr_key in f_in["data"].attrs:
        f_out["data"].attrs[attr_key] = f_in["data"].attrs[attr_key]

    # Copy only the demos NOT in the deletion list
    for demo_key in f_in["data"]:
        if demo_key in demos_to_delete:
            print(f"Skipping (deleting): {demo_key}")
            continue
        f_in["data"].copy(demo_key, f_out["data"])
        print(f"Copied: {demo_key}")

print(f"\n✅ Cleaned file saved as: {output_path}")

