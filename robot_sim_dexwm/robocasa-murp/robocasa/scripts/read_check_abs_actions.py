import h5py
import os
import json
import glob
import argparse

parser = argparse.ArgumentParser(
    description="Process HDF5 files and add a prompt to env_meta."
)
parser.add_argument(
    "--prompt", type=str, help="The prompt to add to the environment metadata."
)
parser.add_argument(
    "--path", type=str, required=True, help="Path to directory or file pattern to search HDF5 files."
)
args = parser.parse_args()

# root_path = "./"
# file_pattern = os.path.join(root_path, "combined_large_actions_absolute.hdf5")
file_pattern=args.path
h5_files = glob.glob(file_pattern)
ds_format = "robomimic"

if not h5_files:
    print("No matching files found.")
else:
    for filename in h5_files:
        print(f"Processing file: {filename}")

        with h5py.File(filename, "r") as f:
            if ds_format == "robomimic":
                breakpoint()
                for demo_key in f["data"].keys():
                    demo_group = f["data"][demo_key]
                    if "actions_abs" not in demo_group:
                        print(f"[WARN] Demo '{demo_key}' missing 'actions_abs' dataset.")
                    else:
                        actions_abs_data = demo_group["actions_abs"][:]
                        if actions_abs_data.size == 0:
                            print(f"[WARN] Demo '{demo_key}' has empty 'actions_abs' dataset.")
            else:
                raise ValueError("Unsupported dataset format")
