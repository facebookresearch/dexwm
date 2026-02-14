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

# root_path = "./robocasa_data/teleop_data/carton_data"
# file_pattern = os.path.join(root_path, "demo_new.hdf5")

file_pattern = args.path
h5_files = glob.glob(file_pattern)
ds_format = "robomimic"

if not h5_files:
    print("No matching files found.")
else:
    for filename in h5_files:
        print(f"Processing file: {filename}")

        with h5py.File(filename, "r") as f:
            if ds_format == "robomimic":
                env_meta = json.loads(f["data"].attrs["env_args"])

                env_meta["prompt"] = args.prompt

                print(f"Updated env_meta for {filename}: {env_meta}")
            else:
                raise ValueError("Unsupported dataset format")
