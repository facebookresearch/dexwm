import h5py
import os
import json
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Split large HDF5 dataset into 5 parts.")
parser.add_argument(
    "--input", type=str, required=True, help="Path to the input HDF5 file."
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Directory to save the output split files."
)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

with h5py.File(args.input, "r") as f:
    data_group = f["data"]
    demo_keys = sorted(data_group.keys()) 
    num_demos = len(demo_keys)
    demos_per_file = num_demos // 5

    # Read original attrs
    data_attrs = dict(data_group.attrs)

    for split_idx in range(5):
        start_idx = split_idx * demos_per_file
        end_idx = (split_idx + 1) * demos_per_file if split_idx < 4 else num_demos
        split_keys = demo_keys[start_idx:end_idx]

        output_file = os.path.join(args.output_dir, f"split_{split_idx+1}.hdf5")
        with h5py.File(output_file, "w") as out_f:
            out_data = out_f.create_group("data")

            # Copy attributes
            for attr_key, attr_val in data_attrs.items():
                out_data.attrs[attr_key] = attr_val

            # Update 'total' attribute to reflect actual number of demos in this file
            out_data.attrs["total"] = len(split_keys)

            for key in split_keys:
                f.copy(f"data/{key}", out_data, name=key)

                # Also copy individual demo attributes
                for attr_key in f[f"data/{key}"].attrs:
                    out_data[key].attrs[attr_key] = f[f"data/{key}"].attrs[attr_key]

        print(f"Saved {len(split_keys)} demos to {output_file}")
