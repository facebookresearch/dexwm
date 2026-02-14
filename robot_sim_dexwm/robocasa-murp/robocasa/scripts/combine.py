import h5py
import sys
import os
import numpy as np


def combine_demos(hdf5_paths, output_path):
    demo_counter = 0
    with h5py.File(output_path, "w") as fout:
        fout.create_group("data")

        for path in hdf5_paths:
            with h5py.File(path, "r") as fin:
                for demo_key in fin["data"].keys():
                    new_key = f"demo_{demo_counter}"
                    fin.copy(f"data/{demo_key}", fout["data"], name=new_key)
                    demo_counter += 1

                # Copy attrs only once (from first file)
                if demo_counter == len(fin["data"].keys()):
                    for attr_key in fin["data"].attrs:
                        fout["data"].attrs[attr_key] = fin["data"].attrs[attr_key]

    print(f"✅ Combined {demo_counter} demos into {output_path}")


if __name__ == "__main__":
    # Usage: python combine.py <folder_path> <output_path>
    folder_path, output_file = sys.argv[1], sys.argv[2]

    input_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.startswith("demo_") and f.endswith(".hdf5")
    ]
    input_files.sort()  # Optional: to get consistent ordering

    combine_demos(input_files, output_file)
