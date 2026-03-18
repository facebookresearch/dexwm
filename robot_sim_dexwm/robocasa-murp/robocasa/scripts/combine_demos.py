import h5py
import sys
import os
import numpy as np
import json
import argparse
import h5py
import h5py
import h5py
import numpy as np
import os
import multiprocessing
from functools import partial
from robocasa.scripts.qa import is_corrupted_frame

TARGET_CAM_NAME = ["robot0_robotview_image"]


def process_file(path):
    valid_demos = []
    skipped_demos = []

    with h5py.File(path, "r") as f:
        for demo_key in f["data"].keys():
            group = f["data"][demo_key]

            obj_on_counter = False
            image_health = True

            try:
                # Check the final location of the object's height, make sure it does not
                # drop below 0.5m
                obj_on_counter = (
                    group["datagen_info"]["object_poses"]["obj"][:][-1][:, 3:][2, 0]
                    > 0.5
                )

                # Check if the target camera name is corrupted in any frame
                for target_cam_name in TARGET_CAM_NAME:
                    for frame in f[f"data/{demo_key}"]["obs"][target_cam_name][()]:
                        expected_shape = frame.shape
                        error = is_corrupted_frame(frame, expected_shape=expected_shape)
                        if len(error) != 0:
                            image_health = False
                            break
            except KeyError:
                skipped_demos.append((path, demo_key))
                continue

            if obj_on_counter and image_health:
                valid_demos.append((path, demo_key))
            else:
                print(
                    f"Demo {path} is invalid due to obj_on_counter: {obj_on_counter}, image_health: {image_health}"
                )
                skipped_demos.append((path, demo_key))

    return valid_demos, skipped_demos


################################### FOR NORMAL COMBINE USING MP ###############################################
def combine_demos_parallel(hdf5_paths, output_path, log_file_path, num_workers=64):
    print(f"🔍 Scanning {len(hdf5_paths)} files with {num_workers} workers...")

    # Step 1: Parallel filter valid/invalid demos
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_file, hdf5_paths)

    all_valid_demos = []
    all_skipped_demos = []

    for valid_demos, skipped_demos in results:
        all_valid_demos.extend(valid_demos)
        all_skipped_demos.extend(skipped_demos)

    # Step 2: Write valid demos to output file and log
    demo_mapping = {}
    demo_counter = 0

    with open(log_file_path, "w") as log_file:
        with h5py.File(output_path, "w") as fout:
            fout.create_group("data")

            for path, demo_key in all_valid_demos:
                with h5py.File(path, "r") as fin:
                    new_key = f"demo_{demo_counter}"
                    fin.copy(f"data/{demo_key}", fout["data"], name=new_key)

                    demo_mapping[new_key] = path
                    log_file.write(f"[INCLUDED] Demo Number: {new_key}, Path: {path}\n")
                    demo_counter += 1

                    # Copy attributes only once
                    if demo_counter == 1:
                        for attr_key in fin["data"].attrs:
                            fout["data"].attrs[attr_key] = fin["data"].attrs[attr_key]

            for path, new_key in all_skipped_demos:
                log_file.write(
                    f"[SKIPPED] Demo Key: {new_key}, Path: {path} (No grasp signal)\n"
                )

            fout["data"].attrs["demo_mapping"] = str(demo_mapping)

    print(f"✅ Combined {demo_counter} valid demos into {output_path}")
    print(f"📝 Log written to {log_file_path}")
    print(f"⚠️ Skipped demos count: {len(all_skipped_demos)}")


################################### FOR SPLIT AND COMBINE ###############################################
def combine_demos_parallel_split(
    hdf5_paths, output_path_base, log_file_base, num_workers=16, num_splits=5
):
    print(f"🔍 Scanning {len(hdf5_paths)} files with {num_workers} workers...")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_file, hdf5_paths)

    all_valid_demos = []
    all_skipped_demos = []

    for valid_demos, skipped_demos in results:
        all_valid_demos.extend(valid_demos)
        all_skipped_demos.extend(skipped_demos)

    total_valid = len(all_valid_demos)
    chunk_size = (total_valid + num_splits - 1) // num_splits

    global_demo_counter = 0

    for split_idx in range(num_splits):
        start_idx = split_idx * chunk_size
        end_idx = min((split_idx + 1) * chunk_size, total_valid)
        valid_chunk = all_valid_demos[start_idx:end_idx]

        output_path = f"{output_path_base}_{split_idx}.hdf5"
        log_file_path = f"{log_file_base}_{split_idx}.txt"

        demo_mapping = {}

        # Gather all input files that contributed to this chunk
        files_in_chunk = set([path for path, _ in valid_chunk])

        with open(log_file_path, "w") as log_file, h5py.File(output_path, "w") as fout:
            fout.create_group("data")

            for path, demo_key in valid_chunk:
                with h5py.File(path, "r") as fin:
                    new_key = f"demo_{global_demo_counter}"
                    fin.copy(f"data/{demo_key}", fout["data"], name=new_key)

                    demo_mapping[new_key] = path
                    log_file.write(f"[INCLUDED] Demo Number: {new_key}, Path: {path}\n")

                    if global_demo_counter == start_idx:
                        for attr_key in fin["data"].attrs:
                            fout["data"].attrs[attr_key] = fin["data"].attrs[attr_key]

                    global_demo_counter += 1

            # Write skipped demos whose path is in this split's files_in_chunk
            for path, demo_key in all_skipped_demos:
                if path in files_in_chunk:
                    log_file.write(
                        f"[SKIPPED] Demo Key: {demo_key}, Path: {path} (No grasp signal)\n"
                    )

            fout["data"].attrs["demo_mapping"] = str(demo_mapping)

        print(
            f"✅ Split {split_idx} done: saved {len(valid_chunk)} demos and related skipped demos"
        )

    print(f"⚠️ Total skipped demos count: {len(all_skipped_demos)}")


################################### FOR NORMAL COMBINE ###############################################


def combine_demos(hdf5_paths, output_path, log_file_path):
    demo_counter = 0
    demo_mapping = {}  # To store mapping of demo numbers to respective paths

    with open(log_file_path, "w") as log_file:  # Open log file for writing
        with h5py.File(output_path, "w") as fout:
            fout.create_group("data")

            for path in hdf5_paths:
                with h5py.File(path, "r") as fin:
                    for demo_key in fin["data"].keys():
                        new_key = f"demo_{demo_counter}"
                        fin.copy(f"data/{demo_key}", fout["data"], name=new_key)
                        # Record the demo mapping
                        demo_mapping[new_key] = path

                        # Write demo number and path mapping to the log file
                        log_file.write(f"Demo Number: {new_key}, Path: {path}\n")

                        demo_counter += 1

                    # Copy attrs only once (from first file)
                    if demo_counter == len(fin["data"].keys()):
                        for attr_key in fin["data"].attrs:
                            fout["data"].attrs[attr_key] = fin["data"].attrs[attr_key]

        # Optionally save the demo path mapping as an attribute in the output HDF5 file
        fout["data"].attrs["demo_mapping"] = str(demo_mapping)

    print(f"✅ Combined {demo_counter} demos into {output_path}")
    print(f"✅ Log saved to {log_file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine HDF5 demo files in different modes."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        help="JSON file containing list of input paths under key 'success_path'",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "parallel", "parallel_split"],
        default="normal",
        help="Choose combining mode: normal, parallel, parallel_split",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="combine_log.txt",
        help="Path to log file",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for parallel modes",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=5,
        help="Number of splits for 'parallel_split' mode",
    )
    args = parser.parse_args()

    # Load input files list from JSON
    with open(args.json_file, "r") as f:
        data = json.load(f)
    input_files = data.get("success_path", [])
    input_files = [p.replace("important_stats.json", "demo.hdf5") for p in input_files]

    if args.mode == "normal":
        print("Running normal combine mode...")
        combine_demos(input_files, args.output_file, args.log_file)

    elif args.mode == "parallel":
        print(f"Running parallel combine mode with {args.num_workers} workers...")
        combine_demos_parallel(
            input_files, args.output_file, args.log_file, num_workers=args.num_workers
        )

    elif args.mode == "parallel_split":
        print(
            f"Running parallel split combine mode with {args.num_workers} workers and {args.num_splits} splits..."
        )
        combine_demos_parallel_split(
            input_files,
            args.output_file,
            args.log_file,
            num_workers=args.num_workers,
            num_splits=args.num_splits,
        )


if __name__ == "__main__":
    main()
