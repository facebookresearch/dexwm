import json
import h5py
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

parser = argparse.ArgumentParser(
    description="Process HDF5 files and add a prompt to env_meta."
)
parser.add_argument(
    "--prompt", type=str, help="The prompt to add to the environment metadata."
)

args = parser.parse_args()

root_path = "/fsx-siro/achvysh07/projects/"
file_pattern = os.path.join(root_path, "combined_large_obj.hdf5")

h5_files = glob.glob(file_pattern)
ds_format = "robomimic"

if not h5_files:
    print("No matching files found.")
else:
    for filename in h5_files:
        print(f"Processing file: {filename}")
        layout_id_counter = Counter()
        style_id_counter = Counter()
        object_name_counter = Counter()
        object_category_counter = Counter()

        with h5py.File(filename, "r") as f:
            for demo_id in f["data"].keys():
                World_T_object = f["data"][demo_id]["datagen_info"]["object_poses"][
                    "obj"
                ][0]
                init_robot_base_pos = f["data"][demo_id]["datagen_info"]["base_pos"][0]
                init_robot_base_rot = f["data"][demo_id]["datagen_info"]["base_rot"][0]
                World_T_robot = np.eye(4)  # Start with an identity matrix
                World_T_robot[:3, :3] = init_robot_base_rot  # Set the rotation part
                World_T_robot[:3, 3] = init_robot_base_pos  # Set the translation part
                Robot_T_object = np.linalg.inv(World_T_robot) @ World_T_object
                diff_poses = Robot_T_object[:, 3:]
                ##############Layout_ID##############
                ep_meta = json.loads(f["data"][demo_id].attrs["ep_meta"])
                layout_id = ep_meta["layout_id"]
                style_id = ep_meta["style_id"]
                object_name = ep_meta["object_cfgs"][0]["info"]["mjcf_path"].split("/")[
                    -2
                ]
                object_category = ep_meta["object_cfgs"][0]["info"]["mjcf_path"].split(
                    "/"
                )[-3]
                layout_id_counter[layout_id] += 1
                style_id_counter[style_id] += 1
                object_name_counter[object_name] += 1
                object_category_counter[object_category] += 1

                ############# Scatter plot #############
                plt.title("Object Poses")
                plt.xlabel("X-axis of Robot frame")
                plt.ylabel("Y-axis of Robot frame")
                # plt.legend()
                plt.scatter(diff_poses[0], diff_poses[1], label=f"Object {demo_id}")
                plt.text(
                    diff_poses[0],
                    diff_poses[1],
                    demo_id,
                    fontsize=5,
                    ha="right",
                    va="bottom",
                )
                plt.savefig("large_object_poses.png", dpi=300)

            if ds_format == "robomimic":
                env_meta = json.loads(f["data"].attrs["env_args"])

                env_meta["prompt"] = args.prompt

                print(f"Updated env_meta for {filename}: {env_meta}")
            else:
                raise ValueError("Unsupported dataset format")

            # Plot histograms
            plt.figure(figsize=(20, 16))

            # Layout ID histogram
            plt.subplot(4, 1, 1)
            plt.bar(layout_id_counter.keys(), layout_id_counter.values(), color="blue")
            plt.title("Layout ID Histogram")
            plt.xlabel("Layout ID")
            plt.ylabel("Occurrences")

            # Style ID histogram
            plt.subplot(4, 1, 2)
            plt.bar(style_id_counter.keys(), style_id_counter.values(), color="green")
            plt.title("Style ID Histogram")
            plt.xlabel("Style ID")
            plt.ylabel("Occurrences")

            # Object Name histogram
            plt.subplot(4, 1, 3)
            plt.bar(
                object_name_counter.keys(), object_name_counter.values(), color="orange"
            )
            plt.title("Object Name Histogram")
            plt.xlabel("Object Name",fontsize=1.0)
            plt.ylabel("Occurrences",fontsize=1.0)
            plt.xticks(rotation=45)            

            plt.subplot(4, 1, 4)
            plt.bar(
                object_category_counter.keys(),
                object_category_counter.values(),
                color="orange",
            )
            plt.title("Object Categroy Histogram")
            plt.xlabel("Object Categroy")
            plt.ylabel("Occurrences")
            plt.xticks(rotation=45)

            plt.subplots_adjust(hspace=0.4)
            plt.savefig("large_object_stats.png", dpi=300)