import argparse
import json
import os
import random
import time

import h5py
import imageio
import numpy as np
import robosuite
from termcolor import colored
import robosuite.utils.transform_utils as T
from PIL import Image, ImageDraw
import robocasa
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import re
import xml.etree.ElementTree as ET
from robosuite.utils.mjcf_utils import find_elements
from robosuite.models.robots.compositional import TMR_ROBOT

BASE_ASSET_ROOT_PATH = os.path.join(robocasa.models.assets_root, "objects")


def get_ee_T_arm_base(env, ee_T_base_pos_quat, arm="right", mode="playback"):
    """
    mode : determines which mode of operation the function is used :
    mode (str): One of the following:
            - "sim": Use live simulation data via self.sim.data during actual simulation
            - "playback": Use static / preloaded data via env.data.

    """
    if mode == "playback":
        if arm == "right":
            arm_base_T_world_pos = env.sim.data.get_site_xpos(
                f"robot0_{arm}_base_center"
            )
            arm_base_T_world_mat = env.sim.data.get_site_xmat(
                f"robot0_{arm}_base_center"
            )
        if arm == "left":
            arm_base_T_world_pos = env.sim.data.get_site_xpos(f"robot0_{arm}_center")
            arm_base_T_world_mat = env.sim.data.get_site_xmat(f"robot0_{arm}_center")
        base_T_world_pos = env.sim.data.get_body_xpos("mobilebase0_base")
        base_T_world_mat = env.sim.data.get_body_xmat("mobilebase0_base")
    if mode == "sim":
        if arm == "right":
            arm_base_T_world_pos = env.data.get_site_xpos(f"robot0_{arm}_base_center")
            arm_base_T_world_mat = env.data.get_site_xmat(f"robot0_{arm}_base_center")
        if arm == "left":
            arm_base_T_world_pos = env.data.get_site_xpos(f"robot0_{arm}_center")
            arm_base_T_world_mat = env.data.get_site_xmat(f"robot0_{arm}_center")
        base_T_world_pos = env.data.get_body_xpos("mobilebase0_base")
        base_T_world_mat = env.data.get_body_xmat("mobilebase0_base")

    base_T_world = np.eye(4)
    base_T_world[:3, -1] = base_T_world_pos
    base_T_world[:3, :3] = base_T_world_mat
    ee_T_base_pos = ee_T_base_pos_quat[0:3]
    ee_T_base_mat = T.quat2mat(T.axisangle2quat(ee_T_base_pos_quat[3:]))

    ee_T_base = np.eye(4)
    ee_T_base[:3, -1] = ee_T_base_pos
    ee_T_base[:3, :3] = ee_T_base_mat

    ee_T_world = base_T_world @ ee_T_base
    # breakpoint()

    # ee_T_world = np.eye(4)
    # ee_T_world[:3, -1] = env.sim.data.get_site_xpos(f'gripper0_right_grip_site')
    # ee_T_world[:3, :3] = env.sim.data.get_site_xmat(f'gripper0_right_grip_site')

    arm_base_T_world = np.eye(4)
    arm_base_T_world[:3, -1] = arm_base_T_world_pos
    arm_base_T_world[:3, :3] = arm_base_T_world_mat

    ee_T_arm = np.linalg.inv(arm_base_T_world) @ ee_T_world

    ee_T_arm_pos = ee_T_arm[:3, -1]
    ee_T_arm_quat = T.mat2quat(ee_T_arm[:3, :3])
    ee_T_arm_axisangle = T.quat2axisangle(ee_T_arm_quat)
    # print("ee->:", np.concatenate((ee_T_arm_pos, ee_T_arm_axisangle)))
    return np.concatenate((ee_T_arm_pos, ee_T_arm_axisangle))


def load_traj_file(traj_file):
    if traj_file.endswith(".npy"):
        data = np.load(traj_file, allow_pickle=True)
    else:
        with open(traj_file, "r") as f:
            data = json.load(f)
    return data


def real2sim_pos(actions, traj_file):
    """
    This function acts as a helper to playback realworld data in robocasa, by
    overriding actions from hdf5 to the ones in npz.
    """
    if not traj_file:
        file_path = "./robocasa-murp/ep_0_robocasa_joints.npy"  # YOURPATH
    else:
        file_path = os.path.expanduser(traj_file)
    # Load data
    # with gzip.open(file_path, 'rb') as f:
    data = load_traj_file(file_path)
    data_len = len(data)
    actions[:data_len, :7] = data[:data_len, :]
    return actions, data_len


def real2sim_osc(actions, traj_file):
    if not traj_file:
        file_path = "./ep_0_robocasa_eef_fk.npy"
    else:
        file_path = os.path.expanduser(traj_file)
    # Load data
    # with gzip.open(file_path, 'rb') as f:
    data = load_traj_file(file_path)
    data_len = len(data)
    actions[:data_len, :6] = np.array(
        [
            np.concatenate(
                (
                    d["right_fr3_link8_T_right_base"][:3],  # local xyz position
                    R.from_quat(d["right_fr3_link8_T_right_base"][3:7]).as_rotvec(),
                )
            )
            for d in data
        ]
    )
    actions[:data_len, 15:31] = [d["right_hand_joint"] for d in data]
    return actions, data_len


def playback_trajectory_with_obs(
    traj_grp,
    video_writer,
    video_skip=5,
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert (
        image_names is not None
    ), "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["obs/{}".format(image_names[0] + "_image")].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k + "_image")][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break


def get_env_metadata_from_dataset(
    dataset_path, ds_format="robomimic", backup_dataset_path=None
):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        try:
            env_meta = json.loads(f["data"].attrs["env_args"])
        except:
            # This is because some of the hdf5 files miss the
            # env_args. Since all hdf5 within the same dataset use the same
            # env_args, you can just load the first split -- which has
            # the env_args.
            # just use the first one as they will be the same
            dataset_path = os.path.expanduser(backup_dataset_path)
            f = h5py.File(dataset_path, "r")
            env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta


class ObservationKeyToModalityDict(dict):
    """
    Custom dictionary class with the sole additional purpose of automatically registering new "keys" at runtime
    without breaking. This is mainly for backwards compatibility, where certain keys such as "latent", "actions", etc.
    are used automatically by certain models (e.g.: VAEs) but were never specified by the user externally in their
    config. Thus, this dictionary will automatically handle those keys by implicitly associating them with the low_dim
    modality.
    """

    def __getitem__(self, item):
        # If a key doesn't already exist, warn the user and add default mapping
        if item not in self.keys():
            print(
                f"ObservationKeyToModalityDict: {item} not found,"
                f" adding {item} to mapping with assumed low_dim modality!"
            )
            self.__setitem__(item, "low_dim")
        return super(ObservationKeyToModalityDict, self).__getitem__(item)


def path_change(xml_string):
    """
    Fix absolute file paths in the MJCF XML by replacing them with local paths
    rooted at BASE_ASSET_ROOT_PATH.
    """

    def replace_path(match):
        original_path = match.group(1)
        model_index = original_path.find("objects/")
        if model_index == -1:
            return f'file="{original_path}"'

        relative_path = original_path[model_index + len("objects/") :]
        new_path = os.path.join(BASE_ASSET_ROOT_PATH, relative_path)
        new_path = os.path.normpath(new_path)

        return f'file="{new_path}"'

    updated_xml = re.sub(r'file="([^"]+)"', replace_path, xml_string)
    return updated_xml


def update_mjcf_paths(object_cfgs):
    """
    Update mjcf_path in object_cfgs by replacing src path with target path.

    Args:
        object_cfgs (list): list of object configuration dicts containing 'info' with 'mjcf_path'.
        src (str): source path substring to replace.
        target (str): target path substring to replace with.

    Returns:
        list: Updated object_cfgs with modified mjcf_path.
    """
    for i, object_cfg in enumerate(object_cfgs):
        path = object_cfg["info"]["mjcf_path"]
        models_index = path.find("objects")
        relative_path = path[
            models_index:
        ]  # e.g. 'models/assets/objects/aigen_objs/apple/apple_5/model.xml'
        full_local_path = os.path.join(
            BASE_ASSET_ROOT_PATH, relative_path[len("objects/") :]
        )
        object_cfgs[i]["info"]["mjcf_path"] = full_local_path
    return object_cfgs


def _apply_ep_meta_and_reset(env, ep_meta):
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    env.reset()


def _prepare_xml(env, model_xml):
    robosuite_version_id = int(robosuite.__version__.split(".")[1])
    if robosuite_version_id <= 3:
        from robosuite.utils.mjcf_utils import postprocess_model_xml

        return postprocess_model_xml(model_xml)
    else:
        return env.edit_model_xml(model_xml)


def draw_bbox_on_image(img, bbox, color=(0, 0, 255), thickness=2):
    u_min, v_min, u_max, v_max = map(int, bbox)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # Draw the rectangle
    draw.rectangle([u_min, v_min, u_max, v_max], outline="blue", width=2)
    img_array = np.array(img)
    return img_array


def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}

        try:
            _apply_ep_meta_and_reset(env, ep_meta)
            xml = _prepare_xml(env, state["model"])
            env.reset_from_xml_string(xml)

        except (FileNotFoundError, PermissionError):
            if "object_cfgs" in ep_meta:
                ep_meta["object_cfgs"] = update_mjcf_paths(ep_meta["object_cfgs"])

            _apply_ep_meta_and_reset(env, ep_meta)

            xml = _prepare_xml(env, state["model"])
            xml = path_change(xml)
            env.reset_from_xml_string(xml)

        # env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None


def plot_eef_and_hand_qpos(eef_pos_list, hand_qpos):
    if len(eef_pos_list) == 0 or len(hand_qpos) == 0:
        print("No data to plot.")
        return

    eef_pos_array = np.array(eef_pos_list)
    hand_qpos_array = np.array(hand_qpos)
    num_sub_plot = 17

    fig, axes = plt.subplots(
        num_sub_plot, figsize=(6, 40), height_ratios=[1] * num_sub_plot
    )

    # Plot end-effector position (X, Y, Z)
    axes[0].plot(eef_pos_array[:, 0], label="X")
    axes[0].plot(eef_pos_array[:, 1], label="Y")
    axes[0].plot(eef_pos_array[:, 2], label="Z")
    axes[0].set_title("EEF Position vs Timestep")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Position (m)")
    axes[0].grid(True)
    axes[0].legend()

    # Plot each hand joint position
    for i in range(16):
        axes[i + 1].plot(hand_qpos_array[:, i])
        axes[i + 1].set_title(f"Hand Joint {i}")
        axes[i + 1].set_xlabel("Timestep")
        axes[i + 1].set_ylabel("Position (rad)")
        axes[i + 1].grid(True)

    plt.tight_layout()
    plt.savefig("eef_and_hand_qpos_vs_timestep.png")
    plt.close()
    print("Plot saved as 'eef_and_hand_qpos_vs_timestep.png'")


def convert_to_right_base(lines):
    """
    This helper function is used to re-position the right_base site
    to the base of the right arm
    """
    for i, line in enumerate(lines):
        if "right_center" in line:
            # Replace or add pos
            if "0 0 0" in line:
                line = line.replace(
                    "0 0 0",
                    "0.247 -0.050681 0.65",
                )

            # Replace or add quat
            if 'quat="' in line:
                line = re.sub(
                    r'quat="[^"]*"', 'quat="0.865807 0.436878 0.0222879 0.242939"', line
                )
            else:
                # Insert quat attribute before closing >
                line = line.rstrip().rstrip("/>")
                line += ' quat="0.865807 0.436878 0.0222879 0.242939"/>'

            lines[i] = line
            break
    return lines


def fix_camera_angles(lines):
    """
    This helper function is used to fix the camera angles
    for initial batches of mimicgen data
    """
    for i, line in enumerate(lines):
        if "robot0_robotview_2" in line:
            # Only replace quaternion for robot0_robotview_2
            if "-0.5 -0.5 0.5 0.5" in line:
                replaced_line = line.replace(
                    "-0.5 -0.5 0.5 0.5",
                    "0.61237244 0.35355339 -0.35355339 -0.61237244",
                )
                print(f"Updated line {i}: replaced quaternion for robot0_robotview_2")

            if "0.152 0 1.31" in replaced_line:
                replaced_line = replaced_line.replace("0.152 0 1.31", "0.152 0 1.614")
                print(f"Updated line {i}: replaced '0.152 0 1.31'")
            lines[i] = replaced_line
        elif "robot0_robotview" in line:
            replaced_line = line  # start with original line

            if "0.152 0 0.9544" in replaced_line:
                replaced_line = replaced_line.replace("0.152 0 0.9544", "0.152 0 1.014")
                print(f"Updated line {i}: replaced '0.152 0 0.9544'")

            lines[i] = replaced_line
    return lines


def fix_camera_angles_gripper(lines):
    """
    This helper function is used to fix the camera angles
    for initial batches of mimicgen data
    """
    root = ET.fromstring(lines)
    camera_att = find_elements(
        root, tags="camera", attribs={"name": "gripper0_right_right_eye_in_hand"}
    )
    camera_att.set("resolution", "480 300")
    modified_model_str = ET.tostring(root, encoding="unicode")
    return modified_model_str



def fix_left_gripper_type_for_backward_compatibility(lines):
    """
    This function checks if there is a gripper0_left_joint.
    If we use WonikAllegroLeft, the name is gripper0_left_joint_l_0.0;
    if we use WonikAllegro, the name is gripper0_left_joint_0.0.
    """
    root = ET.fromstring(lines)
    joint_attribute = find_elements(
        root, tags="joint", attribs={"name": "gripper0_left_joint_0.0"}
    )

    use_WonikAllegro = joint_attribute is not None

    @property
    def old_default_gripper(self):
        return {"left": "WonikAllegro", "right": "WonikAllegro"}

    if use_WonikAllegro:
        print(
            "Overriding TMR_ROBOT's left gripper to WonikAllegro for backward compatibility"
        )
        TMR_ROBOT.default_gripper = old_default_gripper
