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
from scipy.spatial.transform import Rotation as R
import robosuite.utils.transform_utils as T

import robocasa


def playback_trajectory_with_env(
    env,
    initial_state,
    states,
    actions=None,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    first=False,
    verbose=False,
    camera_height=512,
    camera_width=512,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state.
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    write_video = video_writer is not None
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    ## this reset call doesn't seem necessary.
    ## seems ok to remove but haven't fully tested it.
    ## removing for now
    # env.reset()

    if verbose:
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "green"))
        print(colored("Spawning environment...", "yellow"))
    reset_to(env, initial_state)

    traj_len = states.shape[0]
    action_playback = actions is not None
    if action_playback:
        assert states.shape[0] == actions.shape[0]
    eef_pos_list = []
    hand_qpos = []
    base_poses_eef = np.zeros(3)

    if render is False:
        print(colored("Running episode...", "yellow"))
    for _ in range(10):  # try 5–10 steps of no-op
        env.sim.forward()
        env.sim.step()

    file_path = './ep_0_robocasa_eef_fk.npy'
    data = np.load(file_path, allow_pickle=True)
    len_data = len(data)
    # arm_joint_key = "current_robot0_right_joint_pos"
    # hand_joint_key = "current_robot0_right_gripper_qpos"
    actions[:len_data,:6]= np.array([
        np.concatenate((
            d['right_fr3_link8_T_base_link'][:3],  # local xyz position
            T.quat2axisangle(d['right_fr3_link8_T_base_link'][3:7])
        )) for d in data
    ])
    # actions[:len_data,7:23]= [d['current_robot0_right_gripper_qpos'] for d in data_hands[:len_data]]
    ee_T_base_poses = []
    for i in range(len_data):
        start = time.time()

        if action_playback:
            # print(len(actions[i]),len(states))
            # base_T_world_pos = env.sim.data.get_body_xpos('robot0_base')        # shape (3,)
            # base_T_world_mat = env.sim.data.get_body_xmat('robot0_base')      # (w, x, y, z) 
            # base_T_world_quat = env.sim.data.get_body_xquat('robot0_base')      # (w, x, y, z) 
            # base_T_world_pos = env.sim.data.get_site_xpos('robot0_right_center')        # shape (3,)
            # base_T_world_mat = env.sim.data.get_site_xmat('robot0_right_center')
            # base_T_world_pos = env.sim.data.get_body_xpos('robot0_torso')
            # base_T_world_mat = env.sim.data.get_body_xmat('robot0_torso')

            base_T_world_pos = env.sim.data.get_site_xpos('robot0_right_center')
            base_T_world_mat = env.sim.data.get_site_xmat('robot0_right_center')

            ee_T_world_pos = env.sim.data.get_body_xpos('gripper0_right_eef')        # shape (3,)
            ee_T_world_mat = env.sim.data.get_body_xmat('gripper0_right_eef')

            # ee_T_world_pos = env.sim.data.get_site_xpos('robot0_right_center')        # shape (3,)
            # ee_T_world_mat = env.sim.data.get_site_xmat('robot0_right_center')
            # convert mat to quat
            ee_T_world_quat = R.from_matrix(ee_T_world_mat).as_quat()

            ee_T_world = np.eye(4)
            ee_T_world[:3, -1] = ee_T_world_pos
            ee_T_world[:3, :3] = ee_T_world_mat

            base_T_world = np.eye(4)
            base_T_world[:3, -1] = base_T_world_pos
            base_T_world[:3, :3] = base_T_world_mat

            ee_T_base = ee_T_world @ np.linalg.inv(base_T_world)


            ee_T_base = np.linalg.inv(base_T_world) @ ee_T_world
            ee_T_base_inv = ee_T_world @ np.linalg.inv(base_T_world)
            base_T_ee = base_T_world @ np.linalg.inv(ee_T_world)
            base_T_ee_inv = np.linalg.inv(ee_T_world) @ base_T_world

            curr_ee_T_base_pos = ee_T_base[:3, -1]
            curr_ee_T_base_quat = R.from_matrix(ee_T_base[:3, :3]).as_quat() # x y z w

            print('ee_T_world_pos: ', ee_T_world_pos)
            print('base_T_world_pos: ', base_T_world_pos)
            print('curr_ee_T_base_pos: ', curr_ee_T_base_pos)
            print('ee_T_base_inv: ', ee_T_base_inv[:3, -1])
            print('base_T_ee: ', base_T_ee[:3, -1])
            print('base_T_ee_inv: ', base_T_ee_inv[:3, -1])

            ee_T_base_poses.append(np.concatenate([curr_ee_T_base_pos, curr_ee_T_base_quat]))
            obs, _, _, _ = env.step(actions[i])
            # print('same?: ', np.allclose(actions[i][:7], obs['robot0_joint_pos'][:7]))

            ################## DEBUGGING ##################
            # if i!=0:
            #     delta= np.linalg.norm(base_poses_eef - obs['robot0_base_to_right_eef_pos'])

            #     # print(f"Delta Val {delta} and current {i}")
            # base_poses_eef = obs['robot0_base_to_right_eef_pos']
            # eef_pos_list.append(obs['robot0_base_to_right_eef_pos'])
            # hand_qpos.append(obs['robot0_right_gripper_qpos'])

            # breakpoint()
            # print("time for step : {}".format(time.time() - start))
            ################## DEBUGGING ##################

            if i < traj_len - 1:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = np.array(env.sim.get_state().flatten())
                if not np.all(np.equal(states[i + 1], state_playback)):
                    err = np.linalg.norm(states[i + 1] - state_playback)
                    if verbose or i == traj_len - 2:
                        print(
                            colored(
                                "warning: playback diverged by {} at step {}".format(
                                    err, i
                                ),
                                "yellow",
                            )
                        )
        else:
            reset_to(env, {"states": states[i]})

        np.save('./robocasa_ee_T_right_base_torso.npy', ee_T_base_poses)        # on-screen render
        render_depth = True
        if render:
            if env.viewer is None:
                env.initialize_renderer()
        if render_depth:
            depth_images = []
            rgb_images = []
            for cam_name in camera_names:
                rgb_img = env.sim.render(
                    height=camera_height, width=camera_width, camera_name=cam_name
                )[::-1]

                rgb_img2, depth_img = env.sim.render(
                    height=camera_height,
                    width=camera_width,
                    camera_name=cam_name,
                    depth=True,
                )

                depth_img = depth_img[::-1]

                rgb_images.append(rgb_img)

                min_depth_value = 0.1
                max_depth_value = 20.0

                if (
                    cam_name == "gripper0_right_right_eye_in_hand"
                    or "gripper0_right_right_eye_in_hand"
                ):
                    min_depth_value = 0.1
                    max_depth_value = 8.0
                depth_img = np.clip(depth_img, min_depth_value, max_depth_value)

                depth_img[depth_img == -np.inf] = 0.0

                # depth_img = (depth_img - min_depth_value) / (
                #     max_depth_value - min_depth_value
                # )
                depth_img = 1.0 - depth_img
                depth_img = (depth_img * 255).astype(np.uint8)
                depth_images.append(depth_img)

            depth_images = [
                np.expand_dims(depth_img, axis=-1) for depth_img in depth_images
            ]
            depth_images_rgb = [
                np.repeat(depth_img, 3, axis=-1) for depth_img in depth_images
            ]

            rgb_img = np.concatenate(rgb_images, axis=1)

            depth_img = np.concatenate(
                depth_images_rgb, axis=1
            )  # Concatenate depth images horizontally
            combined_img = np.concatenate([rgb_img, depth_img], axis=0)

            max_fr = 60
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

            if write_video:
                if video_count % video_skip == 0:
                    # Use the previously processed images for video writing
                    if render_depth:
                        video_writer.append_data(
                            combined_img
                        )  # Write the combined image (RGB + Depth)
                    else:
                        video_img = np.concatenate(
                            rgb_images, axis=1  # Concatenate RGB images horizontally
                        )
                        video_writer.append_data(video_img)

            video_count += 1

        if first:
            break

    if render:
        env.viewer.close()
        env.viewer = None
    ########################### FOR PLOTTING ###########################
    # if len(eef_pos_list) > 0:
    #     import matplotlib.pyplot as plt
    #     eef_pos_array = np.array(eef_pos_list)
    #     hand_qpos_array = np.array(hand_qpos)
    #     num_sub_plot = 17
    #     fig, axes = plt.subplots(num_sub_plot, figsize=(6, 40), height_ratios=[1] * num_sub_plot)

    #     axes[0].plot(eef_pos_array[:, 0], label="X")
    #     axes[0].plot(eef_pos_array[:, 1], label="Y")
    #     axes[0].plot(eef_pos_array[:, 2], label="Z")
    #     axes[0].set_title("EEF Position vs Timestep")
    #     axes[0].set_xlabel("Timestep")
    #     axes[0].set_ylabel("Position (m)")
    #     axes[0].grid(True)
    #     axes[0].legend()

    #     for i in range(16):
    #         axes[i + 1].plot(hand_qpos_array[:, i])
    #         axes[i + 1].set_title(f"Hand Joint {i}")
    #         axes[i + 1].set_xlabel("Timestep")
    #         axes[i + 1].set_ylabel("Position (rad)")
    #         axes[i + 1].grid(True)

    #     plt.tight_layout()
    #     plt.savefig("eef_and_hand_qpos_vs_timestep.png")
    #     plt.close()
    ########################### FOR PLOTTING ###########################


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


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
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
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])
        # target = "opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/achvysh07/projects"
        # import re

        # match = re.search(rf"\b{target}\b", xml, re.IGNORECASE)
        # xml = re.sub(
        #     rf"\b{target}\b",
        #     "checkpoint/siro/jtruong/repos/robocasa_murp",
        #     xml,
        #     flags=re.IGNORECASE,
        # )
        # env.reset_from_xml_string(xml)
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
def convert_to_right_base(lines):
    """
    This helper function is used to re-position the right_base site
    to the base of the right arm
    """
    for i, line in enumerate(lines):
        if 'right_center' in line:
            # Replace or add pos
            if "0 0 0" in line:
                line = line.replace(
                    "0 0 0",
                    "0.247 -0.050681 0.65",
                )

            # Replace or add quat
            if 'quat="' in line:
                line = re.sub(r'quat="[^"]*"', 'quat="0.865807 0.436878 0.0222879 0.242939"', line)
            else:
                # Insert quat attribute before closing >
                line = line.rstrip().rstrip('/>')
                line += ' quat="0.865807 0.436878 0.0222879 0.242939"/>'

            lines[i] = line
            break
    return lines

def playback_dataset(args):
    # some arg checking
    write_video = args.render is not True
    if args.video_path is None:
        args.video_path = args.dataset.split(".hdf5")[0] + "_joints.mp4"
        if args.use_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_actions_joints.mp4"
        elif args.use_abs_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_abs_actions_joints.mp4"
    assert not (args.render and write_video)  # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        args.render_image_names = "robot0_robotview"

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert (
            not args.use_actions and not args.use_abs_actions
        ), "playback with observations is offline and does not support action playback"

    env = None

    # create environment only if not playing back with observations
    if not args.use_obs:
        # # need to make sure ObsUtils knows which observations are images, but it doesn't matter
        # # for playback since observations are unused. Pass a dummy spec here.
        # dummy_spec = dict(
        #     obs=dict(
        #             low_dim=["robot0_eef_pos"],
        #             rgb=[],
        #         ),
        # )
        # initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        if args.use_abs_actions:
            env_meta["env_kwargs"]["controller_configs"][
                "control_delta"
            ] = False  # absolute action space

        env_kwargs = env_meta["env_kwargs"]
        # env_kwargs["controller_configs"]["body_parts"]["right"][ "type" ] = "JOINT_POSITION"
        # env_kwargs["controller_configs"]["body_parts"]["right"]["output_min"] = [
        #     0.1,
        #     0.1,
        #     0.1,
        #     0.5,
        #     0.5,
        #     0.5,
        #     0.5,
        # ]
        # env_kwargs["controller_configs"]["body_parts"]["right"]["output_max"] = [
        #     -0.1,
        #     -0.1,
        #     -0.1,
        #     -0.5,
        #     -0.5,
        #     -0.5,
        #     -0.5,
        # ]
        # env_kwargs["controller_configs"]["body_parts"]["right"]["kp"]=100.0 
        # env_kwargs["controller_configs"]["body_parts"]["right"]["kd"]=50.0
        # env_kwargs["controller_configs"]["body_parts"]["left"]["type"] = "JOINT_POSITION"
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = write_video
        env_kwargs["use_camera_obs"] = True
        env_kwargs["camera_depths"] = True
        env_kwargs["control_freq"] = 1
        # env_kwargs["controller_configs"]["body_parts"]["right"]["kd"] = 10.0
        env_kwargs["controller_configs"]["body_parts"]["right"]["input_type"]="absolute"


        if args.verbose:
            print(
                colored(
                    "Initializing environment for {}...".format(env_kwargs["env_name"]),
                    "yellow",
                )
            )
        env = robosuite.make(**env_kwargs)

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    elif "data" in f.keys():
        demos = list(f["data"].keys())

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        random.shuffle(demos)
        demos = demos[: args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        if args.use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)],
                video_writer=video_writer,
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                first=args.first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"] # entire scene with robot
        #griper0_allegro_hand
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None) 
        lines = initial_state["model"].splitlines()
        lines = convert_to_right_base(lines)
        initial_state["model"] = "\n".join(lines)


        if args.extend_states:
            states = np.concatenate((states, [states[-1]] * 50))

        # supply actions if using open-loop action playback
        actions = None
        assert not (
            args.use_actions and args.use_abs_actions
        )  # cannot use both relative and absolute actions

        if args.use_actions:
            actions_og = f["data/{}/actions".format(ep)][()]
            actions = np.zeros((len(actions_og), 49))
            actions[:, 0:14] = f["data/{}/obs/robot0_joint_pos".format(ep)][()]
            actions[:, 14:] = actions_og[:, 12:]
            # breakpoint()

        elif args.use_abs_actions:
            actions = f["data/{}/abs_action".format(ep)][()]  # absolute actions

        playback_trajectory_with_env(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
            verbose=args.verbose,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
        )
    # plt.savefig("eef_position_vs_timestep.png")
    # plt.close()
    f.close()
    if write_video:
        print(colored(f"Saved video to {args.video_path}", "green"))
        video_writer.close()

    if env is not None:
        env.close()


def get_playback_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action="store_true",
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action="store_true",
        help="use open-loop action playback instead of loading sim states",
    )

    # Playback stored dataset absolute actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-abs-actions",
        action="store_true",
        help="use open-loop action playback with absolute position actions instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action="store_true",
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs="+",
        default=[
            "robot0_robotview",
            # "robot0_robotview_2",
            "gripper0_right_right_eye_in_hand",
            # "gripper0_left_right_eye_in_hand",
            "robot0_robotview_2",
        ],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
        "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action="store_true",
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--extend_states",
        action="store_true",
        help="play last step of episodes for 50 extra frames",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log additional information",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=512,
        help="(optional, for offscreen rendering) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=512,
        help="(optional, for offscreen rendering) width of image observations",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_playback_args()
    playback_dataset(args)
