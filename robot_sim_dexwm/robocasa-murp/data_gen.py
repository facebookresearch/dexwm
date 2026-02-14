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
from torch import multiprocessing as mp
import gzip
import robocasa
import robosuite.utils.transform_utils as T

TARGET_OBSERVATION = [
    # Store name, the actual name in the env, the index of the interest
    ("robot0_joint_pos_right", "robot0_joint_pos", [7, 14]),  #  franka arm
    ("robot0_right_eef_pos", "robot0_right_eef_pos"),  # 3-dim
    ("robot0_right_eef_quat", "robot0_right_eef_quat"),  # 4-dim
    ("robot0_right_gripper_qpos", "robot0_right_gripper_qpos"),  # 16-dim
]
TARGET_OBSERVATION_HDF5 = [
    # Original key name, name in the saving
    ("robot0_joint_pos", "robot0_joint_pos_right", [7, 14]),
    ("robot0_base_to_right_eef_pos", "robot0_right_eef_pos"),
    ("robot0_base_to_right_eef_quat_site", "robot0_right_eef_quat"),
    ("robot0_right_gripper_qpos", "robot0_right_gripper_qpos"),
    # ("robot0_robotview_2_image", "robot0_robotview_2"),
    # ("gripper0_right_right_eye_in_hand_image", "gripper0_right_right_eye_in_hand"),
]
# dict_keys([
# 'robot0_robotview',
# 'robot0_robotview_2',
# 'gripper0_right_right_eye_in_hand',
# 'gripper0_left_right_eye_in_hand',
# 'frontview'])

# 0.002 sec for sim step
# saving traj with 100 steps skipping
# so 0.002*100 = 0.2 => 5Hz


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
    render_depth=True,
    skip_interval=1,
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

    # if verbose:
    render_depth=True
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

    if render is False:
        print(colored("Running episode...", "yellow"))

    obs = env._get_observations(force_update=True)
    rgb_img = {}
    depth_img_r = {}  # Ensure it's a dict, not reused from previous loop
    for cam_name in camera_names:
        rgb, depth = env.sim.render(
            height=camera_height, width=camera_width, camera_name=cam_name, depth=True
        )
        rgb_img[cam_name]=rgb[::-1]
        # Set clipping range
        if cam_name == "gripper0_right_right_eye_in_hand":
            min_depth_value = 0.1
            max_depth_value = 8.0
        else:
            min_depth_value = 0.1
            max_depth_value = 20.0

        # Clip and normalize depth
        depth = np.clip(depth, min_depth_value, max_depth_value)
        depth[depth == -np.inf] = 0.0
        depth = 1.0 - depth
        depth = (depth * 255).astype(np.uint8)

        # Store in dict safely
        depth_img_r[cam_name+"_depth"] = depth

    data_traj = []
    for i in range(traj_len):
        if i % 20 == 0:
            print(f"step: {i}/{traj_len}")

        start = time.time()
        data_dict = {}
        gripper_body_names = env.robots[0].robot_model.grippers['robot0_right_hand'].bodies
        gripper_body_poses = {}
        for body_name in gripper_body_names:
            body_id = env.sim.model.body_name2id(body_name)
            body_pos = env.sim.data.body_xpos[body_id] # in world frame
            gripper_body_poses[body_name] = body_pos.copy()

        data_dict["gripper_body_poses"] = gripper_body_poses
        data_dict["text"] = lang


        obs_current = {}
        for target in TARGET_OBSERVATION:
            if len(target) == 3:
                obs_current[target[0]] = obs[target[1]][target[2][0] : target[2][1]]
            else:
                if target[0] == "robot0_right_eef_pos":
                    # Make it be in the base frame
                    obs_current[target[0]] = (
                        env.robots[0]
                        .composite_controller.part_controllers["right"]
                        .world_to_origin_frame(obs[target[1]])
                    )
                elif target[0] == "robot0_right_eef_quat":
                    # Use approach 1 to compute the local quat
                    # approach_1 = T.mat2quat(env.robots[0].composite_controller.part_controllers["right"].goal_origin_to_eef_pose())

                    # Use approach 2 to compute the local quat
                    origin_pose = T.make_pose(
                        env.robots[0]
                        .composite_controller.part_controllers["right"]
                        .origin_pos,
                        env.robots[0]
                        .composite_controller.part_controllers["right"]
                        .origin_ori,
                    )
                    ee_pose = T.make_pose(
                        obs["robot0_right_eef_pos"],
                        T.quat2mat(obs["robot0_right_eef_quat_site"]),
                    )
                    origin_pose_inv = T.pose_inv(origin_pose)
                    approach_2 = T.mat2quat(
                        T.pose_in_A_to_pose_in_B(ee_pose, origin_pose_inv)
                    )
                    obs_current[target[0]] = approach_2
                    breakpoint()
                else:
                    obs_current[target[0]] = obs[target[1]]
                
        obs_current = obs_current | rgb_img # merging rgb_img into obs_current
        obs_current = obs_current | depth_img_r # merging rgb_img into obs_current
        action_set={}

        # if action_playback:
        #     action_set["actions"] = actions[i]

        combined = obs_current.copy()
        combined.update(data_dict) 
        if actions is not None:
            combined["actions"] = actions[i]

        data_traj.append(combined)
        # breakpoint()

        if action_playback:
            obs, _, _, _ = env.step(actions[i])
            
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
            obs = env._get_observations(force_update=True)

        # on-screen render
        if render:
            if env.viewer is None:
                env.initialize_renderer()

        depth_images = []
        rgb_images = []
        rgb_img = {}
        depth_img_r = {}

        if render_depth:
            for cam_name in camera_names:
                _rgb_img, _depth_img = env.sim.render(
                    height=camera_height,
                    width=camera_width,
                    camera_name=cam_name,
                    depth=True,
                )

                depth_img = _depth_img[::-1]
                rgb_img[cam_name] = _rgb_img[::-1]
                rgb_images.append(_rgb_img[::-1])

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

                depth_img = 1.0 - depth_img
                depth_img = (depth_img * 255).astype(np.uint8)
                depth_img_r[cam_name+"_depth"] = depth_img

                depth_images.append(depth_img)
        else:
            for cam_name in camera_names:
                _rgb_img = env.sim.render(
                    height=camera_height, width=camera_width, camera_name=cam_name
                )[::-1]
                rgb_img[cam_name] = _rgb_img
                rgb_images.append(_rgb_img)
        ######### INCASE TO ADD DEPTH IMAGES ########################
        # if render_depth:
        #     depth_images = [
        #         np.expand_dims(depth_img, axis=-1) for depth_img in depth_images
        #     ]
        #     depth_images = [
        #         np.repeat(depth_img, 3, axis=-1) for depth_img in depth_images
        #     ]
        #     rgb_images = np.concatenate(rgb_images, axis=1)

        #     depth_img = np.concatenate(
        #         depth_images, axis=1
        #     )  # Concatenate depth images horizontally
        #     combined_img = np.concatenate([rgb_images, depth_img], axis=0)
        # else:
        combined_img = np.concatenate(rgb_images, axis=1)

        # max_fr = 60
        # elapsed = time.time() - start
        # diff = 1 / max_fr - elapsed
        # if diff > 0:
        #     time.sleep(diff)

        if write_video:
            if video_count % video_skip == 0:
                # Use the previously processed images for video writing
                # (1024, 2560, 3) or (512, 2560, 3)
                video_writer.append_data(combined_img)
            video_count += 1

        if first:
            break

        # for target_name in TARGET_OBSERVATION_HDF5:
        #     breakpoint()

        #     if len(target_name) == 3:

        #         data_dict[target_name[1]] = obs[target_name[0]][i][
        #             target_name[2][0] : target_name[2][1]
        #         ]
        #     else:
        #         data_dict[target_name[1]] = obs[target_name[0]][i]
        # data_traj.append(data_dict)

    if render:
        env.viewer.close()
        env.viewer = None
    # breakpoint()
    data_traj = post_process_traj(data_traj, skip_interval)

    return data_traj


def post_process_traj(data_traj, skip_interval):
    data_traj = data_traj[0::skip_interval]
    _data_traj = []
    for i in range(len(data_traj) - 1):
        _data = {}
        for name in data_traj[i]:
            _data[f"current_{name}"] = data_traj[i][name]
        for target in TARGET_OBSERVATION_HDF5:
            _data[f"next_{target[0]}"] = data_traj[i + 1][target[0]]
        _data_traj.append(_data)
    return _data_traj


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

        env.reset_from_xml_string(xml)
        # env.sim.reset()
        # hide teion after restoring from model
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


def playback_single_traj(demos, index_range, args, write_video, conn=None):

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
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = write_video
        env_kwargs["use_camera_obs"] = True
        env_kwargs["camera_depths"] = True

        if args.verbose:
            print(
                colored(
                    "Initializing environment for {}...".format(env_kwargs["env_name"]),
                    "yellow",
                )
            )
        env = robosuite.make(**env_kwargs)

    f = h5py.File(args.dataset, "r")
    for ind in range(len(demos)):
        if index_range[0] <= ind and ind <= index_range[1]:
            ep = demos[ind]
            print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

            # maybe dump video
            video_writer = None

            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            lines = initial_state["model"].splitlines()
            for i, line in enumerate(lines):
                if 'robot0_robotview_2' in line:
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
                elif 'robot0_robotview' in line:
                    replaced_line = line  # start with original line
                    
                    if "0.152 0 0.9544" in replaced_line:
                        replaced_line = replaced_line.replace("0.152 0 0.9544", "0.152 0 1.014")
                        print(f"Updated line {i}: replaced '0.152 0 0.9544'")
                    
                    
                    lines[i] = replaced_line



            initial_state["model"] = "\n".join(lines)
            initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get(
                "ep_meta", None
            )

            if args.extend_states:
                states = np.concatenate((states, [states[-1]] * 50))

            # supply actions if using open-loop action playback
            actions = None

            assert not (
                args.use_actions and args.use_abs_actions
            )  # cannot use both relative and absolute actions
            if args.use_actions:
                actions = f["data/{}/actions".format(ep)][()]
            elif args.use_abs_actions:
                actions = f["data/{}/actions_abs".format(ep)][()]  # absolute actions

            # Play back the traj
            data_traj = playback_trajectory_with_env(
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
                render_depth=args.render_depth,
                skip_interval=args.skip_interval,
            )

            # if write_video:
            #     print(colored(f"Saved video...", "green"))
            #     video_writer.close()

            # Save traj
            file = gzip.GzipFile(f"{args.video_path}episode-{ind}.npy.gz", "w")
            np.save(file=file, arr=data_traj)
            file.close()

    f.close()

    if env is not None:
        env.close()

    if conn is not None:
        conn.send(1)

    return


def playback_dataset(args):
    # some arg checking
    write_video = args.render is not True
    if args.video_path is None:
        args.video_path = "tmp_debug/"

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

    # Force not to write the video
    if args.force_no_video:
        write_video = False

    # Multi-process of the data
    if not args.multi_process:
        index_range = [0, len(demos) - 1]
        playback_single_traj(demos, index_range, args, write_video)
    else:
        assert (
            args.index_interval_inclusive[0] <= args.index_interval_inclusive[1]
        ), f"Wrong value of args.index_interval_inclusive"

        _start_id = max(0, args.index_interval_inclusive[0])
        _end_id = min(len(demos) - 1, args.index_interval_inclusive[1])
        start_time = time.time()
        chunked_index = []
        num_episodes = _end_id - _start_id + 1
        ochunk_size = num_episodes // args.num_process
        start = _start_id
        for i in range(args.num_process):
            chunk_size = ochunk_size
            if i < (num_episodes % args.num_process):
                chunk_size += 1
            end = start + chunk_size - 1
            chunked_index.append((start, end))
            start += chunk_size


        # Define the processor
        mp_ctx = mp.get_context("forkserver")
        proc_infos = []

        for index_range in chunked_index:
            parent_conn, child_conn = mp_ctx.Pipe()
            proc_args = (
                demos,
                index_range,
                args,
                write_video,
                child_conn,
            )
            p = mp_ctx.Process(target=playback_single_traj, args=proc_args)
            p.start()
            proc_infos.append((parent_conn, p))

        # Get back info (blocking call)
        total_received = 0
        for conn, proc in proc_infos:
            total_received += conn.recv()
            proc.join()
        end_time = time.time()
        print(
            colored(
                f"Total processes completed: {total_received}/{args.num_process}; {end_time-start_time} sec",
                "green",
            )
        )


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
        "--video-path",
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
            "robot0_robotview_2",
            "gripper0_right_right_eye_in_hand",
            # "gripper0_left_right_eye_in_hand",
            # "frontview",
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
        # default=True,
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

    parser.add_argument(
        "--multi-process",
        action="store_true",
        help="to use multi process or not",
    )

    parser.add_argument(
        "--num-process",
        type=int,
        default=5,
        help="the number of processes",
    )

    parser.add_argument(
        "--render-depth",
        action="store_true",
        default=True,
        help="To use depth image or not",
    )

    parser.add_argument(
        "--skip-interval",
        type=int,
        default=1,
        help="The interval to determine the current or the next observation",
    )

    parser.add_argument(
        "--force-no-video",
        action="store_true",
        help="no video",
    )
    parser.add_argument(
        "--index-interval-inclusive",
        nargs="+",
        type=int,
        help="The interval to process the data",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_playback_args()
    playback_dataset(args)

# python data_gen.py --dataset /checkpoint/siro/jimmytyyang/robot-skills-dataset/vyshnav_data/d_ratio6/mimicgen_combined_may/demo/demo.hdf5 --multi-process --force-no-video

# python data_gen.py --dataset /checkpoint/siro/jimmytyyang/robot-skills-dataset/vyshnav_data/small_objs_kitchen/small_objs_kitchen/combined/combined_success.hdf5 --video-path /checkpoint/siro/jimmytyyang/robot-skills-dataset/vyshnav_data/small_objs_kitchen_combined_success_may15/