import argparse
import json
import os
import random
import time

from PIL import Image, ImageDraw, ImageFont
import h5py
import imageio
import numpy as np
import robosuite
from termcolor import colored
import robosuite.utils.transform_utils as T
import robocasa.scripts.playback_utils as P
import robocasa
import re
from robosuite.utils.camera_utils import get_real_depth_map


def playback_trajectory_with_env(
    env,
    initial_state,
    states,
    obse,
    actions=None,
    abs_right_arm_base_action=None,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    first=False,
    verbose=False,
    camera_height=300,
    camera_width=480,
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
        free_form_lang = ep_meta.get("free_form_lang", None)
        if lang is not None:
            print(colored(f"Instruction (simple): {lang}", "green"))
            print(colored(f"Instruction (free-form): {free_form_lang}", "green"))
        print(colored("Spawning environment...", "yellow"))
    P.reset_to(env, initial_state)

    traj_len = states.shape[0]
    # breakpoint()
    action_playback = actions is not None
    if action_playback:
        assert states.shape[0] == actions.shape[0]
    eef_pos_list = []
    hand_qpos = []
    base_poses_eef = np.zeros(3)
    # observations = obse
    if args.real2sim:
        if args.pos_control:
            actions, data_len = P.real2sim_pos(actions, args.traj_file)
        else:
            actions, data_len = P.real2sim_osc(actions, args.traj_file)
        traj_len = data_len

    if render is False:
        print(colored("Running episode...", "yellow"))
    for _ in range(30):  # try 5–10 steps of no-op
        env.sim.forward()
        env.sim.step()

    for i in range(traj_len):
        start = time.time()

        if action_playback:
            # print(len(actions[i]),len(states))
            if args.osc_ref_base == "right_base" and not args.real2sim:
                actions[i][:6] = P.get_ee_T_arm_base(env, actions[i][:6])

            obs, _, _, _ = env.step(actions[i])

            if args.plot:
                position = obs["robot0_right_eef_T_right_base_pos"]
                quat = obs["robot0_right_eef_T_right_base_quat_xyzw"]
                robot0_right_eef_T_right_base = np.concatenate((position, quat), axis=0)
                eef_pos_list.append(robot0_right_eef_T_right_base)
                hand_qpos.append(obs["robot0_right_gripper_qpos"])

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
            P.reset_to(env, {"states": states[i]})

        # on-screen render
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

                if cam_name == "robot0_robotview" and action_playback:
                    bbox = obs["obj_bbox_in_robot0_robotview"]
                    rgb_img = P.draw_bbox_on_image(
                        rgb_img, bbox, color=(0, 0, 255), thickness=2
                    )

                if cam_name == "robot0_robotview_2" and action_playback:
                    bbox = obs["obj_bbox_in_robot0_robotview_2"]
                    rgb_img = P.draw_bbox_on_image(
                        rgb_img, bbox, color=(0, 0, 255), thickness=2
                    )
                if cam_name == "gripper0_right_right_eye_in_hand" and action_playback:
                    bbox = obs["obj_bbox_in_gripper0_right_right_eye_in_hand"]
                    rgb_img = P.draw_bbox_on_image(
                        rgb_img, bbox, color=(0, 0, 255), thickness=2
                    )

                rgb_img2, depth_img = env.sim.render(
                    height=camera_height,
                    width=camera_width,
                    camera_name=cam_name,
                    depth=True,
                )

                depth_img = get_real_depth_map(env.sim, depth_img)

                depth_img = depth_img[::-1]
                if depth_img.shape[-1] == 1:
                    depth_img = depth_img[:, :, 0]
                rgb_images.append(rgb_img)

                min_depth_value = 0.3
                max_depth_value = 10.0

                if (
                    cam_name == "gripper0_right_right_eye_in_hand"
                    or "gripper0_right_right_eye_in_hand"
                ):
                    min_depth_value = 0.2
                    max_depth_value = 8.0
                depth_img = np.clip(depth_img, min_depth_value, max_depth_value)

                depth_img[depth_img == -np.inf] = 0.0

                depth_img = (depth_img - min_depth_value) / (
                    max_depth_value - min_depth_value
                )
                # depth_img = 1.0 - depth_img
                depth_img = np.clip(depth_img, 0.0, 1.0)
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
                        # Add step counter in the image
                        pil_image = Image.fromarray(combined_img)
                        draw = ImageDraw.Draw(pil_image)
                        text = f"Step: {i}"
                        position = (10, 10)
                        text_color = (255, 0, 0)
                        font = ImageFont.truetype(
                            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                            40,
                            # "/Library/Fonts/Arial.ttf",40 #For MAC
                        )
                        draw.text(position, text, font=font, fill=text_color)
                        combined_img = np.array(pil_image)
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
    if args.plot:
        P.plot_eef_and_hand_qpos(eef_pos_list, hand_qpos)


def playback_dataset(args):
    # some arg checking
    write_video = args.render is not True
    if args.video_path is None:
        args.video_path = args.dataset.split(".hdf5")[0] + ".mp4"
        if args.use_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_actions.mp4"
        elif args.use_abs_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_abs_actions.mp4"
    assert not (args.render and write_video)  # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = P.get_env_metadata_from_dataset(dataset_path=args.dataset)
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

        env_meta = P.get_env_metadata_from_dataset(dataset_path=args.dataset)
        if args.pos_control:
            if args.use_abs_actions:
                env_meta["env_kwargs"]["controller_configs"][
                    "control_delta"
                ] = False  # absolute action space
                env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"][
                    "input_ref_frame"
                ] = args.osc_ref_base

            env_kwargs = env_meta["env_kwargs"]
            env_kwargs["controller_configs"]["body_parts"]["right"][
                "type"
            ] = "JOINT_POSITION"
            # env_kwargs["controller_configs"]["body_parts"]["right"]["kp"]=130.0
            env_kwargs["controller_configs"]["body_parts"]["right"]["output_max"] = [
                0.1,
                0.1,
                0.1,
                0.5,
                0.5,
                0.5,
                0.5,
            ]
            env_kwargs["controller_configs"]["body_parts"]["right"]["output_min"] = [
                -0.1,
                -0.1,
                -0.1,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
            ]
            env_kwargs["controller_configs"]["body_parts"]["left"][
                "type"
            ] = "JOINT_POSITION"
            env_kwargs["env_name"] = env_meta["env_name"]
            env_kwargs["has_renderer"] = False
            env_kwargs["renderer"] = "mjviewer"
            env_kwargs["has_offscreen_renderer"] = write_video
            env_kwargs["use_camera_obs"] = True
            env_kwargs["camera_depths"] = True
            env_kwargs["control_freq"] = args.control_freq
            env_kwargs["controller_configs"]["body_parts"]["right"]["kd"] = 10.0
        else:
            if args.use_abs_actions:
                env_meta["env_kwargs"]["controller_configs"][
                    "control_delta"
                ] = False  # absolute action space
                env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"][
                    "input_type"
                ] = args.input_type
                env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"][
                    "input_ref_frame"
                ] = args.osc_ref_base

            env_kwargs = env_meta["env_kwargs"]
            env_kwargs["env_name"] = env_meta["env_name"]
            env_kwargs["has_renderer"] = False
            env_kwargs["renderer"] = "mjviewer"
            env_kwargs["has_offscreen_renderer"] = write_video
            env_kwargs["use_camera_obs"] = True
            env_kwargs["camera_depths"] = True
            env_kwargs["control_freq"] = args.control_freq

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
        # random.shuffle(demos)
        demos = demos[: args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        if args.use_obs:
            P.playback_trajectory_with_obs(
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
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)
        #################### UNCOMMENT FOR OLDER DATASETS ################################
        #     lines = initial_state["model"].splitlines()
        #     #################### BUG ##################################
        #     """ These lines of code are added to change the camera viewing angles in older
        #     datasets generated from robocasa. The batch 1,2 and 3 of data have wrong camera
        #     angles compared to real world, and these need changes"""
        #     # lines = initial_state["model"].splitlines()
        #     # lines = P.fix_camera_angles(lines)
        #     # lines = initial_state["model"].splitlines()
        #     # lines = P.fix_camera_angles_gripper(lines)
        #     #######################################################################

        #    ######################### BUG ##########################################
        #     """ These lines of code are added to change the position of "right_center"
        #     site, which is used in xml to refer to the base of the right arm. Since XML
        #     parses body->geom->site, and arm base body is not created by XML, we add this
        #     fix to correct the pose in XML. Applicable to all 4 batches of data
        #     """
        #     lines=P.convert_to_right_base(lines)
        #     # initial_state["model"] = "\n".join(lines)
        #     ######################### BUG ##########################################
        #     # initial_state["model"] = "\n".join(lines)

        #################### UNCOMMENT FOR OLDER DATASETS ################################

        lines = P.fix_camera_angles_gripper(initial_state["model"])
        initial_state["model"] = lines

        if args.extend_states:
            states = np.concatenate((states, [states[-1]] * 50))

        # supply actions if using open-loop action playback
        actions = None
        abs_right_arm_base_action = None
        assert not (
            args.use_actions and args.use_abs_actions
        )  # cannot use both relative and absolute actions
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]
            if args.pos_control:
                actions_og = f["data/{}/actions".format(ep)][()]
                actions = np.zeros((len(actions_og), 49))
                actions[:, 0:14] = f["data/{}/obs/robot0_joint_pos".format(ep)][()]
                actions[:, 14:] = actions_og[:, 12:]
        elif args.use_abs_actions:
            actions = f["data/{}/abs_action".format(ep)][()]  # absolute actions
            abs_right_arm_base_action = f[
                "data/{}/abs_right_arm_base_action".format(ep)
            ][
                ()
            ]  # absolute actions
            if args.pos_control:
                actions_abs = f["data/{}/abs_action".format(ep)][()]  # absolute actions
                actions = np.zeros((len(actions_abs), 49))
                actions[:, 0:14] = f["data/{}/obs/robot0_joint_pos".format(ep)][()]
                actions[:, 14:] = actions_abs[:, 12:]
        try:
            obsers = f[
                "data/{}/obs/robot0_right_eef_T_right_base_quat_xyzw".format(ep)
            ][()]
        except:
            obsers = None

        playback_trajectory_with_env(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            abs_right_arm_base_action=abs_right_arm_base_action,
            render=args.render,
            obse=obsers,
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
        default=300,
        help="(optional, for offscreen rendering) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=480,
        help="(optional, for offscreen rendering) width of image observations",
    )
    parser.add_argument(
        "--src_path",
        type=str,
        default=None,
        help="(optional) Change xml paths in hdf5",
    )

    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
        help="(optional) Change xml paths in hdf5",
    )
    parser.add_argument(
        "--osc_ref_base",
        type=str,
        default="base",
        help="the osc controller reference base",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plotting handqpos and also robot eef poses and quat",
    )
    parser.add_argument(
        "--pos_control",
        action="store_true",
        help="Plotting handqpos and also robot eef poses and quat",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="absolute",
        help="the osc controller reference base",
    )
    parser.add_argument(
        "--real2sim",
        action="store_true",
        help="Mode to play realworld data on sim using position controller",
    )
    parser.add_argument(
        "--control_freq",
        type=int,
        default=10,
        help="Control frequeny for the controller to operate",
    )
    parser.add_argument(
        "--traj_file",
        type=str,
        help="Hardware traj .npy or .json file",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_playback_args()
    ############### Uncomment for reading all hdf5 in a folder and playing back #####################
    # import glob
    # import copy
    # dataset_files = sorted(glob.glob(args.dataset))
    # if not dataset_files:
    #     print(f"No files found for pattern: {args.dataset}")
    # for i, dataset_file in enumerate(dataset_files):

    #     print(f"Playing back file {i + 1}/{args.n}: {dataset_file}")
    #     file_args = copy.deepcopy(args)
    #     file_args.dataset = dataset_file

    #     playback_dataset(file_args)
    ############### Uncomment for reading all hdf5 in a folder and playing back #####################
    playback_dataset(args)
