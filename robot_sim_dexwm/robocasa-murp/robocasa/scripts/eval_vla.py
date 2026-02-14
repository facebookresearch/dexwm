import argparse
import json
import random
import time

import h5py
import imageio
import numpy as np
import robosuite
from termcolor import colored
import traceback
from PIL import Image
import copy
from collections import defaultdict

import pickle
import robocasa
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
import glob
from torch import multiprocessing as mp
import sys
from pathlib import Path
import xmltodict
from playback_utils import get_ee_T_arm_base
from scripts.s3_utils import download_s3_checkpoint, list_s3_checkpoints, init_s3_client
import robocasa.scripts.playback_utils as P

sys.setrecursionlimit(10000)
from vla_utils import (
    infer_action,
    normalize,
    load_vla_skill,
    dennormalize,
    normalize,
    NAME_MAPPING,
    TEXT_BANK,
    rot_mat_six_dim_to_axisangle,
    REPEAT_BATCH,
    process_input,
    get_vla_obs,
    process_lang,
    crop_image,
    add_img_to_obs,
    debug_plot,
    move_target_object_given_global_xyz,
)
import torch
import time


import os

def chmod_with_parents(path, mode=0o777):
    """Set permissions on a directory and all its parent directories."""
    try:
        # Set permissions on the target directory
        if os.path.exists(path):
            os.chmod(path, mode)
        
        # Set permissions on parent directories
        parent = os.path.dirname(path)
        while parent and parent != '/' and parent != os.path.dirname(parent):
            if os.path.exists(parent):
                os.chmod(parent, mode)
            parent = os.path.dirname(parent)
    except PermissionError:
        # If we don't have permission to change parent dirs, just continue
        pass

def assign_control_action(
    env,
    vla_action,
    obs,
):
    """Assign the control action to a correct action input of env.step().
    The default controller uses relative delta between the current pose/location to the target.
    Hence, we compute the delta.
    """
    env_vla_action = np.zeros(env.action_spec[0].shape)
    # Absolute action mode for the ee pose, and absolute action mode for the fingers
    # Assign the delta action to the correct location
    rot_axisangle = rot_mat_six_dim_to_axisangle(vla_action[3:9])
    absolute = {
        "right": np.concatenate((vla_action[0:3], rot_axisangle), axis=0),
        "right_gripper": vla_action[9:25],
    }
    for controller_name in ["right", "right_gripper"]:
        ctrl_index = env.robots[0].composite_controller._action_split_indexes[
            controller_name
        ]
        env_vla_action[ctrl_index[0] : ctrl_index[1]] = absolute[controller_name]

    # ===Dump===
    # Since the OSC controller uses the world frame for self.ref_pose, and
    # the self.goal_pos is in the base frame,
    # we need to get the delta by using the base frame.
    # target_right_eef_pose_in_origin_frame = (
    #     env.robots[0]
    #     .composite_controller.part_controllers["right"]
    #     .world_to_origin_frame(target_right_eef_pose[0:3])
    # )
    # current_right_eef_pose_in_origin_frame = (
    #     env.robots[0]
    #     .composite_controller.part_controllers["right"]
    #     .world_to_origin_frame(current_right_eef_pose[0:3])
    # )

    # Useful APIs
    # env.robots[0].composite_controller.part_controllers
    # env.robots[0].composite_controller._action_split_indexes
    # env.robots[0].composite_controller.run_controller(env.robots[0]._enabled_parts
    # OrderedDict([('right', (0, 6)), ('left', (6, 12)), ('base', (12, 15)), ('right_gripper', (15, 31)), ('left_gripper', (31, 47))])

    return env_vla_action


def run_vla_trajectory_with_env(
    env,
    initial_state,
    render=False,
    states=None,
    actions=None,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    verbose=False,
    camera_height=600,
    camera_width=960,
    vla_path=None,
    max_steps=2500,
    depoly_aciton_interval=20,
    save_name="",
    vla_skill=None,
    vla_processor=None,
    vla_config=None,
    training_stats=None,
    cuda_device="cuda:0",
    image_process_mode="resize",
    dynamic_simulation=False,
    pad_zero_dim=0,
    move_obj_xyz=None,
    training_stats_key=None,
    instruction=None,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state.
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
    """
    write_video = video_writer is not None
    video_count = 0
    assert not (render and write_video)

    ep_meta = json.loads(initial_state["ep_meta"])
    if instruction is not None:
        lang = instruction
    else:
        lang = ep_meta.get("lang", None)
        lang = process_lang(lang)
    print(f"Instruction: {lang} / saving plot in {save_name}")

    P.reset_to(env, initial_state)

    if render is False:
        print(colored("Running episode...", "yellow"))

    # Optionally to move the object to pertube the env
    if move_obj_xyz is not None:
        move_target_object_given_global_xyz(env, move_obj_xyz)

    if dynamic_simulation:
        for _ in range(20):  # try no-op steps to stabalize the scene
            env.sim.forward()
            env.sim.step()

    obs = env._get_observations(force_update=True)
    obs = add_img_to_obs(env, obs, camera_height, camera_width)

    vla_action_list = []
    gt_action_list = []
    obs_list = []
    infer_timestep_list = []

    max_steps = int(max_steps)
    success = False
    obj_up_once = False
    last_metadata = {}
    gt_action_len = actions.shape[0]

    if not dynamic_simulation:
        max_steps = min(gt_action_len, max_steps)

    # Start running the eval
    env.actions_meta = defaultdict(list)

    for i in range(max_steps):
        if i % 50 == 0:
            print(f"Step: {i} / {max_steps}")
            sys.stdout.flush()

        start = time.time()

        # Log the prediction and observation for debugging plot
        obs_list.append(obs.copy())

        # Log gt actions
        if i >= gt_action_len:
            # out of bound
            # just append the last action
            last_action = gt_action_list[-1]
            gt_action_list.append(last_action.copy())
        else:
            actions[i][:6] = P.get_ee_T_arm_base(env, actions[i][:6])
            ee_pos = actions[i][:3]
            ee_qat_mat = T.quat2mat(T.axisangle2quat(actions[i][3:6]))[:2].flatten()
            gt_action_list.append(
                np.concatenate(
                    (ee_pos, ee_qat_mat, actions[i][15 : 15 + 16]), axis=0
                ).copy()
            )

        # check success:
        if env._check_success():
            success = True
            break
        #call the actions meta data using
        actions_meta = env.actions_meta

        if env.obj_up_once and not obj_up_once:
            obj_up_once = True

        # Get the action
        if i % depoly_aciton_interval == 0:
            print(f"Fetch actions @ step {i}")
            sys.stdout.flush()
            infer_timestep_list.append(i)
            pred = []
            vla_obs = get_vla_obs(
                env,
                camera_names,
                camera_height,
                camera_width,
                vla_config,
                obs_list,
                lang,
                time_index=i,
                pad_zero_dim=pad_zero_dim,
            )
            vla_action = infer_action(
                vla_skill,
                vla_processor,
                vla_config,
                vla_obs,
                training_stats=training_stats,
                training_stats_key=training_stats_key,
                cuda_device=cuda_device,
                save_path=save_name[0:-4] + f"_time{i}.pkl",
            )

            for j in range(depoly_aciton_interval):
                _vla_action = torch.mean(vla_action, dim=0)[j].cpu().detach().numpy()
                pred.append(_vla_action.copy())

        # Depoly the action
        vla_action = pred.pop(0)

        # Log the prediction and observation for debugging plot
        vla_action_list.append(vla_action.copy())

        # process the action to be in the correct frame of OSC control
        # The vla action will be transfered to delta action control
        vla_action = assign_control_action(
            env,
            vla_action,
            obs,
        )
        start_time = time.time()

        if dynamic_simulation:
            # Dynamically simulate the robot
            obs, _, _, _ = env.step(vla_action)
        else:
            P.reset_to(env, {"states": states[i]})
            obs = env._get_observations(force_update=True)

        end_time = time.time()
        if i % 100 == 0:
            print(
                f"Take {end_time-start_time}s ({1/(end_time-start_time)}fps) to do env.step()"
            )
        obs = add_img_to_obs(env, obs, camera_height, camera_width)

        # on-screen render
        if render:
            if env.viewer is None:
                env.initialize_renderer()

        rgb_images = []
        for cam_name in camera_names:
            if image_process_mode == "resize":
                img = Image.fromarray(obs[cam_name])
                img = img.resize((480, 300))
                img = img.resize((224, 224))
            else:
                img = crop_image(obs[cam_name])
            rgb_images.append(img)

        if write_video:
            if video_count % video_skip == 0:
                # Use the previously processed images for video writing
                video_img = np.concatenate(
                    rgb_images, axis=1  # Concatenate RGB images horizontally
                )
                video_writer.append_data(video_img)

        video_count += 1
    

    metadata_array = {}
    for key, value in env.actions_meta.items():
        if type(value[0]) == tuple:
            metadata_array[key] = np.stack([np.array(val) for val in value])
        else:
            metadata_array[key] = np.stack([np.array(val) for val in value])
    
    last_metadata = metadata_array # copy.deepcopy(env.actions_meta)
    if render:
        env.viewer.close()
        env.viewer = None

    # plot the debug plotting
    try:
        debug_plot(
            vla_action_list, gt_action_list, obs_list, infer_timestep_list, save_name
        )
    except Exception:
        print("Exception in the code when plotting...")
        traceback.print_exc(file=sys.stdout)

    result = {
        "success": success,
        "obj_up_once": obj_up_once,
        "metadata": last_metadata
    }
    return result


def path_change(source_path, xml, target):
    """Function to change XML path in dataset from custom path to your path"""
    import re

    match = re.search(rf"\b{source_path}\b", xml, re.IGNORECASE)
    print(f"path_change match: {match}")

    xml = re.sub(
        rf"\b{source_path}\b",
        f"{target}",
        xml,
        flags=re.IGNORECASE,
    )
    return xml


def download_and_select_ckpt(args):
    # TODO: maybe set this as the attribute of a class
    original_vla_path = args.vla_path  # Save original path to check if it was S3
    if args.vla_path.startswith("s3://"):
        def extract_params(ckpt_name):
            full_ckpt_name = ckpt_name
            ckpt_name = ckpt_name.split("/")[-1]
            # format is step{step}_0.{loss}
            step = int(ckpt_name.split("_")[0][4:])
            loss = float(ckpt_name.split("_")[1].split(".pt")[0])
            return step, loss, full_ckpt_name

        # Set up s3
        s3_bucket = os.environ.get('CHECKPOINT_S3_BUCKET', 'fair-robotics-dss-2--use2-az3--x-s3')
        s3_prefix = os.environ.get('CHECKPOINT_S3_PREFIX', 'vla_checkpoints')
        print(f"S3 bucket is: {s3_bucket}.")
        s3_client = init_s3_client()
        print("Connected to client", s3_client)
        # Remove s3:// prefix and add s3_prefix
        s3_path = args.vla_path[5:]  # Remove "s3://" prefix
        s3_path = s3_path[len("fair-robotics-dss-2--use2-az3--x-s3/"):]

        # Download the checkpoint
        if "*" in s3_path:
            vla_path_red = "/".join(s3_path.split("/")[:-1])
            s3_search_path = f"{vla_path_red}/"

            all_ckpts = list_s3_checkpoints(s3_client, s3_bucket, s3_search_path)
            all_ckpts = [ct["key"] for ct in all_ckpts]
            ckpts_and_step = [extract_params(ckpt_name) for ckpt_name in all_ckpts]
            sorted_step = sorted(ckpts_and_step, key=lambda x: (x[0], x[1]))
            sorted_loss = sorted(ckpts_and_step, key=lambda x: (x[1], x[0]))
            if args.latest_or_lowest_loss_checkpoint == "latest":
                ckpt_interest = sorted_step[-1]
            elif args.latest_or_lowest_loss_checkpoint == "lowest_loss":
                ckpt_interest = sorted_loss[0][-1]
            elif "/" in args.latest_or_lowest_loss_checkpoint:
                num, denom = map(int, args.latest_or_lowest_loss_checkpoint.split("/"))
                ind_ckpt = int(num / denom * len(ckpts_and_step))
                ind_ckpt = min(ind_ckpt, len(sorted_step) - 1)
                ckpt_interest = sorted_step[ind_ckpt][-1]
            else:
                raise ValueError
        else:
            ckpt_interest = f"{s3_path}"

        # Store checkpoints in robot-skills-sim directory (3 levels up from eval_vla.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        robot_skills_dir = os.path.abspath(os.path.join(script_dir, *([os.pardir] * 4)))
        local_cache_dir = os.path.join(robot_skills_dir, "local_checkpoints")
        download_s3_checkpoint(s3_client, s3_bucket, ckpt_interest, local_cache_dir=local_cache_dir)
        args.vla_path = f"{local_cache_dir}/{ckpt_interest}"
        chmod_with_parents(args.vla_path)
        
    else:
        if "*" in args.vla_path:
            # Use the latest checkpoint
            get_file_name = glob.glob(args.vla_path)
            if args.latest_or_lowest_loss_checkpoint == "latest":
                args.vla_path = sorted(get_file_name, key=os.path.getctime)[-1]
            elif args.latest_or_lowest_loss_checkpoint == "lowest_loss":
                eval_loss = [float(v.split("_")[-1].split(".pt")[0]) for v in get_file_name]
                index = eval_loss.index(min(eval_loss))
                args.vla_path = get_file_name[index]
            elif "/" in args.latest_or_lowest_loss_checkpoint:
                sorted_step = sorted(get_file_name, key=os.path.getctime)
                num, denom = map(int, args.latest_or_lowest_loss_checkpoint.split("/"))
                ind_ckpt = int(num / denom * len(sorted_step))
                ind_ckpt = min(ind_ckpt, len(sorted_step) - 1)
                ckpt_interest = sorted_step[ind_ckpt][-1]
                args.vla_path = ckpt_interest
            else:
                raise ValueError
    args.video_path = f"{args.video_path}/{args.vla_path}/"
    print(f"Loading ckpt {args.vla_path}")
    return original_vla_path  # Return original path to track if it was S3


def playback_dataset(args, index_range, batch_i=0, cuda_device="cuda:0", conn=None):
    print(f"Prcoessing the index of demo: {index_range}")
    sys.stdout.flush()

    # some arg checking
    write_video = args.render is not True
    if args.video_path is None:
        args.video_path = "tmp/debug_vla_action.mp4"
    assert not (args.render and write_video)  # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = P.get_env_metadata_from_dataset(
            dataset_path=args.dataset, backup_dataset_path=args.backup_dataset_path
        )
        args.render_image_names = "robot0_robotview"

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    env = None

    # Get env meta data
    env_meta = P.get_env_metadata_from_dataset(
        dataset_path=args.dataset, backup_dataset_path=args.backup_dataset_path
    )
    if args.use_abs_actions:
        # Aabsolute action space
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
        env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"][
            "input_type"
        ] = "absolute"
        env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"][
            "input_ref_frame"
        ] = "right_base"

    # OSC controller parameters
    # KP value
    if args.kp is not None:
        env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"][
            "kp"
        ] = args.kp

    if args.control_freq is not None:
        try:
            env_meta["env_kwargs"]["control_freq"] = args.control_freq
        except Exception as e:
            print(f"No control freq in osc controller: {e}")

    # Damping ratio
    if args.damping_ratio is not None:
        env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"][
            "damping_ratio"
        ] = args.damping_ratio

    # Output min and max
    default_output_man = env_meta["env_kwargs"]["controller_configs"]["body_parts"][
        "right"
    ]["output_max"]
    if args.output_extreme_pos is not None:
        default_output_man[0:3] = [args.output_extreme_pos] * 3
    if args.output_extreme_rot is not None:
        default_output_man[0:3] = [args.output_extreme_rot] * 3
    env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"][
        "output_max"
    ] = default_output_man
    env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"][
        "output_min"
    ] = [-v for v in default_output_man]

    print(f'Right arm controller: {env_meta["env_kwargs"]["controller_configs"]}')

    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = write_video
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_depths"] = False
    if args.control_freq is not None:
        env_kwargs["control_freq"] = args.control_freq

    # print(f'Use control freq of {env_kwargs["control_freq"]}')

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

    # use only test split
    if args.test_split_dir is not None:
        with open(args.test_split_dir, "r") as file:
            train_test_split = json.load(file)
        test_split = train_test_split["test"]
        test_split = [
            int(file.split("/")[-1].split("-")[1].split(".")[0]) for file in test_split
        ]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    # To load the vla
    main_config = {
        "cond_steps": 1,
        "horizon_steps": 100,
        "use_flex": False,
        "load_depth": False,
        "distribute_model_in_gpus": False,
        "num_images_per_step": 1,
    }



    print(colored("Loading VLA ckpt...", "yellow"))
    sys.stdout.flush()
    vla_skill, vla_processor, vla_config, training_stats = load_vla_skill(
        args.vla_path, main_config, cuda_device
    )
    print(
        colored(
            "Done with loading VLA ckpt -- printing path and config of VLA...",
            "yellow",
        )
    )
    print(f"Target vla checkpoint: {args.vla_path}")
    print(f"Vla_config: {vla_config}")

    if args.depoly_aciton_interval == -1:
        args.depoly_aciton_interval = vla_config.horizon_steps
        print(f"Use the depoly action interval value of {args.depoly_aciton_interval}")

    print(f"Total demo: {len(demos)}")

    append_save_file_name = ""
    if args.dynamic_simulation:
        append_save_file_name += "online"
    else:
        append_save_file_name += "offline"

    result_list = []

    for ind in range(len(demos)):
        if args.test_split_dir is not None:
            in_split = True if ind in test_split else False
        else:
            in_split = True

        if index_range[0] <= ind and ind <= index_range[1] and in_split:
            ep = demos[ind]

            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
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
                actions = f["data/{}/abs_action".format(ep)][()]

            # Create the video writer
            video_writer = None
            if write_video:
                # Create the dir if it does not exist
                os.makedirs(args.video_path, exist_ok=True)
                chmod_with_parents(args.video_path)
                video_writer = imageio.get_writer(
                    f"{args.video_path}/{append_save_file_name}_eval_video_{batch_i}_id{ind}.mp4",
                    fps=20,
                )

            result = {
                "success": False,
                "obj_up_once": False,
            }

            # Use VLA to command the robot
            try:
                result = run_vla_trajectory_with_env(
                    env=env,
                    initial_state=initial_state,
                    render=args.render,
                    states=states[0 :: args.downsample_factor],
                    actions=actions[0 :: args.downsample_factor],
                    video_writer=video_writer,
                    video_skip=args.video_skip,
                    camera_names=args.render_image_names,
                    verbose=args.verbose,
                    camera_height=args.camera_height,
                    camera_width=args.camera_width,
                    vla_path=args.vla_path,
                    max_steps=args.max_steps,
                    depoly_aciton_interval=args.depoly_aciton_interval,
                    save_name=f"{args.video_path}{append_save_file_name}_eval_img_{batch_i}_id{ind}.jpg",
                    vla_skill=vla_skill,
                    vla_processor=vla_processor,
                    vla_config=vla_config,
                    training_stats=training_stats,
                    cuda_device=cuda_device,
                    image_process_mode="resize"
                    if vla_config.data.train.get("resize_rgb", True)
                    else "crop",
                    dynamic_simulation=args.dynamic_simulation,
                    pad_zero_dim=args.pad_zero_dim,
                    move_obj_xyz=args.move_obj_xyz,
                    training_stats_key=args.training_stats_key,
                    instruction=args.instruction,
                )
            except Exception:
                print("Exception in running run_vla_trajectory_with_env()...")
                traceback.print_exc(file=sys.stdout)

            print(
                f'success/object up of eval_video_{batch_i}_id{ind}.mp4: {result["success"]} / {result["obj_up_once"]}'
            )
            result["batch_i"] = batch_i
            result["ind"] = ind

            result_list.append({key: val for key, val in result.items() if "metadata" not in key})

            if write_video:
                out_file = f"{args.video_path}{append_save_file_name}_eval_video_{batch_i}_id{ind}.pkl"
                with open(out_file, "wb+") as fw:
                    pickle.dump(result, fw)
                print(
                    colored(
                        f"Saved video to {args.video_path}{append_save_file_name}_eval_video_{batch_i}_id{ind}.mp4",
                        "green",
                    )
                )
                video_writer.close()
                sys.stdout.flush()

    f.close()

    if env is not None:
        env.close()

    if conn is not None:
        conn.send(result_list)
        return
    else:
        return result_list


def parse_array(s):
    return np.fromstring(s, sep=",")


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
    parser.add_argument(
        "--vla_path",
        type=str,
        help="path to vla",
    )
    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use_actions",
        action="store_true",
        help="use open-loop action playback instead of loading sim states",
    )

    # Playback stored dataset absolute actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use_abs_actions",
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
        default="tmp/",
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
            "gripper0_right_right_eye_in_hand",  # wrist cam
            "robot0_robotview",  # torso cam
            # "robot0_robotview_2", # head cam
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
        help="log additional information",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=600,
        help="(optional, for offscreen rendering) height of image observations. 600 is the one used in the real murp",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=960,
        help="(optional, for offscreen rendering) width of image observations. 960 is the one used in the real murp",
    )

    parser.add_argument(
        "--kp",
        type=float,
        default=None,
        help="kp value for the osc controller to correct the error 1.0",
    )

    parser.add_argument(
        "--output_extreme_pos",
        type=float,
        default=None,
        help="output_extreme_pos",
    )

    parser.add_argument(
        "--output_extreme_rot",
        type=float,
        default=None,
        help="output_extreme_rot",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=2500,
        help="The max steps for the env steps",
    )

    parser.add_argument(
        "--damping_ratio",
        type=float,
        default=None,
        help="A dimensionless parameter that describes how quickly an oscillation in a system decays",
    )

    parser.add_argument(
        "--depoly_aciton_interval",
        type=int,
        default=-1,
        help="depoly_aciton_interval",
    )

    parser.add_argument(
        "--num_process",
        type=int,
        default=1,
        help="The number of processes to use to run the evaluation",
    )

    parser.add_argument(
        "--num_gpu",
        type=int,
        default=1,
        help="The number of GPUs provided to load the VLA checkpoint for multi-process eval",
    )

    parser.add_argument(
        "--control_freq",
        type=float,
        default=10.0,
        help="The control frequency of the robot",
    )

    parser.add_argument(
        "--image_process_mode",
        type=str,
        default="resize",
        help="resize or crop images",
    )

    parser.add_argument(
        "--target_eval_index",
        type=int,
        default=None,
        help="Target eval index for the demo id. It is only used when the num_process=1 for the debugging purpose of the target demo id",
    )

    parser.add_argument(
        "--test_split_dir",
        type=str,
        default=None,
        help="use test split if it is provided",
    )

    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=1,
        help="Downsampling factor for the action/proprioception",
    )

    parser.add_argument(
        "--pad_zero_dim",
        type=int,
        default=0,
        help="The number of zeros to pad for the proprioception",
    )

    parser.add_argument(
        "--src_path",
        type=str,
        default=None,
        help="(optional) Change xml paths in hdf5",
    )

    parser.add_argument(
        "--move_obj_xyz",
        type=parse_array,
        default=None,
        help="(optional) If we want to move the target object for testing VLA",
    )

    parser.add_argument(
        "--dynamic_simulation",
        action="store_true",
        help="if want to dynamcially simulate the robot (online eval mode) or just check the action prediction vs the ground-truth (offline eval mode)",
    )

    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
        help="(optional) Change xml paths in hdf5",
    )

    parser.add_argument(
        "--backup_dataset_path",
        type=str,
        default=None,
        help="(optional) In case env_args attribute is missing",
    )

    parser.add_argument(
        "--latest_or_lowest_loss_checkpoint",
        type=str,
        default="latest",
        # choices=["latest", "lowest_loss"],
        help="Load the latest checkpoint or use the lowest eval loss one",
    )

    parser.add_argument(
        "--training_stats_key",
        type=str,
        default=None,
        help="Useful when the checkpoint is trained with multiple datasets -- we might want to specify the key of stats",
    )

    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="(optional) If you want to provide instruction by yourself",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_playback_args()
    start_time = time.time()
    # Create a directory if it doesn't exist
    print(f"Create a video path {args.video_path}....")
    if not os.path.exists(args.video_path):
        os.makedirs(args.video_path)
    chmod_with_parents(args.video_path)
                

    # Get the size of the demo trajs
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

    # Sort the demos index
    inds = np.argsort([int(elem.split("_")[-1]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    original_vla_path = download_and_select_ckpt(args)

    # Create status file for this checkpoint in the launch directory
    ckpt_name_safe = args.vla_path.replace("/", "_").replace(":", "_")
    launch_dir = os.environ.get('LAUNCH_DIR', os.environ.get('SLURM_SUBMIT_DIR', os.getcwd()))
    ckpt_index = os.environ.get('CKPT_INDEX', 'unknown')
    status_file = os.path.join(launch_dir, f"ckpt_{ckpt_index}_status.txt")
    
    # Write initial running status
    with open(status_file, "w") as f:
        f.write(f"RUNNING\n")
        f.write(f"Checkpoint index: {ckpt_index}\n")
        f.write(f"Checkpoint path: {args.vla_path}\n")
        f.write(f"Results path: {args.video_path}\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.num_process == 1:
        # Support the num_process=1 case
        if args.target_eval_index is not None:
            index_range = [args.target_eval_index, args.target_eval_index]
        else:
            index_range = [0, len(demos) - 1]
        result_list = playback_dataset(args, index_range)
    else:
        _start_id = 0
        _end_id = len(demos) - 1
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

        cuda_devices = []

        for i in range(len(chunked_index)):
            cuda_devices.append(f"cuda:{i%args.num_gpu}")

        print(f"chunked_index: {chunked_index}...")
        print(f"cuda_devices: {cuda_devices}...")

        # Define the processor
        mp_ctx = mp.get_context("forkserver")
        proc_infos = []

        for batch_i, index_range in enumerate(chunked_index):
            parent_conn, child_conn = mp_ctx.Pipe()
            proc_args = (
                args,
                index_range,
                batch_i,
                cuda_devices[batch_i],
                child_conn,
            )
            p = mp_ctx.Process(target=playback_dataset, args=proc_args)
            p.start()
            proc_infos.append((parent_conn, p))

        # Get back info (blocking call)
        for conn, proc in proc_infos:
            proc.join()

        result_list = []
        for conn, proc in proc_infos:
            result_list += conn.recv()

    total_eval = len(result_list)
    print(f"args: {args}")
    print(
        f"Total received trajs: {total_eval}/{args.n}, took {(time.time()-start_time)/60} mins"
    )
    # breakpoint()
    # store list in a pickle
    with open(
        os.path.join(args.video_path, f"eval_result_list.pkl"), "wb+"
    ) as f:
        pickle.dump(result_list, f)
    success_list = [result["success"] for result in result_list]
    print(f"Success performance {np.mean(success_list)}: {success_list}")

    obj_up_list = [result["obj_up_once"] for result in result_list]
    print(f"Object up performance {np.mean(obj_up_list)}: {obj_up_list}")

    # Write average stats to text file
    avg_success = np.mean(success_list)
    avg_obj_up = np.mean(obj_up_list)
    
    stats_text = f"""Evaluation Results Summary
==========================
Checkpoint: {args.vla_path}
Total trajectories: {total_eval}
Evaluation time: {(time.time()-start_time)/60:.2f} mins

Performance Metrics:
- Success rate: {avg_success:.4f} ({np.sum(success_list)}/{total_eval})
- Object up rate: {avg_obj_up:.4f} ({np.sum(obj_up_list)}/{total_eval})

Individual Results:
Success: {success_list}
Object up: {obj_up_list}
"""
    
    with open(os.path.join(args.video_path, f"eval_stats.txt"), "w") as f:
        f.write(stats_text)
    
    print(f"Stats written to {os.path.join(args.video_path, 'eval_stats.txt')}")

    # Update status file with completion and results
    with open(status_file, "w") as f:
        f.write(f"DONE\n")
        f.write(f"Checkpoint index: {ckpt_index}\n")
        f.write(f"Checkpoint path: {args.vla_path}\n")
        f.write(f"Results path: {args.video_path}\n")
        f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total trajectories: {total_eval}\n")
        f.write(f"Success rate: {avg_success:.4f} ({np.sum(success_list)}/{total_eval})\n")
        f.write(f"Object up rate: {avg_obj_up:.4f} ({np.sum(obj_up_list)}/{total_eval})\n")
        f.write(f"Evaluation time: {(time.time()-start_time)/60:.2f} mins\n")

    # Clean up downloaded checkpoint if it was downloaded from S3
    if original_vla_path.startswith("s3://") and os.path.exists(args.vla_path):
        try:
            os.remove(args.vla_path)
            print(f"Cleaned up downloaded checkpoint: {args.vla_path}")
        except Exception as e:
            print(f"Failed to clean up checkpoint {args.vla_path}: {e}")

# CLI example to run the VLA online eval
# python eval_vla.py ---vla_path=/checkpoint/siro/jimmytyyang/vla_ckpt/step345000_golden_checkpoint.pt --dataset=/checkpoint/siro/jimmytyyang/robot-skills-sim-gen-large-scale/0801-episode-clutter-object-combined_demo_0.hdf5 --max_steps=1500 --video_path=/checkpoint/siro/jimmytyyang/vla_ckpt_eval_plot/clutter_0809/  --use_abs_actions --n 1 --target_path home/jimmytyyang/research/robot-skills-sim/robocasa-murp/ --dynamic_simulation

# CLI example to run the VLA offline eval
# python eval_vla.py --vla_path=/checkpoint/siro/jimmytyyang/vla_ckpt/step345000_golden_checkpoint.pt --dataset=/checkpoint/siro/jimmytyyang/robot-skills-sim-gen-large-scale/0801-episode-clutter-object-combined_demo_0.hdf5 --max_steps=1500 --video_path=/checkpoint/siro/jimmytyyang/vla_ckpt_eval_plot/clutter_0809/  --use_abs_actions --n 1 --target_path home/jimmytyyang/research/robot-skills-sim/robocasa-murp/
