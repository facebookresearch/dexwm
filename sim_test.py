import argparse
import json
import time
from collections import OrderedDict
import cv2
import os
import sys
import numpy as np
import h5py
import copy
import yaml
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

import robosuite
from termcolor import colored
from scipy.spatial.transform import Rotation as R

import robocasa
from sim_eval.distributed_controller import DexWMControllerDist

import robocasa.scripts.playback_utils as P
from sim_eval.test import test_trajectory as test_dexwm
from sim_wrappers import EvaluationWrapper

def get_indices(waypoint_strings, task_name, start0):
    if task_name=='reach':
        start_ind = 0
        for i in range(waypoint_strings.shape[0]):
            if waypoint_strings[i]==b's1-w1':
                target_ind = i-1
                break
    elif task_name=='grasp':
        for i in range(waypoint_strings.shape[0]):
            if waypoint_strings[i]==b's1-w1':
                start_ind = i
                break
        if start0:
            print('Starting from 0')
            start_ind = 0
        for i in range(waypoint_strings.shape[0]):
            if waypoint_strings[i]==b's2-w1':
                target_ind = i-1
                break
    elif task_name=='place':
        for i in range(waypoint_strings.shape[0]):
            if waypoint_strings[i]==b's2-w1':
                start_ind = min(i+20, len(waypoint_strings)-1)
                break
        for i in range(waypoint_strings.shape[0]):
            target_ind = None
            if waypoint_strings[i]==b's3-w1':
                target_ind = i-1
                break
        if target_ind is None:
            target_ind = len(waypoint_strings)-1

    return start_ind, target_ind

def main(args):
    fname = args.config
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        params['checkpoint'] = args.checkpoint

    if args.method == 'dexwm':
        # Controller initializes distributed - all GPUs across all nodes work together
        controller = DexWMControllerDist(params, job_dir=args.job_dir, opt_steps=args.opt_steps, num_samples=args.num_samples, topk=args.topk, batch_size=args.batch_size, task_name=args.task_name, pred_steps=args.pred_steps, use_loss_ee=args.use_loss_ee, kp_loss_weight=args.kp_loss_weight, decoder_checkpoint=args.decoder_checkpoint)
        multiprocess = True

        # Get distributed info
        if dist.is_initialized():
            gpu_rank = dist.get_rank()
            world_size = dist.get_world_size()
            print(f"World size: {world_size}, Rank: {gpu_rank}")
        else:
            gpu_rank = 0
            world_size = 1

        # Only rank 0 saves results since all GPUs process episodes together
        success_path = f'{args.job_dir}/res.json'
        if os.path.exists(success_path):
            success_dict = json.load(open(success_path, 'r'))
        else:
            success_dict = {}
    else:
        pass

    # Indices of samples with clear goal observations and well-defined tasks
    indices = [3, 9, 12, 15, 17, 19, 20, 25, 26, 28, 35, 36, 37, 39, 41, 44, 55, 56, 57, 59, 64,
                68, 69, 70, 83, 86, 87, 88, 104, 109, 110, 111, 112, 115, 120, 129, 131, 134, 141,
                145, 146, 149, 151, 152, 155, 157, 158, 160, 163, 168]
    test_demos = json.load(open('utils/split_indices_robocasa_pp.json', 'r'))['test']

    # Filter demos to only those in indices list
    filtered_demos = [(idx, demo_name) for idx, demo_name in enumerate(test_demos) if idx in indices]
    if args.debug:
        # For debugging just run one episode
        filtered_demos = filtered_demos[:1]
        success_dict = {}

    # All GPUs process all episodes together (trajectories are parallelized across GPUs)
    if gpu_rank == 0:
        print(f"Total episodes: {len(filtered_demos)}")
        print(f"Trajectories will be parallelized across all {world_size} GPUs globally")

    for idx, demo_name in filtered_demos:
        [hdf5_file, demo_name] = demo_name
        if demo_name in success_dict:    # when job gets preempted, skip the demos already looked at
            if gpu_rank == 0:
                print(f'Skipping {demo_name}')
            continue
        print('Task name', hdf5_file, demo_name)
        file_loc = os.path.join(args.data_dir,hdf5_file)
        with h5py.File(file_loc) as f:
            ep = demo_name
            states = f["data/{}/states".format(ep)][()]
            waypoint_string = f['data'][ep]['waypoint_generation_stage_string'][:]
            start_id, target_id = get_indices(waypoint_string, args.task_name, args.start0)
            initial_state = dict(states=states[start_id])
            target_state = dict(states=states[target_id])
            env_meta = json.loads(f["data"].attrs["env_args"])
            data_group = f["data"]
            first_key = list(data_group.keys())[0]
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)
            target_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            target_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)
            ep_meta = json.loads(f["data/{}".format(ep)].attrs.get("ep_meta", None))
            len_frames = f["data/{}".format(ep)]['obs']['robot0_robotview_2_image'].shape[0]
            curr_img = f["data/{}".format(ep)]['obs']['robot0_robotview_2_image'][start_id]
            goal_img = f["data/{}".format(ep)]['obs']['robot0_robotview_2_image'][target_id]
            gt_indices = np.linspace(start_id,target_id,3,dtype=np.int32)
            gt_hand_joints = np.array(f["data/{}/obs/robot0_joint_pos".format(ep)])[gt_indices,:7]
            gt_finger_joints = f["data/{}/obs/robot0_right_gripper_qpos".format(ep)][gt_indices]
            gt_joints = np.concatenate([gt_hand_joints, gt_finger_joints], axis=-1)
            gt_images = f["data/{}".format(ep)]['obs']['robot0_robotview_2_image'][gt_indices]
            gt_actions = f["data/{}".format(ep)]['actions'][gt_indices]


        args.renderer = "mjviewer"
        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = args.renderer
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["use_camera_obs"] = True
        env_kwargs["camera_depths"] = args.task_name=='grasp'
        env_kwargs["render_camera"] = "robot0_robotview_2"
        env_kwargs["ignore_done"] = True
        # env_kwargs["camera_heights"] = 600
        # env_kwargs["camera_widths"] = 960
        env_kwargs["camera_heights"] = 1200
        env_kwargs["camera_widths"] = 1920
        env_kwargs['controller_configs']['body_parts']['right']['input_type'] = 'absolute'
        env_kwargs['controller_configs']['body_parts']['left']['input_type'] = 'absolute'

        env_kwargs["control_freq"] = 10
        env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"]["input_ref_frame"] = 'world'

        print(colored(f"Initializing environment...", "yellow"))
        env = robosuite.make(**env_kwargs)

        P.reset_to(env, initial_state)

        if args.task_name=='reach':
            target_env = robosuite.make(**env_kwargs)
            P.reset_to(target_env, target_state)
            body_names_tips = ['robot0_right_hand', 'gripper0_right_link_3.0', 'gripper0_right_link_7.0', 'gripper0_right_link_11.0', 'gripper0_right_link_15.0']
            target_pos = []
            for body_name in body_names_tips:
                body_id = target_env.sim.model.body_name2id(body_name)
                pos = target_env.sim.data.body_xpos[body_id]
                target_pos.append(pos.copy())
            target_pos = np.array(target_pos)
            env = EvaluationWrapper(env,task_name=args.task_name, target_hand_pose=target_pos)

        elif args.task_name=='place':
            target_env = robosuite.make(**env_kwargs)
            P.reset_to(target_env, target_state)
            obj_name_id = 'obj'
            obj = target_env.objects[obj_name_id]
            target_obj_pos = np.array(target_env.sim.data.body_xpos[target_env.obj_body_id[obj.name]])

            env = EvaluationWrapper(env,task_name=args.task_name, target_obj_pos=target_obj_pos)
            obs = env._get_observations(force_update=True)
            curr_img = obs['robot0_robotview_2_image'][::-1]

        elif args.task_name=='grasp':
            env = EvaluationWrapper(env,task_name=args.task_name)


        try:
            if args.method == 'dexwm':
                success = test_dexwm(
                    env,
                    controller,
                    goal_img,
                    "right",
                    "TwoArm",
                    mirror_actions=True,
                    demo_name=demo_name,
                    render=(args.renderer != "mjviewer"),
                    max_fr=30,
                    gt_joints=gt_joints,
                    gt_actions=gt_actions,
                    task_name = args.task_name
                )
            if gpu_rank==0:
                print(f'{demo_name} Success: {success}')
                success_dict[demo_name] = success
                with open(success_path, 'w') as f:
                    json.dump(success_dict, f)

            else:
                pass

            print()
        except KeyboardInterrupt:
            print("\nInterrupted. Saving and exiting.")
            exit(0)

    # Only rank 0 needs to finalize results since all GPUs processed episodes together
    if gpu_rank == 0:
        print(f"Completed processing {len(success_dict)} episodes")
        print(success_path)

        # Calculate and print success rate
        if success_dict:
            success_count = sum(1 for v in success_dict.values() if v)
            success_rate = success_count / len(success_dict) * 100
            print(f"Success rate: {success_count}/{len(success_dict)} ({success_rate:.1f}%)")


if __name__=='__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint to run inference on')
    parser.add_argument('--decoder_checkpoint', type=str, default=None, help='Path to the decoder checkpoint to use')
    parser.add_argument('--job_dir', type=str, required=True, help='Path to the job')
    parser.add_argument('--method', type=str, default='dexwm', help='dexwm or other')
    parser.add_argument('--task_name', default='grasp', type=str, help='choose from reach, grasp, and place')
    parser.add_argument('--start0', default=False, action='store_true')
    parser.add_argument('--use_loss_ee', default=False, action='store_true')
    parser.add_argument('--kp_loss_weight', type=float, required=True)
    parser.add_argument('--pred_steps', type=int, default=3)
    parser.add_argument('--opt_steps', type=int, default=10)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.job_dir, exist_ok=True)
    main(args)
