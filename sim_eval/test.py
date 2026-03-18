import argparse
import json
import time
from collections import OrderedDict
import cv2
import os
import sys
import numpy as np
import torch
import h5py
import copy
import yaml
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import robosuite
from termcolor import colored
from scipy.spatial.transform import Rotation as R

import robocasa

from robosuite.wrappers import DataCollectionWrapper
import torch.distributed as dist

import robocasa.scripts.playback_utils as P
from sim_eval.hand_cam_utils import estimate_object_pos
from sim_wrappers import EvaluationWrapper


def waypoint_control(input_ac_dict, right_abs_original, env, active_robot, dexwm_controller,
                    sub_waypoints = 100, render=True, max_fr=None, start=None, idx=0, aux_movement=False):

    waypoint_action_dict = copy.deepcopy(input_ac_dict)

    right_delta_interp = waypoint_action_dict['right_delta']/sub_waypoints
    new_gripper_angle = waypoint_action_dict['right_gripper']
    right_abs_curr = right_abs_original
    old_gripper_angle = active_robot.sim.data.qpos[dexwm_controller.joint_indices][7:]

    right_gripper_interp = (new_gripper_angle-old_gripper_angle)/sub_waypoints

    success_count = 0
    for interp in range(sub_waypoints):
        action_dict = copy.deepcopy(waypoint_action_dict)
        right_abs_next = right_abs_original + right_delta_interp*(interp+1)   # trajectory to follow
        right_delta_curr = right_abs_next - right_abs_curr
        action_dict['right_abs'] = right_abs_next
        action_dict['right_delta'] = right_delta_curr

        # set arm actions
        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type   # delta is used
            # print(controller_input_type)
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        env_action = active_robot.create_action_vector(action_dict)

        # Run environment step
        obs, success = env.step(env_action)
        # only storing the value of success on rank 0
        success_tensor = torch.tensor([success], dtype=torch.uint8, device='cuda') if dexwm_controller.rank == 0 else torch.tensor([0], dtype=torch.uint8, device='cuda')
        dist.broadcast(success_tensor, src=0)
        success = bool(success_tensor.item())

        if not aux_movement and idx==dexwm_controller.pred_steps-1:
            if success:
                success_count+=1
                if success_count>=10:
                    return obs, success, interp
            else:
                success_count=0

        curr_img = obs['robot0_robotview_2_image'][::-1]
        if dexwm_controller.rank==0:
            dexwm_controller.plot_images(curr_img, None, f'{idx}_{interp+1}', use_cv2=True)
        joint_pos = active_robot.sim.data.qpos[dexwm_controller.joint_indices]
        right_abs_curr = dexwm_controller.get_ee_pose(joint_pos[:7], active_robot.sim)

        if render:
            env.render()

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    return obs, success, interp


def test_trajectory(
    env,
    dexwm_controller,
    goal_img,
    arm,
    env_configuration,
    mirror_actions,
    demo_name,
    render=True,
    max_fr=None,
    print_info=True,
    gt_joints=None,
    gt_actions=None,
    task_name='grasp'
):

    if render:
        # ID = 2 always corresponds to agentview
        env.render()


    nonzero_ac_seen = False

    if task_name != 'place':    # this zero action may cause environment initialization issues with place task
        zero_action = np.zeros(env.action_dim)
        for _ in range(1):
            # do a dummy step thru base env to initalize things, but don't record the step
            if isinstance(env, DataCollectionWrapper):
                env.env.step(zero_action)
            else:
                env.step(zero_action)

    dexwm_controller.set_goal(goal_img, demo_name)
    idx = 0
    obs = env._get_observations(force_update=True)
    joint_indices = [env.robots[0].sim.model.joint_name2id(name) for name in dexwm_controller.joint_names]
    start_joints = env.robots[0].sim.data.qpos[joint_indices]

    while idx<dexwm_controller.pred_steps:
        start = time.time()
        active_robot = env.robots[0]

        # Get the newest action
        input_ac_dict, right_abs_original = dexwm_controller.get_actions(obs, active_robot, env)


        obs, success, interp = waypoint_control(input_ac_dict, right_abs_original, env, active_robot, dexwm_controller,
                                sub_waypoints=100, render=render, max_fr=max_fr, start=start, idx=idx)

        idx+=1

    # cleanup for end of data collection episodes
    env.close()

    return success
