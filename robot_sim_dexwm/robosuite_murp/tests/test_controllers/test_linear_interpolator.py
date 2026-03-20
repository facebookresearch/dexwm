# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import argparse
import json
import os

import numpy as np

import robosuite as suite
import robosuite.controllers.composite.composite_controller_factory as composite_controller_factory
import robosuite.utils.transform_utils as T

# Define the threshold locations, delta values, and ratio #

# Translation trajectory
pos_y_threshold = 0.1
delta_pos_y = 0.01
pos_action_osc = [0, delta_pos_y * 40, 0]
pos_action_ik = [0, delta_pos_y, 0]

# Rotation trajectory
rot_r_threshold = np.pi / 2
delta_rot_r = 0.01
rot_action_osc = [delta_rot_r * 40, 0, 0]
rot_action_ik = [delta_rot_r * 5, 0, 0]

# Concatenated thresholds and corresponding indexes (y = 1 in x,y,z; roll = 0 in r,p,y)
thresholds = [pos_y_threshold, rot_r_threshold]
indexes = [1, 0]

# Threshold ratio
min_ratio = 1.10

# Define arguments for this test
parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true", help="Whether to render tests or run headless")
args = parser.parse_args()

# Setup printing options for numbers
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


# function to run the actual sim in order to receive summed absolute delta torques
def step(env, action, current_torques):
    env.timestep += 1
    policy_step = True
    summed_abs_delta_torques = np.zeros(7)

    for i in range(int(env.control_timestep / env.model_timestep)):
        env.sim.forward()
        env._pre_action(action, policy_step)
        last_torques = current_torques
        current_torques = env.robots[0].composite_controller.part_controllers["right"].torques
        summed_abs_delta_torques += np.abs(current_torques - last_torques)
        env.sim.step()
        policy_step = False

    env.cur_time += env.control_timestep
    out = env._post_action(action)
    return out, summed_abs_delta_torques, current_torques


# Running the actual test #
def test_linear_interpolator():

    for controller_name in [None, "BASIC"]:  # None corresponds to robot's default controller

        for traj in ["pos", "ori"]:

            # Define counter to increment timesteps and torques for each trajectory
            timesteps = [0, 0]
            summed_abs_delta_torques = [np.zeros(7), np.zeros(7)]

            for interpolator in [None, "linear"]:
                # Define numpy seed so we guarantee consistent starting pos / ori for each trajectory
                np.random.seed(3)

                # load a composite controller
                controller_config = composite_controller_factory.load_composite_controller_config(
                    controller=controller_name,
                    robot="Sawyer",
                )

                # Now, create a test env for testing the controller on
                env = suite.make(
                    "Lift",
                    robots="Sawyer",
                    has_renderer=args.render,  # by default, don't use on-screen renderer for visual validation
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    horizon=10000,
                    control_freq=20,
                    controller_configs=controller_config,
                )

                # Reset the environment
                env.reset()

                # Hardcode the starting position for sawyer
                init_qpos = [-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628]
                env.robots[0].set_robot_joint_positions(init_qpos)
                env.robots[0].composite_controller.part_controllers["right"].update_initial_joints(init_qpos)
                env.robots[0].composite_controller.part_controllers["right"].reset_goal()

                # Notify user a new trajectory is beginning
                print(
                    "\nTesting controller {} with trajectory {} and interpolator={}...".format(
                        controller_name, traj, interpolator
                    )
                )

                # If rendering, set controller to front view to get best angle for viewing robot movements
                if args.render:
                    env.viewer.set_camera(camera_id=0)

                # Keep track of state of robot eef (pos, ori (euler)) and torques
                current_torques = np.zeros(7)
                initial_state = [env.robots[0]._hand_pos["right"], T.mat2quat(env.robots[0]._hand_orn["right"])]
                dstate = [
                    env.robots[0]._hand_pos["right"] - initial_state[0],
                    T.mat2euler(
                        T.quat2mat(T.quat_distance(T.mat2quat(env.robots[0]._hand_orn["right"]), initial_state[1]))
                    ),
                ]

                # Define the uniform trajectory action
                if traj == "pos":
                    pos_act = pos_action_osc
                    rot_act = np.zeros(3)
                else:
                    pos_act = np.zeros(3)
                    rot_act = rot_action_osc

                # Compose the action
                action = np.concatenate([pos_act, rot_act, [0]])

                # Determine which trajectory we're executing
                k = 0 if traj == "pos" else 1
                j = 0 if not interpolator else 1

                # Run trajectory until the threshold condition is met
                while abs(dstate[k][indexes[k]]) < abs(thresholds[k]):
                    _, summed_torques, current_torques = step(env, action, current_torques)
                    if args.render:
                        env.render()

                    # Update torques, timestep count, and state
                    summed_abs_delta_torques[j] += summed_torques
                    timesteps[j] += 1
                    dstate = [
                        env.robots[0]._hand_pos["right"] - initial_state[0],
                        T.mat2euler(
                            T.quat2mat(T.quat_distance(T.mat2quat(env.robots[0]._hand_orn["right"]), initial_state[1]))
                        ),
                    ]

                # When finished, print out the timestep results
                print(
                    "Completed trajectory. Avg per-step absolute delta torques: {}".format(
                        summed_abs_delta_torques[j] / timesteps[j]
                    )
                )

                # Shut down this env before starting the next test
                env.close()

    # Tests completed!
    print()
    print("-" * 80)
    print("All linear interpolator testing completed.\n")


if __name__ == "__main__":
    test_linear_interpolator()
