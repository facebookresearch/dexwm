import os
import argparse
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import robocasa.macros as macros
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils import transform_utils as T
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=int, help="kitchen layout (choose number 0-9)")
    parser.add_argument("--style", type=int, help="kitchen style (choose number 0-11)")
    args = parser.parse_args()

    controller_path = (
        "../robosuite/robosuite/controllers/config/robots/default_tmr_robot.json"
    )
    config = {
        "env_name": "PnPCounterTop",
        "robots": "TMR_ROBOT",
        "controller_configs": load_composite_controller_config(
            controller=controller_path, robot="TMR_ROBOT"
        ),
        "translucent_robot": False,
    }

    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        renderer="mujoco",
        render_camera="robot0_robotview_2",
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_heights=300,
        camera_widths=480,
        camera_names=["robot0_robotview_2"],
        camera_depths=True,
    )

    aux = env.reset()
    env.sim.forward()
    env.robots[0].print_action_info()

    # Target setup
    tar_pos = aux["obj_pos"].copy()
    tar_pos[2] += 0.25
    start_pos = aux["robot0_right_eef_pos"].copy()
    start_pos[2] += 0.25
    fixed_quat = aux["robot0_right_eef_quat_site"]  # Constant orientation

    pose_A = start_pos
    pose_B = tar_pos - np.array([0.1, 0.0, -0.2])
    pose_C = tar_pos

    seg_steps = [600, 600, 600]
    total_steps = sum(seg_steps)

    interpolated_positions = np.zeros((total_steps, 3))
    interpolated_orientations = np.tile(fixed_quat, (total_steps, 1))  # Fixed rotation

    for t in range(total_steps):
        if t < seg_steps[0]:
            alpha = t / seg_steps[0]
            pos = pose_A
        elif t < seg_steps[0] + seg_steps[1]:
            alpha = (t - seg_steps[0]) / seg_steps[1]
            pos = pose_B
        else:
            pos = pose_C
        interpolated_positions[t] = pos

    gripper_state = np.array(
        [
            0.10999999940395355,
            1.190000033378601,
            1.1444294333457947,
            0.50000001192092896,
            0.15000000596046448,
            1.190000033378601,
            1.1700000047683716,
            -0.0019986582919955254,
            0.012606220319867134,
            1.190000033378601,
            0.891339099407196,
            0.5501810997724533,
            1.3960000276565552,
            0.6330000162124634,
            0.6200000047683716,
            0.51999998688697815,
        ]
    )

    site_id = env.sim.model.geom_name2id("robot0_debug_sphere")
    env.sim.forward()

    eef_pos_history = np.zeros((total_steps, 3))
    obtained_eef_pos_history = np.zeros((total_steps, 3))
    imgs = []

    for step in range(total_steps):
        eef_pos = interpolated_positions[step]
        eef_quat = interpolated_orientations[step]
        eef_rpy = R.from_quat(eef_quat).as_euler("xyz")

        action_dict = {
            "right": np.concatenate([eef_pos, eef_rpy]),
            "right_gripper": gripper_state,
        }

        action = env.robots[0].create_action_vector(action_dict)
        aux = env.step(action)
        env.sim.data.geom_xpos[site_id] = pose_C.copy()

        obtained_eef_pos = (
            env.robots[0]
            .composite_controller.joint_action_policy.full_model_data.site(4)
            .xpos
        )

        eef_pos_history[step] = eef_pos
        obtained_eef_pos_history[step] = obtained_eef_pos

        imgs.append(env.sim.render(300, 300))
        imgs.append(aux[0]["robot0_robotview_2_image"])

    # Save video
    video_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        video_path, fourcc, 20.0, (imgs[0].shape[1], imgs[0].shape[0])
    )
    for img in imgs:
        out.write(img[::-1, :, ::-1])
    out.release()

    # Plotting
    steps = np.arange(total_steps)
    plt.figure(figsize=(12, 8))
    plt.plot(steps, eef_pos_history[:, 0], "r--", label="Target X")
    plt.plot(steps, eef_pos_history[:, 1], "g--", label="Target Y")
    plt.plot(steps, eef_pos_history[:, 2], "b--", label="Target Z")

    plt.plot(steps, obtained_eef_pos_history[:, 0], "r-", label="Actual X")
    plt.plot(steps, obtained_eef_pos_history[:, 1], "g-", label="Actual Y")
    plt.plot(steps, obtained_eef_pos_history[:, 2], "b-", label="Actual Z")

    plt.xlabel("Step")
    plt.ylabel("EEF Position (m)")
    plt.title("Target vs Actual EEF Position Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("eef_position_vs_step.png")
    plt.show()
