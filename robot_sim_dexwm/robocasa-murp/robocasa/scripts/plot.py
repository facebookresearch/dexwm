import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


JOINTS_MAPPING = {
    0: "j0",
    1: "j1",
    2: "j2",
    3: "j3",
    4: "j4",
    5: "j5",
    6: "j6",
}

EE_NAME_MAPPING = {
    0: "ee_pos_x",
    1: "ee_pos_y",
    2: "ee_pos_z",
    3: "ee_rot_x",
    4: "ee_rot_y",
    5: "ee_rot_z",
    6: "ee_rot_w",
}


def plot_image(gt, pred, time_i, inference_i, save_name, plot_inference=True):
    # Create a figure and a grid of subplots
    num_sub_plot = len(EE_NAME_MAPPING)
    # num_sub_plot = len(JOINTS_MAPPING)
    # figure size is iin (width, height)
    fig, axes = plt.subplots(
        num_sub_plot, figsize=(6, 40), height_ratios=[1] * num_sub_plot
    )  # 7 rows, 1 columns

    # Display each image in a subplot
    for i in range(num_sub_plot):
        axes[i].plot(time_i, gt[:, i].tolist(), color="green", label="Real")
        axes[i].plot(time_i, pred[:, i].tolist(), color="blue", label="Sim")
        if plot_inference:
            axes[i].plot(
                inference_i,
                pred[inference_i, i].tolist(),
                color="red",
                marker="o",
                label="Inference Point",
                linestyle="None",
            )
        axes[i].set_title(f"{EE_NAME_MAPPING[i]}")
        # axes[i].set_title(f"{JOINTS_MAPPING[i]}")

    # Show legend
    axes[num_sub_plot - 1].legend()
    plt.tight_layout()
    # fig.set_size_inches(18.5, 10.5, forward=True)

    plt.savefig(save_name)


if __name__ == "__main__":
    # sim_data = np.load(
    #     "/checkpoint/siro/jtruong/repos/murp/core/dataloader/robocasa_eef_dict.npy", allow_pickle=True
    # )
    # real_data = np.load(
    #     "/checkpoint/siro/jtruong/data/sim_robot_data/robocasa_murp/datasets/ep_0_robocasa_ee_pose.npy", allow_pickle=True
    # )
    real_data = np.load(
        "./mimicgen_data/PnP_CT/mimicgen_dist_adjusted_seed_cluster_june25_smooth_batch50_proc5/demo/ep_0_robocasa_eef_fk.npy", allow_pickle=True
    )
    # sim_data = np.load('/checkpoint/siro/jtruong/data/sim_robot_data/robocasa_murp/datasets/robocasa_ee_T_base.npy', allow_pickle=True)
    sim_data = np.load('./robocasa_ee_T_right_base_torso_fk.npy', allow_pickle=True)

    
    save_name = "simvreal3.png"
    cv2.imwrite(save_name, np.zeros((100, 100, 3), dtype=np.uint8))
    arm_key = "right_fr3_link8_T_right_base"

    # gt_arm = real_data
    gt_arm = [d[arm_key] for d in real_data]
    # quat_data = np.column_stack([sim_data[:, 6], sim_data[:, 3], sim_data[:, 4], sim_data[:, 5], ])
    # pred_arm = np.concatenate([sim_data[:, :3], quat_data], axis=1)
    pred_arm = sim_data


    inference_i = None
    # concatenate all the data
    # real_data_len, _ = gt_arm.shape
    real_data_len = len(gt_arm)
    sim_data_len = len(pred_arm)
    data_len = np.min([real_data_len, sim_data_len])
    gt = np.concatenate([np.array(gt_arm)[:data_len, :], np.zeros((data_len, 16))], axis=1)
    pred = np.concatenate([np.array(pred_arm)[:data_len, :], np.zeros((data_len, 160))], axis=1)

    time_i = np.arange(len(gt))
    plot_image(gt, pred, time_i, inference_i, save_name, plot_inference=False)
