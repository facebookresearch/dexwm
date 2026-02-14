import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import mujoco
import yaml
import os
import gc
import torch
from models.model import DexWM
import torchvision
from tqdm import tqdm
import sys
import copy
from distributed import init_distributed, cleanup
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.decoder_vit import SimpleViTDecoder
from utils.image_utils import get_keypoints_from_beliefmap
from scipy.optimize import minimize
from train_wm import get_patch_size_from_backbone

import robosuite.utils.transform_utils as T

def world_to_pixel(xyz, intrinsic_matrix, extrinsic_matrix):
    X, Y, Z = xyz
    world_point = np.array([X, Y, Z, 1])
    camera_point = extrinsic_matrix @ world_point
    pixel_homogeneous = intrinsic_matrix @ camera_point[:3]
    u = pixel_homogeneous[0] / pixel_homogeneous[2]
    v = pixel_homogeneous[1] / pixel_homogeneous[2]
    return (u, v)

def gather_outputs(output, device):
    """Gather outputs from ALL GPUs globally (not just within a node)"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gathered_outputs = [torch.zeros_like(output) for _ in range(world_size)]

    # Gather globally across all GPUs
    dist.all_gather(gathered_outputs, output.contiguous())
    gathered_outputs = torch.stack(gathered_outputs, dim=1) # stacking along dimension 1 because this is how the sampler samples data
    gathered_outputs = gathered_outputs.flatten(0,1)
    return gathered_outputs

class DexWMControllerDist:
    def __init__(self, model_config, kp_loss_weight, opt_steps=10, num_samples=2048, topk=10, batch_size=64, job_dir='debug', task_name='grasp', pred_steps=3, use_loss_ee=False, decoder_checkpoint=None):
        self.goal = None

        self.body_names = ['robot0_right_hand',
                            'gripper0_right_link_0.0', 'gripper0_right_link_1.0','gripper0_right_link_2.0', 'gripper0_right_link_3.0',
                            'gripper0_right_link_4.0', 'gripper0_right_link_5.0', 'gripper0_right_link_6.0', 'gripper0_right_link_7.0',
                            'gripper0_right_link_8.0', 'gripper0_right_link_9.0', 'gripper0_right_link_10.0', 'gripper0_right_link_11.0',
                            'gripper0_right_link_12.0', 'gripper0_right_link_13.0', 'gripper0_right_link_14.0', 'gripper0_right_link_15.0']

        self.body_names_tips = ['robot0_right_hand', 'gripper0_right_link_3.0', 'gripper0_right_link_7.0', 'gripper0_right_link_11.0', 'gripper0_right_link_15.0']


        self.joint_names = ['robot0_right_fr3_joint1', 'robot0_right_fr3_joint2', 'robot0_right_fr3_joint3', 'robot0_right_fr3_joint4',
                            'robot0_right_fr3_joint5', 'robot0_right_fr3_joint6', 'robot0_right_fr3_joint7', 'gripper0_right_joint_0.0',
                            'gripper0_right_joint_1.0', 'gripper0_right_joint_2.0', 'gripper0_right_joint_3.0', 'gripper0_right_joint_4.0',
                            'gripper0_right_joint_5.0', 'gripper0_right_joint_6.0', 'gripper0_right_joint_7.0', 'gripper0_right_joint_8.0',
                            'gripper0_right_joint_9.0', 'gripper0_right_joint_10.0', 'gripper0_right_joint_11.0', 'gripper0_right_joint_12.0',
                            'gripper0_right_joint_13.0', 'gripper0_right_joint_14.0', 'gripper0_right_joint_15.0']

        self.T_camera_in_base = np.array([[ -0.0000000,  0.5000000, -0.8660254,  0.212],    # calculated from xml file for MURP
                                            [ -1.0000000, -0.0000000,  0.0000000, 0],
                                            [0.0000000,  0.8660254,  0.5000000, 1.614],
                                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        self.T_optical = np.array([     # for the camera base to camera optical frame
                    [1, 0,  0, 0],
                    [0, -1,  0, 0],
                    [0, 0, -1, 0],
                    [0, 0,  0, 1]
                ])
        self.camera_matrix = np.array([[357.52607777826296,   0.,         480.0        ],
                                        [  0.       ,  357.52607777826296, 300.0        ],
                                        [  0.   ,        0.     ,      1.        ]])

        self.model = self.get_model(model_config)
        _, rank, device, _ = init_distributed()
        # Use the model's encoder_dim for the decoder to handle different encoders (dinov2=1024, siglip2=1152, etc.)
        encoder_dim = self.model.encoder_dim
        self.decoder = SimpleViTDecoder(encoder_dim=encoder_dim, target_resolution=224, patch_size=14)
        # Only load decoder weights if using dinov2/dinov3 encoder (decoder was trained for dinov2)
        if decoder_checkpoint is not None:
            self.decoder.load_decoder(decoder_checkpoint)

        self.decoder = self.decoder.to(device)
        self.device = device
        self.rank = rank
        self.model = self.model.to(device)
        self.model = DDP(self.model, device_ids=[device])
        self.model = torch.compile(self.model)
        self.model.eval()
        self.idx = 0
        self.opt_steps = opt_steps
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.topk = topk
        self.action_dim = 23 # number of robot joints (4 per finger for 4 fingers of Allegro gripper + 7 for the franka arm)
        self.seen_frames = []
        self.pred_steps = pred_steps
        self.fig_dir = f'{job_dir}/Figs'
        self.action_step = 1
        assert (self.action_step <= self.pred_steps)
        self.all_joint_angles = torch.randn(0).to(device)
        self.kp_end_idx = 12
        self.ref_joints = np.load('utils/ref_joint_init.npy')
        indices = np.linspace(0,len(self.ref_joints)-1,self.pred_steps,dtype=int)
        self.ref_joints = self.ref_joints[indices]
        self.ref_orientation = np.array([-0.64704954,  1.34930342, -0.77454947])
        self.task_name = task_name
        self.use_loss_ee = use_loss_ee
        self.kp_loss_weight = kp_loss_weight

    def get_model(self, args):
        checkpoint_path = args['checkpoint']
        num_context = args['data']['num_context']
        self.num_context = num_context
        backbone_name = args['model']['backbone_name']
        hidden_dim = args['model']['hidden_dim']
        action_dim = args['model']['action_dim']
        depth = args['model']['depth']
        num_heads = args['model']['num_heads']
        mlp_ratio = args['model']['mlp_ratio']
        self.img_size = args['data']['img_size']

        self.backbone_name = backbone_name

        patch_size, num_patches = get_patch_size_from_backbone(backbone_name)
        self.patch_size = patch_size

        # Determine patch size and image dimensions based on encoder
        if self.patch_size==14:
            self.img_width = 392
        elif self.patch_size==16:
            self.img_width = 384


        howm = DexWM(backbone_name=backbone_name,
                num_patches=num_patches,
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                is_eval=True,
                num_context = num_context,
                emb_loss_fn=torch.nn.MSELoss(reduction='mean'))

        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))["model"]
        pretrained_dict = checkpoint
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            new_key = k.replace("_orig_mod.", "")  # Remove the "module." prefix
            new_state_dict[new_key] = v
        howm.load_state_dict(new_state_dict)
        return howm

    def get_base_pose(self, obs):
        p_base_in_world = obs['robot0_base_pos']
        q_base_in_world = obs['robot0_base_quat']
        R_base_in_world = R.from_quat(q_base_in_world).as_matrix()
        T_base_in_world = np.eye(4)
        T_base_in_world[:3,:3] = R_base_in_world
        T_base_in_world[:3,3] = p_base_in_world
        return T_base_in_world

    def world_to_cam_frame(self, world_keypoints, obs):
        T_base_in_world = self.get_base_pose(obs)
        T_camera_in_base = self.T_camera_in_base
        T_camera_in_world = T_base_in_world @ T_camera_in_base
        T_world_in_camera = np.linalg.inv(T_camera_in_world)
        all_poses = []
        for key in world_keypoints:
            pos = key
            pos_hom = np.append(pos,1.)[...,None]
            pos_cam = T_world_in_camera @ pos_hom
            pos_cam = self.T_optical@pos_cam
            all_poses.append(pos_cam[:3,0])
        return np.array(all_poses)

    def cam_to_world_frame(self, cam_keypoints, obs):
        T_base_in_world = self.get_base_pose(obs)
        T_camera_in_base = self.T_camera_in_base
        T_camera_in_world = T_base_in_world @ T_camera_in_base
        T_world_in_camera = np.linalg.inv(T_camera_in_world)
        all_poses = []
        for key in cam_keypoints:
            pos = key
            pos_hom = np.append(pos,1.)[...,None]
            pos_world = np.linalg.inv(self.T_optical@T_world_in_camera) @ pos_hom
            all_poses.append(pos_world[:3,0])
        return np.array(all_poses)

    def get_current_kp(self, obs):
        keypoints = obs['robot0_right_gripper_keypoint_pose'].reshape(-1,3)
        hand_keypoint = obs['robot0_right_hand_T_world_pose_mat'].reshape(4,4)[:3,3:].T
        keypoints = keypoints[-16:]
        keypoints = np.concatenate([hand_keypoint, keypoints], axis=0)
        world_keypoints = keypoints[[0,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,9,10,11,12]]  # rearanging to match egodex data format and copying ring finger as proxy to little finger
        cam_keypoints = self.world_to_cam_frame(world_keypoints, obs)
        return cam_keypoints

    def plot_images(self, curr_img, curr_kp, idx, pixels=False, use_cv2=False):
        if use_cv2:
            cv2.imwrite(f'{self.fig_dir}/{self.demo_name}/img_{idx}.png', curr_img[:,:,::-1])
            return
        plt.imshow(curr_img)
        if curr_kp is not None:
            for i, kp in enumerate(curr_kp):
                if not pixels:
                    u,v = world_to_pixel(kp, self.camera_matrix, np.eye(4))
                else:
                    u, v = kp.cpu().numpy()
                    u, v = u/self.img_width*960, v/224*548 + 26
                plt.scatter(u, v)
        plt.axis('off')
        plt.savefig(f'{self.fig_dir}/{self.demo_name}/img_{idx}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    def image_transform(self, img):
        # img, pad = pad_and_resize(img, self.img_size)
        ext_pix = (img.shape[0]-548)//2
        img = img[ext_pix:-ext_pix]
        h, w = img.shape[:2]
        ar = w/h
        img = cv2.resize(img, (int(self.img_size*ar), self.img_size), interpolation=cv2.INTER_LINEAR)

        # Crop for patch size 16 encoders to make width divisible by 16
        if self.patch_size == 16:
            img = img[:, 4:-4]  # Crop 392 -> 384

        img = torchvision.transforms.ToTensor()(img)
        if 'siglip' in self.backbone_name:
            img = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
        else:
            img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img

    def init_mu_sigma(self, obs_0, traj_len, curr_mu):
        n_evals = obs_0.shape[0]
        mu = torch.Tensor(curr_mu).unsqueeze(0).unsqueeze(1).repeat(n_evals, traj_len, 1).numpy()
        sigma = np.ones(mu.shape)
        sigma[:,:,:7] *= (0.3)
        if self.task_name=='place':
            sigma[:,:,7:] *= (0.01)
        else:
            sigma[:,:,7:] *= (0.1)
        mu_ref = torch.Tensor(self.ref_joints[-traj_len:]).unsqueeze(0).repeat(n_evals,1,1).numpy()
        return mu, sigma, mu_ref

    def get_current_joints(self, sim):
        all_joints = sim.data.qpos[self.joint_indices]
        return all_joints

    def get_kp_from_joints(self, joint_angles_samples, sim, obs):
        res = []
        for joint_angles in joint_angles_samples:
            data_copy = mujoco.MjData(sim.model._model)     # creating a copy of the sim to not impact the original sim
            data_copy.qpos[self.joint_indices] = joint_angles
            mujoco.mj_kinematics(sim.model._model, data_copy)
            current_positions = []
            for body_name in self.body_names:
                body_id = sim.model.body_name2id(body_name)
                pos = data_copy.xpos[body_id]
                current_positions.append(pos.copy())
            current_positions = np.array(current_positions)
            current_positions = self.world_to_cam_frame(current_positions, obs)
            current_positions = current_positions[[0,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,9,10,11,12]] # changing order to match our world model
            res.append(current_positions)
        res = np.array(res).reshape(-1,21*3)
        return res

    def plot_all_kp(self, imgs, all_kp, idx, world_kp=True):
        if world_kp:
            all_kp = all_kp.reshape(-1,21,3).cpu().numpy()
        img = self.obs['robot0_robotview_2_image'][::-1]
        fig, axes = plt.subplots(nrows=1, ncols=len(all_kp), figsize=(4 * len(all_kp), 4))
        if len(all_kp)==1:
            axes = [axes]
        for b in range(0, len(all_kp)):
            img_curr = imgs[b].float().cpu().numpy()
            img_curr = img_curr.transpose(1, 2, 0)
            # img_curr = (img_curr * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_curr = (img_curr + 1.0) / 2.0
            img_curr = np.clip(img_curr, 0, 1)
            resized_img_curr = (img_curr * 255).astype(np.uint8)
            axes[b].imshow(resized_img_curr)
            axes[b].axis('off')  # Hide the axis
            for i, kp in enumerate(all_kp[b]):
                if world_kp:
                    u,v = world_to_pixel(kp, self.camera_matrix, np.eye(4))
                    axes[b].scatter(u/960*self.img_width, (v-26)/548*224)
                else:
                    u, v = kp
                    axes[b].scatter(u, v)
        fig.tight_layout()
        plt.savefig(f'{self.fig_dir}/{self.demo_name}/all_kps_{idx}.png', bbox_inches='tight', dpi=50)
        plt.close(fig)

    def optimize_actions(self, obs_image, goal_image, rel_t, len_traj_pred, sim, obs, curr_kp):
        n_evals = obs_image.shape[0] # default value is 1 (batch size 1)
        start_mu = self.get_current_joints(sim)
        mu, sigma, mu_ref = self.init_mu_sigma(obs_image, len_traj_pred, curr_mu=start_mu)  # initializing all joints as optimization variables

        goal_emb_0 = self.model.module.encode_image(goal_image[0].unsqueeze(0).unsqueeze(1)).squeeze(1)
        goal_kp_0, _ = self.model.module.forward_kp(goal_emb_0.unsqueeze(1).repeat(1,2,1,1), None, None, None)
        _, _, H, W = goal_image.shape
        goal_kp_0 = goal_kp_0.view(1,1,-1,H,W)        # (1,1,12,H,W)     6 keypoints per hand
        goal_kp_0 = goal_kp_0[:,0,6:self.kp_end_idx]                # (1,1,6,H,W) only cconsidering right hand
        goal_kp_uv = get_keypoints_from_beliefmap(goal_kp_0)     # (1,1,6,2)
        if self.rank==0:
            self.plot_all_kp(goal_image, goal_kp_uv.cpu().numpy(), f'goal_kp_{self.idx}', world_kp=False)

        if self.task_name=='grasp':
            mu[:,:,7:] = mu_ref[:,:,7:]
            mu[:,:,:7] = mu_ref[:,:,:7]

        if self.mu is not None:      # initializing mu and sigma with prev estimates for pred_steps_id>0 of MPC
            mu = self.mu[:,-len_traj_pred:]
            sigma = self.sigma[:,-len_traj_pred:]

        start_kp = curr_kp.reshape(1,1,-1)
        start_kp = torch.Tensor(start_kp).repeat(self.num_samples,1,1).to(self.device)

        if self.rank==0:
            iterator = tqdm(range(self.opt_steps), dynamic_ncols=True, leave=True, file=sys.stdout)
        else:
            iterator = range(self.opt_steps)
        for i in iterator:
            losses = []
            for sample_id in range(n_evals):
                joint_sample = (np.random.randn(self.num_samples, len_traj_pred, self.action_dim) * sigma[sample_id] + mu[sample_id])

                # calculate the end-effector pose for each sampled set of joint angles
                # this is used for end-effector orientation loss for grasping
                ee_pose = []
                for temporal_joint_angles in joint_sample:
                    ee_pose_curr = []
                    for joint_angles in temporal_joint_angles:
                        ee_pose_curr.append(self.get_ee_pose(joint_angles[:7], sim))
                    ee_pose.append(np.array(ee_pose_curr))
                ee_pose = np.array(ee_pose)
                loss_ee = np.linalg.norm(ee_pose[:,:,3:] - self.ref_orientation.reshape(1,1,-1),axis=(2)).mean(1)
                loss_ee = torch.Tensor(loss_ee).to(self.device)

                # create the delta keypoints (dexterous actions in DexWM) using the keypoints corresponding to sampled joints
                kp_sample = self.get_kp_from_joints(joint_sample.reshape(-1,self.action_dim), sim, obs).reshape(self.num_samples,len_traj_pred, -1)
                kp_sample = torch.Tensor(kp_sample).to(self.device)
                all_kp = torch.cat([start_kp, kp_sample],dim=1)
                sample = all_kp[:,1:]-all_kp[:,:-1]
                deltas_right = sample
                deltas_left = torch.zeros_like(deltas_right)
                deltas = torch.cat([deltas_left, deltas_right],dim=-1)

                # self.seen_frames is populated as the robot takes steps in the simulation
                seen_images = torch.cat(self.seen_frames,dim=1)
                cur_obs_image = seen_images[sample_id].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1)
                goal_emb_0 = self.model.module.encode_image(goal_image[sample_id].unsqueeze(0).unsqueeze(1)).squeeze(1)
                goal_emb = goal_emb_0.repeat(self.num_samples, 1, 1)
                goal_kp_0, _ = self.model.module.forward_kp(goal_emb_0.unsqueeze(1).repeat(1,2,1,1), None, None, None)
                _, _, H, W = goal_image.shape
                goal_kp_0 = goal_kp_0.view(1,1,-1,H,W)        # (1,1,12,H,W)
                goal_kp_0 = goal_kp_0[:,0,6:self.kp_end_idx]                # (1,1,6,H,W)
                goal_kps = goal_kp_0.repeat(self.num_samples, 1, 1, 1)     # (B,6,H,W)
                goal_kp_uv = get_keypoints_from_beliefmap(goal_kps)      # (B,6,2)

                exp_weights = torch.pow(0.6, torch.arange(self.pred_steps-1, -1, -1)).to(self.device)
                exp_weights = exp_weights/exp_weights.sum()
                exp_weights = exp_weights[-len_traj_pred:]

                expanded_deltas = deltas
                expanded_obs_image = cur_obs_image

                dataset = TensorDataset(expanded_obs_image, expanded_deltas)
                # Parallelize trajectories across ALL GPUs globally
                sampler = DistributedSampler(dataset, shuffle=False)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

                sampler.set_epoch(0)
                preds, pred_kp_uv = self.autoregressive_rollout(dataloader)  # (B, T, P, F), (B, T, 6, 2)


                # The loss is computed between the goal and *each* predicted temporal state (not only the final state)
                # exp_weights increases the contribution of later timesteps exponentially
                # In most cases, results are similar even without exp_weights
                B1 = preds.shape[0]
                max_batch = self.batch_size*8
                if B1<=max_batch:     # for larger batches, cuda is out of memory
                    loss1 = self.loss_fn(preds.to(self.device), goal_emb.unsqueeze(1).repeat(1,len_traj_pred, 1, 1).to(self.device)).mean([2,3]) # (B,8)
                    loss1 = (loss1*exp_weights).sum(dim=1)  # (B)
                    loss_kp = torch.nn.MSELoss(reduction='none')(pred_kp_uv, goal_kp_uv.unsqueeze(1).repeat(1,len_traj_pred,1,1)).mean([2,3])
                    loss_kp = (loss_kp*exp_weights).sum(dim=1)
                else:
                    loss1 = []
                    loss_kp = []
                    for kk in range(0,B1,max_batch):
                        loss1_kk = self.loss_fn(preds[kk:kk+max_batch].to(self.device), goal_emb[kk:kk+max_batch].unsqueeze(1).repeat(1,len_traj_pred, 1, 1).to(self.device)).mean([2,3]) # (B,8)
                        loss1_kk = (loss1_kk*exp_weights).sum(dim=1)
                        loss1.append(loss1_kk)
                        loss_kp_kk = torch.nn.MSELoss(reduction='none')(pred_kp_uv[kk:kk+max_batch], goal_kp_uv[kk:kk+max_batch].unsqueeze(1).repeat(1,len_traj_pred,1,1)).mean([2,3])
                        loss_kp_kk = (loss_kp_kk*exp_weights).sum(dim=1)
                        loss_kp.append(loss_kp_kk)
                    loss1 = torch.cat(loss1)
                    loss_kp = torch.cat(loss_kp)

                loss = loss1 + self.kp_loss_weight*loss_kp
                if self.use_loss_ee:
                    loss += 0.4*loss_ee

                # selecting the top_k sampled joints to update mu and sigma
                sorted_idx = torch.argsort(loss)
                topk_idx = sorted_idx[:self.topk]
                topk_action = joint_sample[topk_idx.cpu().numpy()]
                losses.append(loss[topk_idx[0]].item())
                mu[sample_id] = topk_action.mean(axis=0)
                sigma[sample_id] = topk_action.std(axis=0)
                if self.rank==0:
                    iterator.set_postfix({'Loss': f'{loss[topk_idx][0].detach().cpu().numpy():.3f}',
                                          'Loss1': f'{loss1[topk_idx][0].detach().cpu().numpy():.3f}',
                                          'Loss_kp': f'{loss_kp[topk_idx][0].detach().cpu().numpy():.7f}',
                                          'Loss_ee': f'{loss_ee[topk_idx][0].detach().cpu().numpy():.3f}'})
            del dataset, dataloader, seen_images, cur_obs_image, goal_emb, goal_emb_0, preds, loss1, loss_kp, goal_kp_uv
            gc.collect()
            torch.cuda.empty_cache()

        # Final rollout

        self.mu = mu
        self.sigma = sigma
        joint_sample = mu    # this is what we finally use for controlling the robot
        kp_sample = self.get_kp_from_joints(joint_sample.reshape(-1,self.action_dim), sim, obs).reshape(1,len_traj_pred, -1)
        kp_sample = torch.Tensor(kp_sample).to(self.device)
        all_kp = torch.cat([start_kp[0:1], kp_sample],dim=1)

        sample = all_kp[:,1:]-all_kp[:,:-1]
        deltas_right = sample
        deltas_left = torch.zeros_like(deltas_right)
        deltas = torch.cat([deltas_left, deltas_right],dim=-1)

        # for visualization
        dataset = TensorDataset(expanded_obs_image, deltas.repeat(expanded_obs_image.shape[0],1,1))
        # Parallelize trajectories across ALL GPUs globally
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

        pred_states, _ = self.autoregressive_rollout(dataloader)

        pred_states2 = torch.cat([pred_states[0:1,0:1], pred_states[0:1]], dim=1)
        pred_kps, _ = self.model.module.forward_kp(pred_states2[0:1], None, None, None)
        _, _, H, W = goal_image.shape
        pred_kps = pred_kps.view(1,-1,12,H,W)
        pred_kps = pred_kps[0,:,6:self.kp_end_idx]
        pred_kps = get_keypoints_from_beliefmap(pred_kps)

        if self.rank==0:
            hw = [224//self.patch_size, self.img_width//self.patch_size]  # e.g., [16,28] or [14,24]
            pred_imgs = self.decoder(pred_states[0:1], hw_input = hw)
            pred_states = torch.cat([pred_states[0:1,0:1], pred_states[0:1]], dim=1)
            self.plot_all_kp(pred_imgs[0], kp_sample[0], f'{self.idx}')
            pred_kps, _ = self.model.module.forward_kp(pred_states[0:1], None, None, None)
            _, _, H, W = goal_image.shape
            pred_kps = pred_kps.view(1,-1,12,H,W)
            pred_kps = pred_kps[0,:,6:self.kp_end_idx]
            pred_kps = get_keypoints_from_beliefmap(pred_kps)
            self.plot_all_kp(pred_imgs[0], pred_kps.cpu().numpy(), f'pred_kp_{self.idx}', world_kp=False)

        actions = deltas_right
        return actions[0], joint_sample[0], pred_kps

    def autoregressive_rollout(self, dataloader, rollout_stride=1, cam_actions=None, rel_t=None):
        all_emb_list = []
        all_kp_list = []
        for obs_image, deltas in dataloader:
            obs_image = obs_image.to(self.device)
            _,_,_,H,W = obs_image.shape
            deltas = deltas.to(self.device)
            deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)
            T = self.pred_steps
            b = deltas.shape[0]
            cam_actions = torch.zeros([deltas.shape[0], deltas.shape[1], 6]).to(self.device)
            rel_t = torch.zeros([deltas.shape[0], deltas.shape[1]]).to(self.device)
            deltas = torch.cat([deltas, cam_actions], -1)
            preds = []
            curr_obs = obs_image.clone().to(self.device)

            T_context = self.num_context
            T1 = T + T_context - deltas.shape[1]
            deltas_padded = torch.cat([torch.zeros_like(deltas)[:,0:1].repeat(1,T1-1,1), deltas], axis=1)
            rel_t_padded = torch.cat([torch.zeros_like(rel_t)[:,0:1].repeat(1,T1-1), rel_t], axis=1)

            pad_left = curr_obs[:,0:1].repeat(1,T_context,1,1,1)
            num_pad_right = T - curr_obs[:,1:].shape[1]
            pad_right = curr_obs[:,-1:].repeat(1,num_pad_right,1,1,1)
            curr_obs_padded = torch.cat([pad_left, curr_obs[:,1:], pad_right], dim=1)

            prev_emb = None
            pred_kp = None
            start_ind = T-deltas.shape[1]
            for n in range(start_ind, start_ind+deltas.shape[1]):
                actions_n = deltas_padded[:,n:n+T_context]
                rel_t_n = rel_t_padded[:,n:n+T_context]
                curr_frames_n = curr_obs_padded[:,n:n+T_context+1]
                with torch.inference_mode():
                    goal_pred_n, goal_tgt_n, pred_kp_n, emb_loss, kp_loss = self.model(curr_frames_n, actions_n, rel_t_n, prev_emb=prev_emb, action_diff=True)
                pred_kp_n = pred_kp_n.view(b,-1,12,H,W)
                if prev_emb is None:
                    prev_emb = goal_pred_n[:,-1:]
                    pred_kp = pred_kp_n[:,-1:]
                else:
                    prev_emb = torch.cat([prev_emb, goal_pred_n[:,-1:]], axis=1)
                    pred_kp = torch.cat([pred_kp, pred_kp_n[:,-1:]], axis=1)
            preds = prev_emb

        pred_kp = pred_kp[:,:,6:self.kp_end_idx]
        pred_kp = pred_kp.reshape(-1,self.kp_end_idx-6,H,W)
        pred_kp = get_keypoints_from_beliefmap(pred_kp)
        pred_kp = pred_kp.reshape(b,-1,self.kp_end_idx-6,2)

        # gather from all GPUs
        gathered_preds = gather_outputs(preds, self.device)
        gathered_kp = gather_outputs(pred_kp, self.device)
        return gathered_preds, gathered_kp

    def get_delta_kp(self, img, sim, obs, curr_kp):
        curr_frame = self.image_transform(img).unsqueeze(0).unsqueeze(0).to(self.device)
        goal_frame = self.image_transform(self.goal_img).unsqueeze(0).to(self.device)
        self.seen_frames.append(curr_frame)
        len_traj_pred = self.pred_steps-(len(self.seen_frames)-1)
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                delta, joint_angles, pred_kps = self.optimize_actions(curr_frame, goal_frame, rel_t=None, len_traj_pred=len_traj_pred, sim=sim, obs=obs, curr_kp=curr_kp)
        delta = delta[:self.action_step].view(self.action_step,-1,3)
        delta = delta.cpu().numpy()
        joint_angles = joint_angles[:self.action_step]
        return delta, joint_angles, pred_kps

    def set_goal(self, goal_img, demo_name):
        self.goal_img = goal_img
        self.seen_frames = []
        self.demo_name = demo_name
        os.makedirs(f'{self.fig_dir}/{demo_name}', exist_ok=True)
        self.idx = 0
        if self.rank==0:
            self.plot_images(self.goal_img, None, 'goal', use_cv2=True)
        self.mu = None
        self.sigma = None

    def get_ee_pose(self, franka_joints, sim):
        # forward kinematics
        data_copy = mujoco.MjData(sim.model._model)  # creating a copy so as to not disturb the actual simulation
        data_copy.qpos[self.joint_indices[:7]] = franka_joints # set gripper joints
        mujoco.mj_kinematics(sim.model._model, data_copy)

        body_name = 'robot0_right_hand'
        body_id = sim.model.body_name2id(body_name)
        pos = data_copy.xpos[body_id]
        xmat = data_copy.xmat[body_id].reshape(3,3)
        quat = R.from_matrix(xmat).as_quat()
        axisangle = T.quat2axisangle(quat)
        pose = np.concatenate([pos, axisangle])    # required format by the low level controller
        return pose

    def get_current_img(self, obs):
        curr_img_1200p = obs['robot0_robotview_2_image'][::-1]
        if curr_img_1200p.shape[0] > 600:
            h, w = curr_img_1200p.shape[:2]
            curr_img = cv2.resize(curr_img_1200p, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
        else:
            curr_img = curr_img_1200p
        return curr_img, curr_img_1200p

    def get_actions(self, obs, robot, env):
        self.joint_indices = [robot.sim.model.joint_name2id(name) for name in self.joint_names]
        self.obs = obs
        curr_kp = self.get_current_kp(obs) # in camera coordinate frame
        curr_img, curr_img_1200p = self.get_current_img(obs)    # also tracking 1200p images for plotting
        if self.rank==0:
            self.plot_images(curr_img_1200p, None, f'{self.idx}_0', use_cv2=True)

        if self.all_joint_angles.shape[0]==0:
            all_delta_kp, all_joint_angles, all_pred_kps = self.get_delta_kp(curr_img, robot.sim, obs, curr_kp)
            self.all_delta_kp = all_delta_kp
            self.all_joint_angles = all_joint_angles

        delta_kp = self.all_delta_kp[0]
        joint_angles = self.all_joint_angles[0]

        self.all_delta_kp = self.all_delta_kp[1:]
        self.all_joint_angles = self.all_joint_angles[1:]
        new_kp = curr_kp + delta_kp # in camera frame
        new_kp = new_kp[:-4]  # no pinky finger in allgero gripper
        if self.rank==0:
            self.plot_images(curr_img, new_kp, f'{self.idx}_newkp')
        self.idx+=1
        new_kp = self.cam_to_world_frame(new_kp, obs) # in world frame
        self.joint_pos = robot.sim.data.qpos[self.joint_indices]
        ee_pose =  self.get_ee_pose(self.joint_pos[:7], robot.sim)
        gripper_angle = self.joint_pos[7:]

        new_joint_pos = joint_angles
        new_ee_pose = self.get_ee_pose(new_joint_pos[:7], robot.sim)
        new_gripper_angle = new_joint_pos[7:]

        delta_ee_pose = new_ee_pose - ee_pose

        actions = {}
        actions['right_abs'] = new_ee_pose
        actions['right_delta'] = delta_ee_pose
        actions['right_gripper'] = new_gripper_angle
        actions['left_abs'] = new_ee_pose*0.0
        actions['left_delta'] = delta_ee_pose*0.0
        actions['left_gripper'] = new_gripper_angle*0.0
        actions['base'] = np.array([0,0,0])
        actions['base_mode'] = np.array([-1])
        return actions, ee_pose
