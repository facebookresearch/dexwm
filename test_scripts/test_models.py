# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import numpy as np
import argparse
import yaml
import torch
import torch.nn as nn
import cv2
import h5py
import random
import copy
import sys
import os
import json
import torchvision
import imageio
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from distributed import init_distributed, cleanup
from datasets.egodex import EgoDexDataset
from datasets.droid import DroidDataset
from models.model import DexWM
import matplotlib.pyplot as plt
from models.decoder_vit import SimpleViTDecoder
from utils.image_utils import get_keypoints_from_beliefmap, calculate_pck_batch
from train_wm import get_patch_size_from_backbone

def tensor_to_uint8_img(tensor, norm_type='siglip'):
    # Denormalize (assuming ImageNet stats)
    img = tensor.cpu().numpy().transpose(1, 2, 0)

    if norm_type == 'imagenet':
        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    # Bad naming, basically if input is in range -1 to 1
    elif norm_type == 'siglip':
        img = (img + 1.0) / 2.0
    else:
        raise Exception(f'Norm type {norm_type} not supported')

    img = np.clip(img, 0, 1)  # Ensure valid range
    img = (img * 255).astype(np.uint8)
    return img

def generate_gifs(img_gt, img_pred, i, output_dir):
    gt_frames = []
    pred_frames = []
    for b in range(len(img_gt)):
        # if b>=8:
        #     break
        gt_img = tensor_to_uint8_img(img_gt[b], norm_type = 'imagenet')
        if b == 0:
            norm_type = 'imagenet'
        else:
            # Misnomer, just means input is in range -1 to 1 (the case for David's ViT decoder)
            norm_type = 'siglip'
        pred_img = tensor_to_uint8_img(img_pred[b], norm_type)
        gt_frames.append(gt_img)
        pred_frames.append(pred_img)

    imageio.mimsave(f'{output_dir}/test_gt_{i}.gif', gt_frames, format='GIF', fps=5.0)
    imageio.mimsave(f'{output_dir}/test_pred_{i}.gif', pred_frames, format='GIF', fps=5.0)
    print(f'{output_dir}/test_gt_{i}.gif')
    print(f'{output_dir}/test_pred_{i}.gif')

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')
    del model_parameters, params

def get_cosine_sim(pred_emb, goal_emb):
    pred_flat = pred_emb.view(pred_emb.size(0), pred_emb.size(1), -1)
    goal_flat = goal_emb.view(goal_emb.size(0), goal_emb.size(1), -1)
    cos_sim = F.cosine_similarity(pred_flat, goal_flat, dim=2)  # shape: [batch, channels]
    mean_cos_sim = cos_sim.mean(0).cpu().numpy()  # shape: [channels]
    return mean_cos_sim

def cam_action_to_transformation(action):
    cam_pos = action[0]
    cam_rot = action[1]
    cam_rotation = R.from_euler('xyz', cam_rot).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = cam_rotation
    transformation_matrix[:3, 3] = cam_pos
    return transformation_matrix

def get_pixels_from_kp(xyz, extrinsic_matrix):
    intrinsic_matrix = np.array([[736.6339, 0., 960.],
                        [0., 736.6339, 540.],
                        [0., 0., 1.]])
    X, Y, Z = xyz
    world_point = np.array([X, Y, Z, 1])
    camera_point = extrinsic_matrix @ world_point
    pixel_homogeneous = intrinsic_matrix @ camera_point[:3]
    u = pixel_homogeneous[0] / pixel_homogeneous[2]
    v = pixel_homogeneous[1] / pixel_homogeneous[2]
    return (u, v)

def plot_results(pred_img, actual_img, pred_act, rel_t, init_kp, curr_cams, idx, output_dir, pred_kps=None):
    fig, axes = plt.subplots(nrows=2, ncols=len(pred_img), figsize=(60, 3))
    curr_time = 0
    curr_kp = init_kp
    cam_0_in_w = cam_action_to_transformation(curr_cams[0])
    for b in range(0, len(pred_img)):
        cam_b_in_w = cam_action_to_transformation(curr_cams[b])
        img_curr = actual_img[b].cpu().numpy()
        img_curr = img_curr.transpose(1, 2, 0)
        img_curr = (img_curr * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        resized_img_curr = (img_curr * 255).astype(np.uint8)
        axes[0, b].imshow(resized_img_curr)
        axes[0, b].axis('off')  # Hide the axis
        # if b == 0:
        #     axes[0, b].text(-0.1, 0.5, 'Actual Img', transform=axes[0, b].transAxes, ha='center', va='center', rotation=90, fontsize=15)
        axes[0, b].set_title(f'{curr_time/30:.1f} Sec.', fontsize=15)

        img_curr = pred_img[b].cpu().numpy()
        img_curr = img_curr.transpose(1, 2, 0)
        if b>0:
            img_curr = (img_curr + 1.0) / 2.0
        else:
            img_curr = (img_curr * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_curr = np.clip(img_curr, 0, 1)  # Ensure valid range
        resized_img_curr = (img_curr * 255).astype(np.uint8)
        axes[1, b].imshow(resized_img_curr)
        axes[1, b].axis('off')  # Hide the axis
        cam_b_in_0 = np.linalg.inv(cam_0_in_w)@cam_b_in_w
        cam_0_in_b = np.linalg.inv(cam_b_in_0)
        if pred_kps is not None:
            curr_kp = pred_kps[b]
            cam_0_in_b = np.eye(4) # prediction is already in the current cam frame
        for kp in curr_kp:
            u, v = get_pixels_from_kp(kp, cam_0_in_b)
            if u<0 or u>1920 or v<0 or v>1080:
                continue
            # axes[1, b].scatter(u/1920*784-6, v/1080*448, s=1)
            axes[1, b].scatter(u/1920*398-3, v/1080*224, s=1)
        # if b == 0:
        #     axes[1, b].text(-0.1, 0.5, 'Pred Img', transform=axes[1, b].transAxes, ha='center', va='center', rotation=90, fontsize=15)
        if b!=(len(pred_img)-1):
            curr_time += rel_t[b]
            curr_kp += pred_act[b].reshape(42,3)
    plt.savefig(f'{output_dir}/test_{idx}.png', bbox_inches='tight')
    plt.close(fig)

def get_kp_model(args_file, device):
    with open(args_file, 'r') as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)

    num_context = args['data']['num_context']

    backbone_name = args['model']['backbone_name']
    hidden_dim = args['model']['hidden_dim']
    action_dim = args['model']['action_dim']
    depth = args['model']['depth']
    num_heads = args['model']['num_heads']
    mlp_ratio = args['model']['mlp_ratio']
    heatmap_dim = args['model'].get('heatmap_dim', 256)
    heatmap_layers = args['model'].get('heatmap_layers', 6)

    patch_size, num_patches = get_patch_size_from_backbone(backbone_name)

    kp_model = DexWM(backbone_name=backbone_name,
                num_patches=num_patches,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                is_eval=True,
                num_context = num_context,
                emb_loss_fn=nn.MSELoss(reduction='mean'),
                heatmap_dim=heatmap_dim,
                heatmap_layers=heatmap_layers).to(device)

    return kp_model

def main(args):
    device = 'cuda'
    model_checkpoint_path = args['model_checkpoint']
    decoder_checkpoint = args['decoder_checkpoint']
    target_resolution = args['target_resolution']
    kp_checkpoint_path = args['kp_checkpoint']
    dataset_name = args['dataset']
    root_folder = args['data']['egodex_root_folder'] if 'egodex_root_folder' in args['data'] else args['data']['root_folder']
    max_context_len_og = args['data']['max_context_len']
    num_context = args['data']['num_context']
    long_context_len = 20*6 # 20*6, 20*100
    long_context = 20 # 20*0.2ms = 4s
    img_size = args['data']['img_size']
    aug = args['data']['aug']
    keys = args['data'].get('keys', 'all')
    full_seq = args['data'].get('full_seq', False)

    backbone_name = args['model']['backbone_name']
    hidden_dim = args['model']['hidden_dim']
    action_dim = args['model']['action_dim']
    depth = args['model']['depth']
    num_heads = args['model']['num_heads']
    mlp_ratio = args['model']['mlp_ratio']

    batch_size = args['train']['batch_size']
    batch_size = 15 if dataset_name=='egodex' else 1
    batch_size = 1 if args['visualize'] else batch_size
    num_workers = args['train']['num_workers']
    global_seed = args['train']['global_seed']

    # Get output directory, defaulting to 'debug' if not specified
    output_dir = args.get('output_dir', 'debug')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    patch_size, num_patches = get_patch_size_from_backbone(backbone_name)

    if dataset_name=='egodex':
        val_subset = EgoDexDataset(root_folder=root_folder, max_context_len=long_context_len, num_context=long_context, patch_size=patch_size,
                                        backbone_name=backbone_name, img_size=img_size, aug=False, train=False, evaluate=True, keys=keys)

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False,
        shuffle=True if args['use_model_pred_kps'] else False,  # use_model_pred_kps is used only for plotting images, shuffling to get varied images
        drop_last=True if dataset_name=='egodex' else False)


    dexwm = DexWM(backbone_name=backbone_name,
                num_patches=num_patches,
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                is_eval=True,
                num_context = num_context,
                emb_loss_fn=nn.MSELoss(reduction='mean')).to(device)

    kp_model = get_kp_model(args['kp_config'], device)

    # print('DROID trained model has a different kp_layer for some reason, need to manually change it for now till we figure the issue!')
    checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))["model"]
    pretrained_dict = checkpoint
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        # if "kp_layer" not in k:
            new_key = k.replace("_orig_mod.", "")  # Remove the "module." prefix
            new_state_dict[new_key] = v
    # dexwm.load_state_dict(new_state_dict, strict=False)
    dexwm.load_state_dict(new_state_dict)

    checkpoint = torch.load(kp_checkpoint_path, map_location=torch.device('cpu'))["model"]
    pretrained_dict = checkpoint
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_key = k.replace("_orig_mod.", "")  # Remove the "module." prefix
        new_state_dict[new_key] = v
    kp_model.load_state_dict(new_state_dict)

    decoder = SimpleViTDecoder(encoder_dim=dexwm.encoder_dim, target_resolution=target_resolution, patch_size=patch_size).cuda()

    # load decoder checkpoint if provided
    if args['visualize'] or args['decoder_checkpoint']:
        decoder.load_decoder(decoder_checkpoint)

    if not args['visualize']:
        dexwm = torch.compile(dexwm)

    dexwm.eval()
    kp_model.eval()
    tot_emb_loss = 0
    tot_recon_loss = 0
    all_l2_cost = []
    all_kp_cost = []
    for i, (curr_frames, actions, rel_t, heatmaps, valid_kp, metadata) in tqdm(enumerate(val_loader), total=len(val_loader)):
        orig_actions = copy.deepcopy(actions)
        curr_frames = curr_frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)

        num_kpts = len(val_subset.all_keys)

        rel_t = torch.Tensor(rel_t).to(device, non_blocking=True)
        init_kp = actions[:,0,:num_kpts].cpu().numpy()
        curr_cams = np.concatenate([actions[:,0:1,num_kpts:num_kpts+2].cpu().numpy(), actions[:,:,-2:].cpu().numpy()],axis=1)

        curr_xyz = actions[:,:,:num_kpts+2]
        next_xyz = actions[:,:,num_kpts+2:]
        xyz = next_xyz - curr_xyz
        xyz_norm = xyz
        xyz_norm[:,:,-1] = xyz_norm[:,:,-1]%(2*np.pi)
        actions = xyz_norm

        T = 8
        T1 = actions.shape[1]
        actions_padded = torch.cat([torch.zeros_like(actions)[:,0:1].repeat(1,T-1,1,1), actions], axis=1)
        rel_t_padded = torch.cat([torch.zeros_like(rel_t)[:,0:1].repeat(1,T-1), rel_t], axis=1)
        curr_frames_padded = curr_frames[:,0:1].repeat(1,T+T1,1,1,1)
        prev_emb_list = []
        goal_emb_list = []
        kp_list = []
        prev_emb=None
        for n in range(actions.shape[1]):
            actions_n = actions_padded[:,n:n+T]
            rel_t_n = rel_t_padded[:,n:n+T]
            curr_frames_n = curr_frames_padded[:,n:n+T+1]
            # print(n,n+T,actions_n.shape, rel_t_n.shape, curr_frames_n.shape)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.inference_mode():
                    goal_pred_n, goal_tgt_n, pred_kp_n, emb_loss, kp_loss = dexwm(curr_frames_n, actions_n, rel_t_n, prev_emb=prev_emb, action_diff=True)
                    if patch_size == 14:
                        hw = [16,28]
                    elif patch_size == 16:
                        hw = [14,24]
            B, _, _, _ = goal_pred_n.shape
            _, H, W = pred_kp_n.shape
            pred_kp_n = pred_kp_n.view(B, T, -1, H, W)
            prev_emb_list.append(goal_pred_n[:,-1:])
            goal_emb_list.append(goal_tgt_n[:,-1:])
            kp_list.append(pred_kp_n[:,-1:])
            prev_emb = torch.cat(prev_emb_list, dim=1)


        pred_emb = torch.cat(prev_emb_list, dim=1)
        goal_emb = torch.cat(goal_emb_list, dim=1)
        goal_emb = dexwm.encode_image(curr_frames)[:,1:]
        model_pred_kps = torch.cat(kp_list, dim=1).float()
        l2_dist = (torch.sqrt((pred_emb-goal_emb)**2).mean([0,2,3])).cpu().numpy()
        cos_similarity = get_cosine_sim(pred_emb, goal_emb)
        all_l2_cost.append(l2_dist)

        B, T, _, _, _ = model_pred_kps.shape
        img_emb = dexwm.encode_image(curr_frames)
        # img_emb = dexwm.input_proj(img_emb)
        # print(img_emb.shape)
        model_pred_kps, _ = dexwm.forward_kp(img_emb, None, None, None)
        model_pred_kps = model_pred_kps.view(B, T, -1, H, W)

        B, T, _, _ = pred_emb.shape

        if not args['use_model_pred_kps']:
            pred_emb2 = torch.cat([pred_emb[:,0:1], pred_emb], axis=1)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.inference_mode():
                    pred_kp, kp_loss = kp_model.forward_kp(pred_emb2, cam_pose=None, gt_kps=None, valid_kp=None)
            _, H, W = pred_kp.shape
            pred_kp = pred_kp.view(B*T,-1,H,W)
            heatmaps = heatmaps.view(B*T,-1,H,W)
            pred_uv = get_keypoints_from_beliefmap(pred_kp)
            gt_uv = get_keypoints_from_beliefmap(heatmaps)
            pred_uv = pred_uv.view(B,T,-1,2)
            gt_uv = gt_uv.view(B,T,-1,2)

            pck_scores = calculate_pck_batch(pred_keypoints=pred_uv.view(B*T,-1,2).cpu().numpy(),
                                            gt_keypoints=gt_uv.view(B*T,-1,2).cpu().numpy(),
                                            image_width=392,
                                            image_height=img_size,
                                            thresholds=[2.5,5.0,10.0,20.0])
            pck_scores = pck_scores.reshape(B,T,-1)
            # print(pck_scores)
            all_kp_cost.extend(pck_scores)    # pck_scores is shaped (T, number of thresholds)


        if not args['visualize']:
            continue

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.inference_mode():
                goal_pred_frames = decoder(pred_emb, hw_input=hw)

        assert batch_size==1
        goal_pred_frames = goal_pred_frames[0]

        if args['use_model_pred_kps']:
            model_pred_kps = model_pred_kps.view(B*T,-1,H,W)
            model_pred_kps = get_keypoints_from_beliefmap(model_pred_kps)
            pred_uv = model_pred_kps.view(B,T,-1,2)
            pred_uv = pred_uv[:,:,6:,:]

        pred_uv = pred_uv[0]
        curr_frames = curr_frames[0]
        for ii in range(goal_pred_frames.shape[0]):
            # curr_frame = goal_pred_frames[ii].cpu().numpy().transpose(1,2,0)
            # curr_frame = (curr_frame * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            # resized_img = (curr_frame*255).astype(np.uint8)
            # resized_img = tensor_to_uint8_img(goal_pred_frames[ii], 'siglip')
            resized_img = tensor_to_uint8_img(curr_frames[ii+1], 'siglip')
            plt.imshow(resized_img)
            plt.axis('off')
            plt.savefig(f'{output_dir}/curr_{i}_{ii}.png', bbox_inches='tight', pad_inches=0)
            for u, v in pred_uv[ii].cpu().numpy():
                # if ii<goal_pred_frames.shape[0]//2:
                plt.scatter(
                            u, v,
                            s=60,                # marker size
                            color='red',         # fill color
                            edgecolors='white',  # edge color for contrast
                            linewidths=1.5,      # edge thickness
                            alpha=0.7,           # transparency
                            marker='o'           # marker shape
                        )
            plt.savefig(f'{output_dir}/kp_{i}_{ii}.png', bbox_inches='tight', pad_inches=0)
            plt.close()

        np.save(f'{output_dir}/kp_3d_{i}.npy', orig_actions[0].numpy())

        generate_gifs(img_gt=curr_frames[1:], img_pred=goal_pred_frames,
                        i=i, output_dir=output_dir)


    all_l2_cost = np.array(all_l2_cost).mean(0)
    np.savetxt(f'{output_dir}/{dataset_name}_l2.txt', all_l2_cost, fmt='%.6f')

    if not args['use_model_pred_kps']:
        all_kp_cost = np.array(all_kp_cost).mean(0)
        np.savetxt(f'{output_dir}/{dataset_name}_pck.txt', all_kp_cost, fmt='%.6f')
        print(output_dir)



if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the checkpoint to run inference on')
    parser.add_argument('--decoder_checkpoint', type=str, required=False, help='Path to decoder checkpoint')
    parser.add_argument('--target_resolution', type=int, nargs='+', default=[224], help='Target resolution(s) for decoder (e.g., 224 or 224 392)')
    parser.add_argument('--kp_config', type=str, required=True, help='Path to the checkpoint to run inference on')
    parser.add_argument('--kp_checkpoint', type=str, required=True, help='Path to the checkpoint to run inference on')
    parser.add_argument('--output_dir', type=str, default='debug2', help='Directory to save output files (default: debug)')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--use_model_pred_kps', default=False, action='store_true')
    args = parser.parse_args()
    fname = args.config

    # Either a list of 2 numbers or just a int
    if len(args.target_resolution) == 1:
        args.target_resolution = args.target_resolution[0]

    os.makedirs(args.output_dir, exist_ok=True)

    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        params['model_checkpoint'] = args.model_checkpoint
        params['decoder_checkpoint'] = args.decoder_checkpoint
        params['target_resolution'] = args.target_resolution
        params['kp_checkpoint'] = args.kp_checkpoint
        params['kp_config'] = args.kp_config
        params['output_dir'] = args.output_dir
        params['visualize'] = args.visualize
        params['use_model_pred_kps'] = args.use_model_pred_kps
    main(params)
