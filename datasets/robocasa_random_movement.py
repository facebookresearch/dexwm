import numpy as np
import torch
import os
import h5py
import cv2
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation as R
import pickle as pkl
from utils.misc_utils import pad_and_resize, center_crop_and_resize
import torchvision
from torchvision import tv_tensors
import torchvision.transforms.v2 as transforms
import random
import itertools
from .augmentations import apply_color_jitter, apply_occlusion, apply_rgb_aug
import matplotlib.pyplot as plt
import json
from utils.image_utils import create_belief_map


def get_pixels_from_kp(xyz, extrinsic_matrix, patch_size):
    intrinsic_matrix = np.array([[357.52607778,   0.,         480.        ],
                            [  0.,         357.52607778, 274.        ],
                            [  0.,           0.,           1.        ]])
    X, Y, Z = xyz
    world_point = np.array([X, Y, Z, 1])
    camera_point = extrinsic_matrix @ world_point
    pixel_homogeneous = intrinsic_matrix @ camera_point[:3]
    u = pixel_homogeneous[0] / pixel_homogeneous[2]
    v = pixel_homogeneous[1] / pixel_homogeneous[2]
    if patch_size==14:
        u,v = u/960*392, (v-26)/548*224 # 26 is the pixels cropped on each size (52 / 2)
    if patch_size==16:
        u,v = u/960*392 - 4, (v-26)/548*224 # 26 is the pixels cropped on each size (52 / 2)
    return (u, v)

class RobocasaRandomDataset(Dataset):
    def __init__(self, root_folder, max_context_len=90, num_context=4, patch_size=14, img_size=224,
                context_frame_step=2, aug=None, backbone_name='dinov2', train=False,
                evaluate=False, full_seq=False, var_time=False):
        super(RobocasaRandomDataset, self).__init__()
        self.data_root = root_folder
        self.all_tasks = []
        task_class = 'gripper_open_and_close'
        self.hdf5_files1 = os.listdir(os.path.join(self.data_root, task_class))
        self.hdf5_files1 = [file for file in self.hdf5_files1 if file.endswith('.hdf5')]
        for hdf5_file in self.hdf5_files1:
            path = os.path.join(self.data_root, task_class, hdf5_file)
            with h5py.File(path,'r') as f:
                tasks = list(f['data'].keys())
                tasks = [[task_class, hdf5_file, task] for task in tasks]
                self.all_tasks.extend(tasks)

        task_class = 'exploratory_movements'
        self.hdf5_files2 = os.listdir(os.path.join(self.data_root, task_class))
        self.hdf5_files2 = [file for file in self.hdf5_files2 if file.endswith('.hdf5')]
        for hdf5_file in self.hdf5_files2:
            path = os.path.join(self.data_root, task_class, hdf5_file)
            with h5py.File(path,'r') as f:
                tasks = list(f['data'].keys())
                tasks = [[task_class, hdf5_file, task] for task in tasks]
                self.all_tasks.extend(tasks)

        self.all_files = self.all_tasks

        all_files = self.all_files
        split_file_path = 'utils/split_indices_robocasa_random.json'
        if os.path.exists(split_file_path):
            with open(split_file_path, 'r') as f:
                split_data = json.load(f)
            train_files = split_data['train']
            test_files = split_data['test']
        else:
            random.shuffle(all_files)
            split_index = int(len(all_files) * 0.9)
            train_files = all_files[:split_index]
            test_files = all_files[split_index:]
            split_data = {
                'train': train_files,
                'test': test_files
            }
            with open(split_file_path, 'w') as f:
                json.dump(split_data, f, indent=4)
        if train:
            self.all_files = train_files
        else:
            self.all_files = test_files

        self.max_context_len = max_context_len
        self.num_context = num_context
        self.img_size = img_size
        self.patch_size = patch_size
        self.var_time = var_time
        if aug:
            if 'siglip' in self.backbone_name:
                self.aug = [
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                        p=0.5,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3)],
                        p=0.5,
                    ),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]
                self.aug = transforms.Compose(self.aug)

            else:
                self.aug = [
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                        p=0.5,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3)],
                        p=0.5,
                    ),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            self.aug = transforms.Compose(self.aug)
        else:
            self.aug = None
        self.train = train
        clip_range = np.arange(0, len(self.all_files))
        frame_range = np.arange(1, 11)/10   # dividing the videos into different intervals so as not to load the same part everytime
        self.full_seq = full_seq
        if evaluate or full_seq:
            frame_range = np.array([0.99])
        self.idx_to_data = list(itertools.product(clip_range, frame_range))
        self.backbone_name = backbone_name
        self.T_camera_in_base = np.array([[ -0.0000000,  0.5000000, -0.8660254,  0.212],    # calculated from xml file for MURP
                                            [ -1.0000000, -0.0000000,  0.0000000, 0],
                                            [0.0000000,  0.8660254,  0.5000000, 1.614],
                                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        self.T_optical = np.array([          # for the camera base to camera optical frame
                    [1, 0,  0, 0],
                    [0, -1,  0, 0],
                    [0, 0, -1, 0],
                    [0, 0,  0, 1]
                ])
        self.hand_and_tip_keys = [0,4,8,12,16,20]

    def __len__(self):
        return len(self.idx_to_data)

    def image_transform(self, img, goal=False):
        # RoboCasa images are 960 x 600, but we crop the height to 548 first -> 960 x 548
        # -> 224 x 392 -> (patch size 14: 392, patch size 16: 384)
        ext_pix = (img.shape[0]-548)//2
        img = img[ext_pix:-ext_pix]
        h, w = img.shape[:2]
        ar = w/h
        img = cv2.resize(img, (int(self.img_size*ar), self.img_size), interpolation=cv2.INTER_LINEAR)

        if self.patch_size==14:
            img = img # keep the same
        if self.patch_size==16:
            img = img[:, 4:-4] # center cropping to the closest multiple of 16

        img = torch.Tensor(img.copy()).permute(2,0,1)
        return img

    def get_base_pose(self, obs, idx):
        p_base_in_world = obs['robot0_base_pos'][idx]
        q_base_in_world = obs['robot0_base_quat'][idx]
        R_base_in_world = R.from_quat(q_base_in_world).as_matrix()
        T_base_in_world = np.eye(4)
        T_base_in_world[:3,:3] = R_base_in_world
        T_base_in_world[:3,3] = p_base_in_world
        return T_base_in_world

    def process_annotation(self, obs, idx, prev_cam_ext=None, do_flip=False, do_belief_map=False):
        all_poses = []
        if prev_cam_ext is None:
            T_base_in_world = self.get_base_pose(obs, idx)
            T_camera_in_base = self.T_camera_in_base
            T_camera_in_world = T_base_in_world @ T_camera_in_base
            cam_ext = np.linalg.inv(T_camera_in_world)  # world to camera0 transformation (T_world_in_camera)
            cam_curr = np.linalg.inv(T_camera_in_world)
        else:
            cam_ext = prev_cam_ext  # points are represented in the frame of input (not goal) frame for consistency
            T_base_in_world = self.get_base_pose(obs, idx)
            T_camera_in_base = self.T_camera_in_base
            T_camera_in_world = T_base_in_world @ T_camera_in_base
            cam_curr = np.linalg.inv(T_camera_in_world)  # world to camera0 transformation (T_world_in_camera)

        keypoints = obs['robot0_right_gripper_keypoint_pose'][idx].reshape(-1,3)
        hand_keypoint = obs['robot0_right_hand_T_world_pose_mat'][idx].reshape(4,4)[:3,3:].T
        keypoints = keypoints[-16:]
        keypoints = np.concatenate([hand_keypoint, keypoints], axis=0)
        keypoints = keypoints[[0,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,9,10,11,12]]  # rearanging to match egodex data format and copying ring finger as proxy to little finger
        all_u = []
        all_v = []
        for ii, key in enumerate(keypoints):
            pos = key
            pos_hom = np.append(pos,1.)[...,None]
            pos_cam = cam_ext @ pos_hom
            pos_cam = self.T_optical@pos_cam
            all_poses.append(pos_cam[:3,0])
            if ii in self.hand_and_tip_keys:
                u,v = get_pixels_from_kp(pos_hom[:3,0], self.T_optical@cam_curr, patch_size=self.patch_size)
                all_u.append(u)
                all_v.append(v)
        if do_belief_map:
            all_u = np.array(all_u)[...,None]
            all_v = np.array(all_v)[...,None]


            valid_kp_mask = (all_u > 0) & (all_u < self.img_width) & (all_v > 0) & (all_v < 224)
            belief_map = create_belief_map(image_resolution=(self.img_width, 224),
                                                pointsBelief=np.concatenate([all_u,all_v],axis=1))
        else:
            belief_map = None
            valid_kp_mask = None

        T_base_in_world = self.get_base_pose(obs, idx)
        T_camera_in_base = self.T_camera_in_base
        T_camera_in_world = T_base_in_world @ T_camera_in_base # for the robocasa data, the camera pose stays the same across all frames of the video
        cam_pos = T_camera_in_world[:3,3]   # pos of camera in the world
        cam_rot_mat = T_camera_in_world[:3,:3]  # rot of camera in the world
        rotation = R.from_matrix(cam_rot_mat)
        cam_rot = rotation.as_euler('xyz') # angles in radians

        all_poses.append(cam_pos)
        all_poses.append(cam_rot)
        all_poses = np.array(all_poses)
        all_poses = np.concatenate([all_poses[:21]*0.0,all_poses])  # setting left hand values to zero
        if do_belief_map:
            belief_map = np.concatenate([belief_map*0.0, belief_map])
            valid_kp_mask = np.concatenate([valid_kp_mask*0.0, valid_kp_mask])

        return all_poses, cam_ext, belief_map, valid_kp_mask

    def process_data(self, file_loc, clip_name, indices):
        curr_frames = []
        actions = []
        with h5py.File(file_loc, 'r') as f:
            obs = f['data'][clip_name]['obs']
            if len(obs['robot0_base_pos'])<indices[-1]:
                print(len(obs['robot0_base_pos']), indices)

            # Determine image width based on patch_size
            if self.patch_size==14:
                self.img_width = 392
            elif self.patch_size==16:
                self.img_width = 384

            heatmaps = np.empty([len(indices)-1, len(self.hand_and_tip_keys)*2, 224, self.img_width])
            all_valid_kp = np.empty([len(indices)-1, len(self.hand_and_tip_keys)*2])
            for ii in range(len(indices)-1):
                idx = indices[ii]
                frame = self.image_transform(obs['robot0_robotview_2_image'][idx])
                curr_frames.append(frame)

                goal_idx = indices[ii+1]
                if ii==0:
                    curr_poses, cam_ext, _, _ = self.process_annotation(obs, idx, do_belief_map=False)
                else:
                    curr_poses, _, _, _ = self.process_annotation(obs, idx, cam_ext, do_belief_map=False)
                next_poses, _, belief_maps, valid_kp_mask = self.process_annotation(obs, goal_idx, cam_ext, do_belief_map=True)

                all_actions = np.concatenate([curr_poses, next_poses])
                actions.append(all_actions)
                heatmaps[ii] = belief_maps
                all_valid_kp[ii] = valid_kp_mask[:,0]

            idx = indices[-1]
            frame = self.image_transform(obs['robot0_robotview_2_image'][idx])
            curr_frames.append(frame)

        curr_frames = torch.stack(curr_frames)/255.
        curr_frames = tv_tensors.Video(curr_frames)
        if self.aug is not None:
            curr_frames = self.aug(curr_frames)
        else:
            if 'siglip' in self.backbone_name:
                curr_frames = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(curr_frames)
            else:
                curr_frames = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(curr_frames)
        actions = torch.Tensor(np.array(actions))
        heatmaps = torch.Tensor(heatmaps)
        all_valid_kp = torch.Tensor(all_valid_kp)
        return curr_frames, actions, heatmaps, all_valid_kp



    def __getitem__(self, idx):
        clip_id, frame_segment = self.idx_to_data[idx]
        [task_class, hdf5_file, clip_name] = self.all_files[clip_id]
        file_loc = os.path.join(self.data_root,task_class,hdf5_file)
        with h5py.File(file_loc) as f:
            len_vr = f['data'][clip_name]['obs']['robot0_robotview_2_image'].shape[0]
            h, w = f['data'][clip_name]['obs']['robot0_robotview_2_image'][0].shape[:2]
        if self.full_seq:
            frame_id = len_vr-1
            indices = random.sample(range(0,frame_id), min(self.num_context,len(range(0,frame_id))))  # some videos may be smaller than num_context
            indices = np.sort(indices).tolist()
        else:
            max_context_len = self.max_context_len
            frame_skip = int(self.max_context_len/self.num_context)
            if len_vr>max_context_len:
                frame_id_max = int(max_context_len + frame_segment*(len_vr-max_context_len))-1
                frame_id_min = frame_id_max-max_context_len
                frame_id = np.random.randint(low=frame_id_min, high=frame_id_max)
            else:
                max_context_len = (len_vr-1) // frame_skip * frame_skip  # closest multiple of frame_skip less than len(vr)
                frame_id = (len_vr-1) // frame_skip * frame_skip
            # frame_id represents the last frame in the (to be) selected sequence
            if self.var_time:
                indices = random.sample(range(frame_id-max_context_len, frame_id), min(self.num_context,len(range(frame_id-max_context_len, frame_id))))  # some videos may be smaller than num_context
                indices = np.sort(indices).tolist()
            else:
                indices = list(range(frame_id-max_context_len, frame_id, frame_skip))

        if len(indices)==0:
            indices = [0]

        if len(indices)<self.num_context:
            pad_count = self.num_context - len(indices)
            indices = [indices[0]] * pad_count + indices
        indices.append(frame_id)

        metadata={}
        metadata['vid_file'] = clip_name
        indices = np.sort(indices)
        indices = np.clip(indices, 0, len_vr)
        rel_t = indices[1:] - indices[:-1]
        metadata['indices'] = torch.Tensor(np.array(indices))

        curr_frames, actions, heatmaps, all_valid_kp = self.process_data(file_loc, clip_name, indices)

        return curr_frames, actions, rel_t, heatmaps, all_valid_kp, metadata
