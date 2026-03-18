import numpy as np
import torch
import os
import h5py
import cv2
import copy
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import pickle as pkl
import torchvision
from torchvision import tv_tensors
import torchvision.transforms.v2 as transforms
import random
import itertools
import matplotlib.pyplot as plt
import json

def pose_to_matrix(pose, order='xyz', degrees=False):
    x, y, z, rx, ry, rz = pose
    # Create rotation matrix from Euler angles
    rot = R.from_euler(order, [rx, ry, rz], degrees=degrees)
    R_mat = rot.as_matrix()  # 3x3 rotation matrix
    # Construct the 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [x, y, z]
    return T

class DroidDataset(Dataset):
    def __init__(self, root_folder, max_context_len=90, num_context=4, patch_size=14, img_size=224,
                context_frame_step=2, aug=None, backbone_name='dinov2', train=False,
                evaluate=False, var_time=False, num_keypoints=22):
        super(DroidDataset, self).__init__()

        self.data_root = root_folder

        with open('utils/droid_files.json', 'r') as f:
            self.all_files = json.load(f)

        all_files = self.all_files
        split_file_path = 'utils/split_indices_droid.json'
        if os.path.exists(split_file_path):
            with open(split_file_path, 'r') as f:
                split_data = json.load(f)
            train_files = split_data['train']
            test_files = split_data['test']
        else:
            random.shuffle(all_files)
            split_index = int(len(all_files) * 0.99)
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
        self.train = train
        self.var_time = var_time
        self.num_keypoints = num_keypoints
        assert not (self.train == False and aug==True), "self.train == False and aug == True must not happen simultaneously"
        if aug:
            if 'siglip' in backbone_name:
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
        clip_range = np.arange(0, len(self.all_files))
        frame_range = np.arange(1, 11)/10   # dividing the videos into different intervals so as not to load the same part everytime
        if evaluate:
            frame_range = np.array([0.99])
        self.idx_to_data = list(itertools.product(clip_range, frame_range))
        self.backbone_name = backbone_name

    def __len__(self):
        return len(self.idx_to_data)

    def image_transform(self, img, goal=False, do_flip=False):
        # img, pad = pad_and_resize(img, self.img_size)
        h, w = img.shape[:2]
        ar = w/h
        img = cv2.resize(img, (int(self.img_size*ar), self.img_size), interpolation=cv2.INTER_LINEAR)

        # Droid images are 1280x720
        # 224 x 398 -> (patch size 14: 392, patch size 16: 384)
        if self.patch_size==14:
            img = img[:,3:-3] # center cropping to the closest multiple of 14
        if self.patch_size==16:
            img = img[:, 7:-7] # center cropping to the closest multiple of 16

        img = torch.Tensor(img.copy()).permute(2,0,1)
        return img

    def sample_points_around(self, xyz, r1, r2, r3, r4):
        points = []
        x, y, z = xyz
        radii = [r1, r2, r3, r4]
        num_per_radius = 5
        for i in range(num_per_radius):
            for r in radii:
                theta = 2 * np.pi * i / num_per_radius
                px = x + r * np.cos(theta)
                py = y + r * np.sin(theta)
                pz = z
                points.append((px, py, pz))
        return np.array(points)

    def process_annotation(self, hdf5_file, T_w_in_cam, idx, prev_cam_ext=None):
        all_poses = []
        if prev_cam_ext is None:
            cam_ext = T_w_in_cam  # world to camera0 transformation
            cam_curr = T_w_in_cam
        else:
            cam_ext = prev_cam_ext  # points are represented in the frame of input (not goal) frame for consistency
            cam_curr = T_w_in_cam
        with h5py.File(hdf5_file,'r') as f:
            if idx>=len(f['observation']['robot_state']['cartesian_position']):
                return np.zeros([44,3]), cam_ext
            xyz = f['observation']['robot_state']['cartesian_position'][idx][:3]
            gripper_state = f['observation']['robot_state']['gripper_position'][idx]
            kps = [xyz]
            # dummy keypoints to imitate fingers are sampled on concentric circles near the end-effector
            if gripper_state<0.5:
                kps.extend(self.sample_points_around(xyz, 0.03, 0.04, 0.05, 0.06))
            else:
                kps.extend(self.sample_points_around(xyz, 0.03, 0.04, 0.035, 0.03))

        all_poses.extend(np.zeros([len(kps),3]).tolist())  # for left hand

        for kp in kps:
            pos = np.array(kp)  # pos of kp in the world frame
            pos_hom = np.append(pos,1.)[...,None]
            pos_cam = cam_ext @ pos_hom  # pos of kp in the camera frame
            all_poses.append(pos_cam[:3,0])

        T_cam_in_w = np.linalg.inv(T_w_in_cam)
        cam_pos = np.array(T_cam_in_w)[:3,3]   # pos of camera in the world frame
        cam_rot_mat = np.array(T_cam_in_w)[:3,:3]  # rot of camera in the world frame
        rotation = R.from_matrix(cam_rot_mat)
        cam_rot = rotation.as_euler('xyz') # angles in radians

        all_poses.append(cam_pos)
        all_poses.append(cam_rot)
        all_poses = np.array(all_poses)

        return all_poses, cam_ext

    def process_data(self, frames, hdf5_file, T_w_in_cam, indices):
        curr_frames = []
        actions = []
        for ii in range(len(indices)-1):
            idx = indices[ii]
            frame = self.image_transform(frames[idx].asnumpy())
            curr_frames.append(frame)

            goal_idx = indices[ii+1]
            if ii==0:
                curr_poses, cam_ext = self.process_annotation(hdf5_file, T_w_in_cam, idx)
            else:
                curr_poses, _ = self.process_annotation(hdf5_file, T_w_in_cam, idx, cam_ext)
            next_poses, _ = self.process_annotation(hdf5_file, T_w_in_cam, goal_idx, cam_ext)
            all_actions = np.concatenate([curr_poses, next_poses])
            actions.append(all_actions)

        idx = indices[-1]
        frame = self.image_transform(frames[idx].asnumpy())
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
        return curr_frames, actions


    def __getitem__(self, idx):
        while True:  # DROID dataset has some corrupted files
            task_id, frame_segment = self.idx_to_data[idx]
            json_file = self.all_files[task_id]
            data = json.load(open(json_file, 'r'))
            left_cam_file = data['left_mp4_path']
            if left_cam_file == data['ext1_mp4_path']:
                cam_name = 'ext1'
            elif left_cam_file == data['ext2_mp4_path']:
                cam_name = 'ext2'
            else:
                print('Camera name error')
                # Optionally, pick a new idx here as well
                idx = np.random.randint(0, len(self.idx_to_data))
                continue
            clip_name = data[f'{cam_name}_mp4_path'].split('/')[-1]
            clip_file = os.path.join('/', *json_file.split('/')[:-1], 'recordings/MP4', clip_name)
            try:
                vr = VideoReader(clip_file, num_threads=-1, ctx=cpu(0))
                if len(vr) < 8:
                    raise ValueError("Video too short")
                break  # Success: exit the loop
            except Exception:
                idx = np.random.randint(0, len(self.idx_to_data))
                continue

        h, w = vr[0].shape[:2]
        frames = vr
        max_context_len = self.max_context_len
        frame_skip = int(self.max_context_len/self.num_context)
        if len(vr)>max_context_len:
            frame_id_max = int(max_context_len + frame_segment*(len(vr)-max_context_len))-1
            frame_id_min = frame_id_max-max_context_len
            frame_id = np.random.randint(low=frame_id_min, high=frame_id_max)
        else:
            max_context_len = (len(vr)-1) // frame_skip * frame_skip  # closest multiple of frame_skip less than len(vr)
            frame_id = (len(vr)-1) // frame_skip * frame_skip

        h, w = frames[0].shape[:2]
        metadata={}
        metadata['vid_file'] = clip_file

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
        indices = np.sort(indices)
        indices = np.clip(indices, 0, len(vr))
        rel_t = indices[1:] - indices[:-1]
        metadata['indices'] = torch.Tensor(np.array(indices))

        cam_ext = data[f'{cam_name}_cam_extrinsics']
        T_cam_in_w = pose_to_matrix(cam_ext)
        T_w_in_cam = np.linalg.inv(T_cam_in_w)         # camera pose in the world frame

        hdf5_file = os.path.join('/',*json_file.split('/')[:-1],'trajectory.h5')

        curr_frames, actions = self.process_data(frames, hdf5_file, T_w_in_cam, indices)

        if self.patch_size==14:
            heatmaps = torch.zeros([self.num_context, self.num_keypoints, 224, 392])  # Creating zero-valued heatmaps for the keypoints. DexWM does not predict keypoints for DROID
        elif self.patch_size==16:
            heatmaps = torch.zeros([self.num_context, self.num_keypoints, 224, 384])
        valid_kp = torch.zeros([heatmaps.shape[0], heatmaps.shape[1]])   # T, kp

        return curr_frames, actions, rel_t, heatmaps, valid_kp, metadata
