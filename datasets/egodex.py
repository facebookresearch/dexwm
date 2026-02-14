import numpy as np
import torch
import os
import h5py
import cv2
import copy
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation as R
import pickle as pkl
from PIL import Image
import torchvision
from torchvision import tv_tensors
import torchvision.transforms.v2 as transforms
import random
import itertools
import matplotlib.pyplot as plt
import json
from utils.image_utils import create_belief_map

ALL_KEYS = {'right_keys': ['rightHand', 'rightThumbKnuckle', 'rightThumbIntermediateBase', 'rightThumbIntermediateTip', 'rightThumbTip', 'rightIndexFingerKnuckle', 'rightIndexFingerIntermediateBase', 'rightIndexFingerIntermediateTip', 'rightIndexFingerTip', 'rightMiddleFingerKnuckle', 'rightMiddleFingerIntermediateBase', 'rightMiddleFingerIntermediateTip', 'rightMiddleFingerTip', 'rightRingFingerKnuckle', 'rightRingFingerIntermediateBase', 'rightRingFingerIntermediateTip', 'rightRingFingerTip', 'rightLittleFingerKnuckle', 'rightLittleFingerIntermediateBase', 'rightLittleFingerIntermediateTip', 'rightLittleFingerTip'],
'left_keys': ['leftHand', 'leftThumbKnuckle', 'leftThumbIntermediateBase', 'leftThumbIntermediateTip', 'leftThumbTip', 'leftIndexFingerKnuckle', 'leftIndexFingerIntermediateBase', 'leftIndexFingerIntermediateTip', 'leftIndexFingerTip', 'leftMiddleFingerKnuckle', 'leftMiddleFingerIntermediateBase', 'leftMiddleFingerIntermediateTip', 'leftMiddleFingerTip', 'leftRingFingerKnuckle', 'leftRingFingerIntermediateBase', 'leftRingFingerIntermediateTip', 'leftRingFingerTip', 'leftLittleFingerKnuckle', 'leftLittleFingerIntermediateBase', 'leftLittleFingerIntermediateTip', 'leftLittleFingerTip'],
'body_keys': []}

NWM = {'right_keys': [], 'left_keys': [], 'body_keys': []}

PEVA = {'right_keys': ['rightArm', 'rightForearm', 'rightHand', 'rightShoulder'],
        'left_keys': ['leftArm', 'leftForearm', 'leftHand', 'leftShoulder'],
        'body_keys': ['hip', 'neck1', 'neck2', 'neck3', 'neck4', 'spine1', 'spine2', 'spine3', 'spine4', 'spine5', 'spine6', 'spine7']}

MODELS = {'all': ALL_KEYS, 'nwm': NWM, 'peva': PEVA}


def get_pixels_from_kp(xyz, extrinsic_matrix, patch_size):
    intrinsic_matrix = np.array([[736.6339, 0., 960.],
                        [0., 736.6339, 540.],
                        [0., 0., 1.]])
    X, Y, Z = xyz
    world_point = np.array([X, Y, Z, 1])
    camera_point = extrinsic_matrix @ world_point
    pixel_homogeneous = intrinsic_matrix @ camera_point[:3]
    u = pixel_homogeneous[0] / pixel_homogeneous[2]
    v = pixel_homogeneous[1] / pixel_homogeneous[2]
    if patch_size==14:
        u,v = u/1920*398-3, v/1080*224
    if patch_size==16:
        u,v = u/1920*398-7, v/1080*224
    return (u, v)

class EgoDexDataset(Dataset):
    def __init__(self, root_folder, max_context_len=90, num_context=4, patch_size=14, img_size=224, aug=None,
                backbone_name='dinov2', train=False, evaluate=False, keys='all', var_time=False):
        super(EgoDexDataset, self).__init__()
        if train:
            self.data_root = os.path.join(root_folder, 'train')
        else:
            self.data_root = os.path.join(root_folder, 'test')

        self.all_tasks = sorted(os.listdir(self.data_root))
        self.all_files = []
        for task in self.all_tasks:
            files = sorted(os.listdir(os.path.join(self.data_root, task)))
            files = [os.path.join(self.data_root, task, file) for file in files if file.endswith('.mp4')]
            self.all_files.extend(files)

        self.max_context_len = max_context_len
        self.num_context = num_context
        self.img_size = img_size
        self.patch_size = patch_size
        self.train = train
        self.keys = keys
        self.var_time = var_time
        self.all_keys = MODELS[keys]['left_keys'] + MODELS[keys]['right_keys'] + MODELS[keys]['body_keys']
        self.hand_and_tip_keys = [MODELS['all']['left_keys'][i] for i in [0, 4, 8, 12, 16, 20]] + [MODELS['all']['right_keys'][i] for i in [0, 4, 8, 12, 16, 20]]
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
        frame_range = np.arange(1, 11)/10   # dividing the videos into different intervals so as not to load the same part of the video everytime
        if evaluate:
            frame_range = np.array([0.99])

        self.idx_to_data = list(itertools.product(clip_range, frame_range))
        self.backbone_name = backbone_name

    def __len__(self):
        return len(self.idx_to_data)

    def image_transform(self, img):
        h, w = img.shape[:2]
        ar = w/h
        img = cv2.resize(img, (int(self.img_size*ar), self.img_size), interpolation=cv2.INTER_LINEAR)
        # EgoDex images are 1920 x 1080, so
        # 224 -> 224 x 398 (patch size 14: width 392, patch size 16: width 384)
        if self.patch_size==14:
            img = img[:,3:-3]  # center cropping to the closest multiple of 14
        elif self.patch_size==16:
            img = img[:,7:-7]  # center cropping to the closest multiple of 16

        img = torch.Tensor(img.copy()).permute(2,0,1)
        return img

    def process_annotation(self, annotations, idx, prev_cam_ext=None, do_belief_map=False):
        all_poses = []
        if prev_cam_ext is None:
            cam_ext = np.linalg.inv(np.array(annotations['transforms']['camera'][idx]))  # world to camera0 transformation
            cam_curr = np.linalg.inv(np.array(annotations['transforms']['camera'][idx]))
        else:
            cam_ext = prev_cam_ext  # points are represented in the frame of input (not goal) frame for consistency
            cam_curr = np.linalg.inv(np.array(annotations['transforms']['camera'][idx]))
        all_u = []
        all_v = []
        for key in self.all_keys:
            pos = np.array(annotations['transforms'][key][idx])[:3,3]  # pos of kp in the world frame
            pos_hom = np.append(pos,1.)[...,None] # 4 x 1
            pos_cam = cam_ext @ pos_hom    # pos of kp in the camera frame
            all_poses.append(pos_cam[:3,0])
        for key in self.hand_and_tip_keys:
            pos = np.array(annotations['transforms'][key][idx])[:3,3]  # pos of kp in the world
            pos_hom = np.append(pos,1.)[...,None] # 4 x 1
            pos_cam = cam_ext @ pos_hom   # pos of kp in the camera frame
            u,v = get_pixels_from_kp(pos_hom[:3,0], cam_curr, patch_size=self.patch_size)
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

        cam_pos = np.array(annotations['transforms']['camera'][idx])[:3,3]   # pos of camera in the world frame
        cam_rot_mat = np.array(annotations['transforms']['camera'][idx])[:3,:3]  # rot of camera in the world frame
        rotation = R.from_matrix(cam_rot_mat)
        cam_rot = rotation.as_euler('xyz') # angles in radians

        all_poses.append(cam_pos)
        all_poses.append(cam_rot)
        all_poses = np.array(all_poses)

        return all_poses, cam_ext, belief_map, valid_kp_mask

    def process_data(self, frames, annotations, indices):
        curr_frames = []
        actions = []
        # Determine image width based on patch_size
        if self.patch_size==14:
            self.img_width = 392
        elif self.patch_size==16:
            self.img_width = 384

        heatmaps = np.empty([len(indices)-1, len(self.hand_and_tip_keys), 224, self.img_width])
        all_valid_kp = np.empty([len(indices)-1, len(self.hand_and_tip_keys)])
        for ii in range(len(indices)-1):
            idx = indices[ii]
            frame = self.image_transform(frames[idx].asnumpy())
            curr_frames.append(frame)

            goal_idx = indices[ii+1]
            if ii==0:
                curr_poses, cam_ext, _, _ = self.process_annotation(annotations, idx, do_belief_map=False)
            else:
                curr_poses, _, _, _ = self.process_annotation(annotations, idx, cam_ext, do_belief_map=False)
            next_poses, _, belief_maps, valid_kp_mask = self.process_annotation(annotations, goal_idx, cam_ext, do_belief_map=True)

            all_actions = np.concatenate([curr_poses, next_poses])
            actions.append(all_actions)
            heatmaps[ii, :len(belief_maps)] = belief_maps                      # this is useful for running some baselines
            all_valid_kp[ii, :len(belief_maps)] = valid_kp_mask[:,0]

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
        heatmaps = torch.Tensor(heatmaps)
        all_valid_kp = torch.Tensor(all_valid_kp)
        return curr_frames, actions, heatmaps, all_valid_kp


    def __getitem__(self, idx):
        clip_id, frame_segment = self.idx_to_data[idx]
        clip_name = self.all_files[clip_id]
        clip_file = os.path.join(clip_name)
        vr = VideoReader(clip_file, num_threads=-1, ctx=cpu(0))
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

        # frame_id represents the last frame in the (to be) selected sequence

        h, w = frames[0].shape[:2]
        metadata={}
        metadata['vid_file'] = clip_file


        annotation_file = clip_file.replace('.mp4', '.hdf5')
        annotations = h5py.File(annotation_file, 'r')

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

        curr_frames, actions, heatmaps, all_valid_kp = self.process_data(frames, annotations, indices)

        return curr_frames, actions, rel_t, heatmaps, all_valid_kp, metadata
