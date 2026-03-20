# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from torch.utils.data import Dataset
from datasets.egodex import EgoDexDataset
from datasets.droid import DroidDataset

class EgodexDroidDataset(Dataset):
    def __init__(self, egodex_root_folder, droid_root_folder,
                max_context_len=90, num_context=4,  patch_size=14, img_size=224,
                context_frame_step=2, aug=False, backbone_name='dinov2', train=False, keys='all', var_time=False):
        super(EgodexDroidDataset, self).__init__()

        self.img_size = img_size
        self.num_context = num_context
        self.backbone_name = backbone_name
        self.egodex_dataset = EgoDexDataset(root_folder=egodex_root_folder, max_context_len=max_context_len, num_context=num_context, patch_size=patch_size,
                                    backbone_name=backbone_name, img_size=img_size, aug=aug, train=train, keys=keys, var_time=var_time)

        self.droid_dataset = DroidDataset(root_folder=droid_root_folder, max_context_len=max_context_len//2, num_context=num_context, patch_size=patch_size,
                                    backbone_name=backbone_name, img_size=img_size, aug=aug, train=train, var_time=var_time,
                                    num_keypoints=len(self.egodex_dataset.hand_and_tip_keys))  # max_context_len//2 because droid at 15Hz, others at 30Hz

        self.len_egodex = len(self.egodex_dataset)
        self.len_droid = len(self.droid_dataset)

    def __len__(self):
        return self.len_egodex + self.len_droid

    def __getitem__(self, idx):
        if idx<self.len_egodex:
            curr_frames, actions, rel_t, heatmaps, valid_kp, metadata = self.egodex_dataset.__getitem__(idx)
        else:
            curr_frames, actions, rel_t, heatmaps, valid_kp, metadata = self.droid_dataset.__getitem__(idx-self.len_egodex)

        return curr_frames, actions, rel_t, heatmaps, valid_kp, metadata
