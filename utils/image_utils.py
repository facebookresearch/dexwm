# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import random
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R
from scipy.ndimage.filters import gaussian_filter
import torch # type: ignore
import torchvision.transforms.functional as TVTransformsFunc # type: ignore
import torch.nn.functional as F

def create_belief_map(
        image_resolution,
        # image size (width x height)
        pointsBelief,
        # list of points to draw in a 7x2 tensor
        sigma=2
        # the size of the point
        # returns a tensor of n_points x h x w with the belief maps
    ):

    # Input argument handling
    assert (
        len(image_resolution) == 2
    ), 'Expected "image_resolution" to have length 2, but it has length {}.'.format(
        len(image_resolution)
    )
    image_width, image_height = image_resolution
    image_transpose_resolution = (image_height, image_width)
    out = np.zeros((len(pointsBelief), image_height, image_width))

    w = int(sigma * 2)

    for i_point, point in enumerate(pointsBelief):
        pixel_u = int(point[0])
        pixel_v = int(point[1])
        array = np.zeros(image_transpose_resolution)

        # TODO makes this dynamics so that 0,0 would generate a belief map.
        if (
            pixel_u - w >= 0
            and pixel_u + w + 1 < image_width
            and pixel_v - w >= 0
            and pixel_v + w + 1 < image_height
        ):
            for i in range(pixel_u - w, pixel_u + w + 1):
                for j in range(pixel_v - w, pixel_v + w + 1):
                    array[j, i] = np.exp(
                        -(
                            ((i - pixel_u) ** 2 + (j - pixel_v) ** 2)
                            / (2 * (sigma ** 2))
                        )
                    )
        out[i_point] = array

    return out

def get_keypoints_from_beliefmap(belief_maps):
    # Assuming belief_maps and gt_keypoints have shapes (B, 7, H, W)
    B, C, H, W = belief_maps.shape

    # Flatten the spatial dimensions (H, W) into a single dimension for argmax
    pred_flat = belief_maps.view(B, C, -1)  # (B, 7, H*W)

    # Get the argmax over the flattened dimension (which gives index in H*W)
    pred_max_idx = pred_flat.argmax(dim=-1)    # (B, 7)

    # Convert the 1D indices into 2D coordinates (row, col)
    pred_y = pred_max_idx // W                 # (B, 7), row coordinate
    pred_x = pred_max_idx % W                  # (B, 7), column coordinate

    # Stack the coordinates to get shape (B, 7, 2) for both predicted and ground truth keypoints
    pred_coords = torch.stack((pred_x, pred_y), dim=-1).float()  # (B, 7, 2)
    return pred_coords

def get_soft_keypoints_from_beliefmap(belief_maps, temperature=0.0001):
    # Assuming belief_maps has shape (B, 7, H, W)
    B, C, H, W = belief_maps.shape

    # Apply softmax to get probabilities for each pixel location
    belief_maps = belief_maps.view(B, C, -1)  # Flatten H*W to prepare for softmax
    softmax_maps = F.softmax(belief_maps / temperature, dim=-1).view(B, C, H, W)  # Reshape back to (B, C, H, W)

    # Create coordinate grids for x and y
    coords_x = torch.linspace(0, W - 1, W, device=belief_maps.device)
    coords_y = torch.linspace(0, H - 1, H, device=belief_maps.device)
    coords_x, coords_y = torch.meshgrid(coords_x, coords_y)
    coords_x = coords_x.t().view(1, 1, H, W)  # Transpose and reshape to (1, 1, H, W)
    coords_y = coords_y.t().view(1, 1, H, W)

    # Use softmax maps as weights to calculate expected x and y coordinates
    pred_x = (softmax_maps * coords_x).sum(dim=(2, 3))  # Weighted sum along H and W
    pred_y = (softmax_maps * coords_y).sum(dim=(2, 3))

    # Stack coordinates to get shape (B, 7, 2)
    pred_coords = torch.stack((pred_x, pred_y), dim=-1)  # Shape: (B, C, 2)
    return pred_coords

def get_xyz_from_kp(kp_uv, kp_z, K):
    # Extract intrinsics from K
    f_x = K[:, 0, 0]  # Focal length in x
    f_y = K[:, 1, 1]  # Focal length in y
    c_x = K[:, 0, 2]  # Principal point in x
    c_y = K[:, 1, 2]  # Principal point in y

    # Calculate the 3D coordinates (x, y, z) for each keypoint
    u = kp_uv[..., 0]  # Shape: (B, num_keypoints)
    v = kp_uv[..., 1]  # Shape: (B, num_keypoints)
    z = kp_z  # Shape: (B, num_keypoints)

    # Compute x, y, z in 3D
    x_world = (u - c_x.unsqueeze(1)) * z / f_x.unsqueeze(1)
    y_world = (v - c_y.unsqueeze(1)) * z / f_y.unsqueeze(1)
    z_world = z  # z is already the depth

    # Stack x, y, z to get the 3D coordinates
    kp_xyz = torch.stack((x_world, y_world, z_world), dim=-1)  # Shape: (B, num_keypoints, 3)
    return kp_xyz


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    Args:
        q: (B, 4) quaternion in (x, y, z, w) format.
    Returns:
        rot_matrix: (B, 3, 3) batched rotation matrices.
    """
    B = q.shape[0]
    # Normalize the quaternion to avoid errors due to non-unit quaternions
    q = F.normalize(q, p=2, dim=-1)

    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Rotation matrix from quaternion
    rot_matrix = torch.zeros((B, 3, 3), dtype=q.dtype, device=q.device)

    rot_matrix[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rot_matrix[:, 0, 1] = 2 * (x*y - z*w)
    rot_matrix[:, 0, 2] = 2 * (x*z + y*w)

    rot_matrix[:, 1, 0] = 2 * (x*y + z*w)
    rot_matrix[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rot_matrix[:, 1, 2] = 2 * (y*z - x*w)

    rot_matrix[:, 2, 0] = 2 * (x*z - y*w)
    rot_matrix[:, 2, 1] = 2 * (y*z + x*w)
    rot_matrix[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return rot_matrix


def batch_transform_points(points, t, q):
    """
    Transform the points from robot frame to camera frame using batched transformations.
    Args:
        points: (B, 7, 3) points in the robot frame.
        t: (B, 3) translation vectors.
        q: (B, 4) quaternions representing rotation.
    Returns:
        transformed_points: (B, 7, 3) points transformed into the camera frame.
    """
    B = points.shape[0]

    # Get the rotation matrix from the quaternion
    rot_matrix = quaternion_to_rotation_matrix(q)  # (B, 3, 3)

    # Apply the rotation
    transformed_points = torch.bmm(rot_matrix, points.transpose(1, 2)).transpose(1, 2)  # (B, 7, 3)

    # Apply the translation
    transformed_points = transformed_points + t.unsqueeze(1)  # (B, 7, 3)

    return transformed_points


# Define a function to calculate PCK at different thresholds
def calculate_pck(pred_keypoints, gt_keypoints, image_width=640, image_height=480, thresholds=[2.5,5.0,10.0]):
    # Calculate Euclidean distances between predicted and ground truth keypoints
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)
    distances = distances.reshape(-1, 1)
    valid_indices = np.where(
        (gt_keypoints[:, 0] > 0) & (gt_keypoints[:, 0] < image_width) &
        (gt_keypoints[:, 1] > 0) & (gt_keypoints[:, 1] < image_height)
    )[0]
    distances = distances[valid_indices]
    # Calculate PCK for each threshold
    pck_scores = []
    for threshold in thresholds:
        correct_keypoints = (distances <= threshold).sum()
        pck = (correct_keypoints / len(gt_keypoints)) * 100
        pck_scores.append(pck)

    return np.array(pck_scores)

def calculate_pck_batch(pred_keypoints, gt_keypoints, image_width=640, image_height=480, thresholds=[2.5, 5.0, 10.0]):
    """
    pred_keypoints: predicted keypoints of shape (B, 7, 2)
    gt_keypoints: ground truth keypoints of shape (B, 7, 2)
    image_width: width of the image
    image_height: height of the image
    thresholds: list of thresholds for PCK calculation
    """
    batch_size = pred_keypoints.shape[0]
    pck_scores = []
    for i in range(batch_size):
        scores = calculate_pck(pred_keypoints[i], gt_keypoints[i], image_width, image_height, thresholds)
        pck_scores.append(scores)
    return np.array(pck_scores)
