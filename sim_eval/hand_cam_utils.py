import numpy as np
import cv2
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat
import math
import cv2
from scipy.spatial.transform import Rotation as R

def argmax(li,func = lambda x: x):
    index, max_val,max_el = None,None,None
    for i,el in enumerate(li):
        val = func(el)
        if max_val is None or val > max_val:
            index, max_val,max_el = i, val,el
    return index,max_el,max_val

# interable, comprabarable metric function -> max index, max element, max value
def argmin(li,func = lambda x: x):
    ind,el,val = argmax(li,lambda x: -func(x))
    return ind,el,-val

def mujoco_to_scipy_quat(q):
    return np.array([q[1], q[2], q[3], q[0]])

def estimate_object_pos(img, sim, vis=True, folder=None):
    T_z = np.array([
                    [-1, 0,  0, 0],
                    [0, -1,  0, 0],
                    [0, 0, 1, 0],
                    [0, 0,  0, 1]
                ])
    camera_name = 'gripper0_right_right_eye_in_hand'
    camera_id = sim.model.camera_name2id(camera_name)
    hand_body_id = sim.model.cam_bodyid[camera_id]
    # parent_body_name = sim.model.body_id2name(hand_body_id)
    # print(parent_body_name)

    hand_pos = sim.data.body_xpos[hand_body_id].copy()
    hand_quat = sim.data.body_xquat[hand_body_id].copy()
    hand_rot = R.from_quat(mujoco_to_scipy_quat(hand_quat)).as_matrix()

    # cam_pos_local = sim.model.cam_pos[camera_id]
    # cam_quat_local = sim.model.cam_quat[camera_id]
    # cam_rot_local = R.from_quat(mujoco_to_scipy_quat(cam_quat_local)).as_matrix()

    T_hand_in_world = np.eye(4)
    T_hand_in_world[:3,:3] = hand_rot
    T_hand_in_world[:3,3] = hand_pos

    T_cam_in_hand = np.array([[-0.2514527,  0.5712023, -0.7813447, -0.0212],
                                [0.0806303,  0.8168395,  0.5712023, 0.115],
                                [0.9645052,  0.0806303, -0.2514527, -0.042],
                                [0., 0., 0., 1.]])         # calculated from xml file, can also be done using mujoco sim
    T_cam_in_world = T_hand_in_world@T_cam_in_hand@(np.linalg.inv(T_z))      # the last term is required to change some image frame conventions

    fx, fy = 370.46914696, 370.46914696
    cx, cy = 480., 300.
    height = img.shape[0]
    width = img.shape[1]
    cammatrix = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)))
    center = np.array(img.shape[:2]) / 2

    meters = img
    bg_limit = 2.0  # everything beyond this can be considered as background
    bg = meters >= bg_limit
    # bg = (meters >= bg_limit) | (meters <= 0.1)
    median = np.median(meters[bg == False])
    diff = meters-median

    diff_adjusted = diff

    # floor_offset = -0.000 if mosaic else -0.01
    floor_offset = 0.00
    floor = diff_adjusted > floor_offset
    # objmask = np.loical_not(floor)

    min_depth = -median+0.1
    too_close = diff_adjusted < min_depth
    hand_pixels = np.zeros_like(diff_adjusted)
    hand_pixels[:, :280] = 1.0
    hand_pixels[:360, -220:] = 1.0
    hand_pixels[:190, :] = 1.0
    # Now, update objmask to exclude too-close points
    objmask = np.logical_not(floor) & np.logical_not(hand_pixels)


    # find connected regions and take closest that isn't too big or too small
    num,objim = cv2.connectedComponents(objmask.astype(np.uint8),)
    potential_objects = []
    for i in range(1,num+1):
        if (objim == i).sum() > 1000: # size of object
            indices = np.where((objim == i))
            # print(np.mean(meters[indices]))
            potential_objects.append(i)
    if vis:
    # if True:
        # plt.clf()
        # plt.imshow(meters)
        # plt.colorbar()
        # plt.show()
        plt.clf()
        plt.imshow(np.concatenate((bg,floor,objmask,objim,hand_pixels),axis=0))
        # plt.imshow(objmask)
        plt.colorbar()
        plt.savefig(f'{folder}/temp.png')
        plt.close()

    # no objects found
    visim = repeat(objim*255,'h w -> h w 3')
    if len(potential_objects) == 0:
        print('No Objects Detected')
        return None

    obj_locs = [np.stack(np.where(objim == po),axis=1).mean(axis=0) for po in potential_objects]
    ind,el,value = argmin(obj_locs,lambda x: np.linalg.norm(x-center))
    obj_loc = el
    p1,p2 = obj_loc
    # for obj_loc in obj_locs:
    if True:
        # p1, p2 = height-p1, width-p2
        plt.imshow(np.clip(meters,a_min=0,a_max=3), cmap='gray')
        plt.scatter(p2, p1, c='red', s=50, marker='x')  # Note: p2 is x, p1 is y
        plt.text(p2 + 5, p1 - 5, f"({p1:.0f}, {p2:.0f})", color='yellow', fontsize=9)
        plt.title("Depth Map with Object Location")
        plt.axis('off')
        plt.savefig(f'{folder}/plot2.png')
        plt.close()
    d = meters[int(p1),int(p2)]
    if d==0:
        print('Some problem with distance')
        print(d)
    # u,v = (obj_loc-center)
    d = -d      # mujoco cameras face in negative z-axis of the camera frame instead of positive, so adjusting it here
    u, v = p2, p1
    x_over_z = (u - cx)/ fx
    y_over_z = (v - cy)/ fy
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2) # d is w.r.t. the camera center point
    x = x_over_z * z
    y = y_over_z * z

    # z += 0.05 # to give it space to grasp
    p_cam = np.array((x,y,z)) # This is completely accurate
    p_cam_hom = np.append(p_cam,1.)[...,None]
    p_world_hom = T_cam_in_world@p_cam_hom
    p_world = p_world_hom[:3,0]
    return p_world

# [ 5.36419683 -1.70745739  1.06168507]
