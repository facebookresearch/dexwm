import numpy as np

from robosuite.models.robots import *


class PandaOmron(Panda):
    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Panda"}

    @property
    def init_qpos(self):
        return np.array([0, np.pi / 16.0 - 0.2, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.4, np.pi / 4])

    @property
    def init_torso_qpos(self):
        return np.array([0.2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.6, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }


class SpotWithArm(SpotArm):
    @property
    def default_base(self):
        return "Spot"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}

    @property
    def init_qpos(self):
        return np.array([0.0, -2, 1.26, -0.335, 0.862, 0.0])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-1.05, -0.1, -0.22),
            "empty": (-1.1, 0, -0.22),
            "table": lambda table_length: (-0.5 - table_length / 2, 0.0, -0.22),
        }


class SpotWithArmFloating(SpotArm):
    def __init__(self, idn=0):
        super().__init__(idn=idn)

    @property
    def init_qpos(self):
        return np.array([0.0, -2, 1.26, -0.335, 0.862, 0.0])

    @property
    def default_base(self):
        return "SpotFloating"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.7, -0.1, 0.0),
            "empty": (-0.6, 0, 0.0),
            "table": lambda table_length: (-0.5 - table_length / 2, 0.0, 0.0),
        }


class PandaDexRH(Panda):
    @property
    def default_gripper(self):
        return {"right": "InspireRightHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [-0.5, 0.5, 0.5, -0.5]}


class PandaDexLH(Panda):
    @property
    def default_gripper(self):
        return {"right": "InspireLeftHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.5, -0.5, 0.5, -0.5]}

class PandaAllegro(Panda):
    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_gripper(self):
        return {"right": "WonikAllegro"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.1]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [1.0, 0.0, 0.0, 0.0]}

class fr3Allegro(fr3):
    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "fr3"}

    @property
    def default_gripper(self):
        return {"right": "WonikAllegro"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.1]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [1.0, 0.0 ,0.0 ,0.0]}

class TMR_ROBOT(Tmr):
    @property
    def default_base(self):
        return "MURP"

    # @property
    # def default_arms(self):
    #     return {"right": "fr3"}
    @property
    def default_gripper(self):
        # return {"right": "WonikAllegro","left": "WonikAllegro"}
        return {"right": "WonikAllegro","left": "WonikAllegroLeft"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.1],"left": [0.0, 0.0, 0.1]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.9238795, 0.0 ,0.0 ,0.3826834],"left": [-0.3826834, 0.0 ,0.0 ,0.9238795]}
