"""This is a controller that controls the fingers / grippers to do naive gripping. No matter how many fingers the gripper has, they all move in the same direction."""

import numpy as np

from robosuite.controllers.parts.gripper.gripper_controller import GripperController
from robosuite.utils.control_utils import *

# Supported impedance modes

class SimpleGripController(GripperController):
    def __init__(
        self,
        sim,
        joint_indexes,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=1,
        output_min=-1,
        policy_freq=20,
        qpos_limits=None,
        interpolator=None,
        use_action_scaling=True,
        **kwargs,
    ):
        super().__init__(
            sim,
            joint_indexes,
            actuator_range,
            part_name=kwargs.get("part_name", None),
            naming_prefix=kwargs.get("naming_prefix", None),
        )

        self.control_dim = len(joint_indexes["actuators"])
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        self.position_limits = np.array(qpos_limits) if qpos_limits is not None else None
        self.control_freq = policy_freq
        self.interpolator = interpolator
        self.use_action_scaling = use_action_scaling

        self.goal_qpos = None  # ← now we store absolute joint positions

    def set_goal(self, action, set_qpos=None):
        self.update()

        if set_qpos is not None:
            self.goal_qpos = np.array(set_qpos)
        else:
            delta = np.array(action)
            assert len(delta) == self.control_dim

            if self.use_action_scaling:
                delta = self.scale_action(delta)

            self.goal_qpos = self.joint_pos + delta  # absolute goal

        # clamp to joint limits
        if self.position_limits is not None:
            self.goal_qpos = np.clip(self.goal_qpos, self.position_limits[0], self.position_limits[1])

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qpos)

    def run_controller(self):
        if self.goal_qpos is None:
            self.set_goal(np.zeros(self.control_dim))

        self.update()

        if self.interpolator is not None:
            desired_qpos = self.interpolator.get_interpolated_goal()
        else:
            desired_qpos = self.goal_qpos

        # action scaling
        if self.use_action_scaling:
            ctrl_range = np.stack([self.actuator_min, self.actuator_max], axis=-1)
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            ctrl = (desired_qpos - self.joint_pos)  # simple PD-style (can tune later)
            ctrl = np.clip(ctrl, -1, 1)
            self.ctrl = bias + weight * ctrl
        else:
            self.ctrl = desired_qpos

        super().run_controller()
        return self.ctrl

    def reset_goal(self):
        self.goal_qpos = self.joint_pos
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qpos)

    @property
    def control_limits(self):
        return self.input_min, self.input_max

    @property
    def name(self):
        return "JOINT_POSITION"

    def update(self):
        super().update()
        # make sure joint_pos is up to date
        self.joint_pos = self.sim.data.qpos[self.qpos_index]
