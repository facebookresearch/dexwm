# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from robosuite.wrappers import Wrapper
import numpy as np

class EvaluationWrapper(Wrapper):
    def __init__(self, env, task_name, target_hand_pose=None, target_obj_pos=None):
        """
        Initializes the evaluation wrapper.

        Args:
            env (MujocoEnv): The environment to monitor.
            task_name (str): Evaluation task. Choose out of reach, place, grasp
        """
        super().__init__(env)
        self.task_name = task_name
        self.target_hand_pose = target_hand_pose
        self.body_names_tips = ['robot0_right_hand', 'gripper0_right_link_3.0', 'gripper0_right_link_7.0', 'gripper0_right_link_11.0', 'gripper0_right_link_15.0']
        self.target_obj_pos = target_obj_pos

    def check_grasp_success(self):
        obj_name_id = 'obj'
        obj = self.env.objects[obj_name_id]
        obj_pos = np.array(self.env.sim.data.body_xpos[self.env.obj_body_id[obj.name]])

        hand_pos = np.array(
            self.sim.data.body_xpos[
                self.sim.model.body_name2id(self.robots[0].gripper["right"].root_body)
            ]
        )

        hand_contacts_obj = self.env.check_contact(
            self.env.robots[0].gripper["right"], self.env.objects[obj_name_id]
        )

        hand_obj_dist = np.linalg.norm(hand_pos-obj_pos)
        hand_obj_dist = hand_obj_dist<0.20

        success = hand_obj_dist and hand_contacts_obj
        return success

    def check_reach_success(self):
        curr_hand_pos = []
        for body_name in self.body_names_tips:
            body_id = self.env.sim.model.body_name2id(body_name)
            pos = self.env.sim.data.body_xpos[body_id]
            curr_hand_pos.append(pos.copy())
        curr_hand_pos = np.array(curr_hand_pos)
        dist = np.linalg.norm(curr_hand_pos-self.target_hand_pose,axis=1).mean()
        success = dist<0.15
        return success

    def check_place_success(self):
        obj_name_id = 'obj'
        obj = self.env.objects[obj_name_id]
        curr_obj_pos = np.array(self.env.sim.data.body_xpos[self.env.obj_body_id[obj.name]])
        dist = np.linalg.norm(curr_obj_pos-self.target_obj_pos)
        success = dist<0.1
        return success

    def check_success(self):
        if self.task_name=='grasp':
            success = self.check_grasp_success()
        elif self.task_name=='reach':
            success = self.check_reach_success()
        elif self.task_name=='place':
            success = self.check_place_success()
        return success

    def step(self, action):
        obs, _, _, _ = super().step(action)
        success = self.check_success()
        return obs, success
