'''
This code probably doesn't work
'''
import numpy as np

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
        use_action_scaling=False,
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

        self.goal_link_xyz = None  # ← now store absolute xyz link poses as (16,3) array

    def set_goal(self, set_link_xyz,action=None,set_qpos=None):
        self.update()

        if set_link_xyz is not None:
            self.goal_link_xyz = set_link_xyz
            assert self.goal_link_xyz.shape == (18, 3), "set_link_xyz must be shape (16, 3)"
        else:
            raise ValueError("Must provide set_link_xyz for link pose control")

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_link_xyz)

    def run_controller(self):
        # if self.goal_link_xyz is None:
        #     self.goal_link_xyz = self.get_current_link_positions()

        self.update()

        if self.interpolator is not None:
            desired_link_xyz = self.interpolator.get_interpolated_goal()
        else:
            desired_link_xyz = self.goal_link_xyz
        # Convert desired link xyz poses to joint positions via inverse kinematics
        desired_qpos = self.inverse_kinematics(desired_link_xyz)
        # print("DESIRED QPOS",desired_qpos)
    # breakpoint()

        # Clamp to joint limits if defined
        if self.position_limits is not None:
            desired_qpos = np.clip(desired_qpos, self.position_limits[0], self.position_limits[1])
        self.use_action_scaling = False
        if self.use_action_scaling:
            ctrl_range = np.stack([self.actuator_min, self.actuator_max], axis=-1)
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])

            ctrl = desired_qpos - self.joint_pos  # PD style control signal
            ctrl = np.clip(ctrl, -1, 1)
            self.ctrl = bias + weight * ctrl
        else:
            self.ctrl = desired_qpos

        super().run_controller()
        return self.ctrl

    def reset_goal(self):
        self.goal_link_xyz = self.get_current_link_positions()
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_link_xyz)

    @property
    def control_limits(self):
        return self.input_min, self.input_max

    @property
    def name(self):
        return "LINK_POSE"
    def inverse_kinematics(self, target_link_xyz, max_iters=100, tol=1e-3, step_size=1e-1):
        """
        Approximates joint positions (qpos) using iterative Jacobian-based inverse kinematics.

        Args:
            target_link_xyz (np.ndarray): Desired link positions, shape (16, 3).
            max_iters (int): Maximum number of IK iterations.
            tol (float): Position error tolerance.
            step_size (float): Step size for the update.

        Returns:
            np.ndarray: Computed joint positions.
        """
        qpos = np.copy(self.joint_pos)  # initial guess

        for i in range(max_iters):
            # self.sim.data.qpos[self.joint_index["qpos"]] = qpos
            # mujoco.mj_forward(self.sim.model._model, self.sim.data._data)
            mujoco.mj_kinematics(self.sim.model._model, self.sim.data._data)


            current_positions = []
            for body_name in self.sim.data.model.body_names[17:35]:
                body_id = self.sim.model.body_name2id(body_name)
                pos = self.sim.data.body_xpos[body_id]
                current_positions.append(pos.copy())

            current_positions = np.array(current_positions)
            pos_err = target_link_xyz - current_positions
            error_norm = np.linalg.norm(pos_err)

            if error_norm < tol:
                break

            pos_err_flat = pos_err.flatten()

            J_full = []
            for body_name in self.sim.data.model.body_names[17:35]:
                body_id = self.sim.model.body_name2id(body_name)
                J_pos = np.zeros((3, self.sim.model.nv), dtype=np.float64)
                J_rot = np.zeros((3, self.sim.model.nv), dtype=np.float64)
                mujoco.mj_jacBodyCom(self.sim.model._model, self.sim.data._data, J_pos, J_rot, body_id)

                # Proper joint DOF indexing
                self.joint_qpos_indices = [ self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(joint_name)]for joint_name in self.joint_index]
                # breakpoint()
                # J_full.append(J_pos[:, self.joint_indexes["qpos"]])
                J_full.append(J_pos[:, self.joint_qpos_indices])


            J_full = np.vstack(J_full)

            dq = step_size * J_full.T @ pos_err_flat
            qpos += dq

            if self.position_limits is not None:
                qpos = np.clip(qpos, self.position_limits[0], self.position_limits[1])
        return qpos



    def get_current_link_positions(self):
        """
        Query simulation for current link xyz positions.
        Must implement based on sim API.
        """
        print("Warning: get_current_link_positions() is not implemented. Returning zeros.")
        return np.zeros((16, 3))
