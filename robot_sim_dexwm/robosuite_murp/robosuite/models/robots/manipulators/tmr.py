import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Tmr(ManipulatorModel):
    """
    Baxter is a hunky bimanual robot designed by Rethink Robotics.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right", "left"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/TMR/tmr2.xml"), idn=idn)
    
    def format_action(self, action):
        """
        Format the action to align with the actuator configuration of the Allegro Right Hand.
        - Thumb has 4 actuated joints.
        - Index, Middle, and Ring fingers each have 4 actuated joints.
        """
        assert len(action) == self.dof, "Action dimension mismatch!"
        # return np.clip(action, self.action_range[0], self.action_range[1])
        return action

    @property
    def default_base(self):
        return "MURP"

    @property
    def default_gripper(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific gripper names
        """
        return {"right": "WonikAllegro", "left": "WonikAllegro"}

    @property
    def default_controller_config(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific default controller config names
        """
        return {"right": "default_panda", "left": "default_panda"}

    @property
    def init_qpos(self):
        """
        Since this is bimanual robot, returns [right, left] array corresponding to respective values

        Note that this is a pose such that the arms are half extended

        Returns:
            np.array: default initial qpos for the right, left arms
        """
        # [right, left]
        # Arms half extended
        return np.array(
            [0.14936262, -0.65780519, -0.26952777, -2.65130757, 0.6578265,2.40055512, 0.56525831,-0.15623517, -0.52802177, 0.20902551, -2.50047633, -0.63356878, 2.29227613, 0.897565271]
        )

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.29, 0, 0),
            "table": lambda table_length: (-0.26 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, -1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific eef names
        """
        return {"right": "right_hand", "left": "left_hand"}
