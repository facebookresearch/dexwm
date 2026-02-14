import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion

"""
Dexterous hands for the Wonik Allegro Hand (Right).
"""


class AllegroLeftHand(GripperModel):
    """
    Dexterous right hand of the Wonik Allegro Hand.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance.
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/wonik_3.xml"), idn=idn)

    def format_action(self, action):
        """
        Format the action to align with the actuator configuration of the Allegro Right Hand.
        - Thumb has 4 actuated joints.
        - Index, Middle, and Ring fingers each have 4 actuated joints.
        """
        assert len(action) == self.dof, "Action dimension mismatch!"
        return action
        # assert len(action) == self.dof
        # action = np.array(action)
        # indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8 ,9 ,10, 11, 12 ,13 ,14,15])
        # breakpoint()
        # return action[indices]
        # action = np.array(action)
        # indices = np.array([4, 4, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3,3, 3])
        # return action[indices]

    @property
    def init_qpos(self):
        """
        Initial joint positions for the Allegro Right Hand.
        """
        return np.zeros(self.dof)

    @property
    def speed(self):
        """
        Speed of the Allegro Right Hand actuators.
        """
        return 0.1

    @property
    def dof(self):
        """
        Degrees of freedom for the Allegro Right Hand.
        - Thumb: 4 joints
        - Index: 4 joints
        - Middle: 4 joints
        - Ring: 4 joints
        """
        return 16

    @property
    def action_range(self):
        """
        Define the valid range for actuator controls based on the XML joint limits.
        """
        # Joint limits based on the XML file
        return np.array(
            [
                -0.47,
                -0.196,
                -0.174,
                -0.227,  # Index finger
                -0.47,
                -0.196,
                -0.174,
                -0.227,  # Middle finger
                -0.47,
                -0.196,
                -0.174,
                -0.227,  # Ring finger
                0.263,
                -0.105,
                -0.189,
                -0.162,  # Thumb
            ]
        ), np.array(
            [
                0.47,
                1.61,
                1.709,
                1.618,  # Index finger
                0.47,
                1.61,
                1.709,
                1.618,  # Middle finger
                0.47,
                1.61,
                1.709,
                1.618,  # Ring finger
                1.396,
                1.163,
                1.644,
                1.719,  # Thumb
            ]
        )

    @property
    def _important_geoms(self):
        return {
            "right_hand": [
                "g0_col",
                "g1_col",
                "g2_col",
                "g3_col",
                "g4_col",
                "g5_col",
                "g6_col",
                "g7_col",
                "g8_col",
                "g9_col",
                "g10_col",
                "g11_col",
                "g12_col",
                "g13_col",
                "g14_col",
                "g15_col",
            ],
        }

    @property
    def grasp_qpos(self):

        return {
            -1: np.array(
                [
                    0.04918514937162399,
                    -0.019272545352578163,
                    0.014429383911192417,
                    -0.01086236909031868,
                    0.09369415044784546,
                    0.03855705261230469,
                    -0.032034676522016525,
                    -0.011998646892607212,
                    0.012606220319867134,
                    0.008006482385098934,
                    -0.01866091787815094,
                    -0.01981888897716999,
                    0.24044487476348877,
                    -0.24079933762550354,
                    0.16510280966758728,
                    -0.1014380231499672,
                ]
            ),
            0: np.array(
                [
                    0.1091851532459259,
                    0.30000001192092896,
                    0.4744293987751007,
                    0.800000011920929,
                    -0.18630585074424744,
                    0.2585570514202118,
                    0.8379653096199036,
                    0.3680013418197632,
                    0.012606220319867134,
                    0.20800648629665375,
                    0.6213390827178955,
                    0.6001811027526855,
                    1.3960000276565552,
                    0.37299999594688416,
                    -0.019999999552965164,
                    0.29999998211860657,
                ]
            ),
            1: np.array(
                [
                    0.10999999940395355,
                    1.090000033378601,
                    1.1444294333457947,
                    0.40000001192092896,
                    0.15000000596046448,
                    1.090000033378601,
                    1.1700000047683716,
                    -0.0019986582919955254,
                    0.012606220319867134,
                    1.190000033378601,
                    0.791339099407196,
                    0.3501810997724533,
                    1.3960000276565552,
                    0.6330000162124634,
                    0.6200000047683716,
                    0.51999998688697815,
                ]
            ),
            2: np.array(
                [
                    0.10999999940395355,
                    1.190000033378601,
                    1.1444294333457947,
                    0.50000001192092896,
                    0.15000000596046448,
                    1.190000033378601,
                    1.1700000047683716,
                    -0.0019986582919955254,
                    0.012606220319867134,
                    1.190000033378601,
                    0.891339099407196,
                    0.5501810997724533,
                    1.3960000276565552,
                    0.6330000162124634,
                    0.6200000047683716,
                    0.51999998688697815,
                ]
            ),
        }
