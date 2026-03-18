import numpy as np

from robosuite.models.bases.mobile_base_model import MobileBaseModel
from robosuite.utils.mjcf_utils import xml_path_completion
# from robosuite.models.bases.mount_model import MountModel


class MURP(MobileBaseModel):
    """
    MURP Mobile Base.

    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("bases/murp3.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.array((0, 0, 0.0))

    @property
    def horizontal_radius(self):
        # TODO: This may be inaccurate; just a placeholder for now
        return 0.25
    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0])