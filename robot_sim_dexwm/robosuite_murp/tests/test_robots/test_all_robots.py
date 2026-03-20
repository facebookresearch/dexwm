# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot, LeggedRobot, WheeledRobot


def test_all_robots():
    for name, robot in ROBOT_CLASS_MAPPING.items():
        print(f"Testing {name}")
        if robot not in [FixedBaseRobot, WheeledRobot, LeggedRobot]:
            raise ValueError(f"Invalid robot type: {robot}")
        else:
            _test_contact_geoms(robot(name))


def _test_contact_geoms(robot):
    robot.load_model()
    contact_geoms = robot.robot_model._contact_geoms
    for geom in contact_geoms:
        assert isinstance(geom, str), f"The geom {geom} is of type {type(geom)}, but should be {type('placeholder')}"


if __name__ == "__main__":
    # test_single_arm_robots()
    test_all_robots()
    print("Robot tests completed.")
