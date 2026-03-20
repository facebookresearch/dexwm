# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------


from robosuite.models.grippers import GripperTester, RethinkGripper


def test_two_finger():
    two_finger_tester(False)


def two_finger_tester(render, total_iters=1, test_y=True):
    gripper = RethinkGripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=-0.07,
        gripper_high_pos=0.02,
        render=render,
    )
    tester.start_simulation()
    tester.loop(total_iters=total_iters, test_y=test_y)
    tester.close()


if __name__ == "__main__":
    two_finger_tester(True, 20, True)
