# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------


from robosuite.models.grippers import GripperTester, AllegroRightHand


def test_panda_gripper():
    panda_gripper_tester(False)


def panda_gripper_tester(render, total_iters=1, test_y=True):
    gripper = AllegroRightHand()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=-0.10,
        gripper_high_pos=0.01,
        render=render,
    )
    tester.start_simulation()
    tester.loop(total_iters=total_iters, test_y=test_y)
    tester.close()


if __name__ == "__main__":
    panda_gripper_tester(True, 16, True)
    panda_gripper_tester(True, 16, True)
