# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------


from robosuite.models.grippers import GRIPPER_MAPPING


def test_all_gripper():
    for name, gripper in GRIPPER_MAPPING.items():
        # Test all grippers except the null gripper
        if name not in {None, "WipingGripper"}:
            print("Testing {}...".format(name))
            _test_gripper(gripper())


def _test_gripper(gripper):
    action = gripper.format_action([1] * gripper.dof)
    assert action is not None

    assert gripper.init_qpos is not None
    assert len(gripper.init_qpos) == len(gripper.joints)


if __name__ == "__main__":
    test_all_gripper()
    print("Gripper tests completed.")
