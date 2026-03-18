import random
import numpy as np
import robosuite.utils.transform_utils as T

PICK_TEXT_BANK = (
    "pick up the {object}",
    "hey robot, could you please pick up the {object}",
    "I want to get {object}",
    "could you fetch {object}",
    "I want to find {object}",
    "return {object} to me",
    "find the {object} for me",
    "get {object}",
    "please find {object}",
    "hey murp robot, pick up {object} for me",
    "where could I find {object}",
    "is there a {object}? if yes, could you please fetch it for me",
    "get the {object}",
    "pick the {object} up",
    "grasp a {object}",
    "grab the {object} up",
    "grab the {object}",
    "fetch the {object}",
    "I need {object}",
    "hi robot, fetch the {object}",
    "grasp the {object}",
    "find {object}",
    "search for {object}",
    "pick up {object} from the counter",
    "pick up {object} on the table",
    "pick up {object} on the surface",
    "see if there is a {object} in front of you",
    "I want to find {object} on the counter",
    "get the {object} from the counter",
    "pick the {object} up on the table",
    "pick the {object} up in front of you",
    "please pick up the {object}",
    "pick up {object}",
    "hey murp, could you please pick up {object} for me",
    "get {object} for me",
    "hand over {object}",
    "perform the action of picking {object} up",
    "you are a helpful robot that follows my command: could you please fetch {object} for me",
)

PLACE_TEXT_BANK = {
    "next-to-object": (
        "move to the next of {object}",
        "place it to the next of {object}",
        "place to the next of {object}",
        "drop it next to the {object}",
        "drop it so that it is close to the {object}",
    ),
    "right-of-object": (
        "move to the right of {object}",
        "place it to the right of {object}",
        "place to the right of {object}",
        "drop it right to the {object}",
        "move to the next of {object}",
        "place it to the next of {object}",
        "place to the next of {object}",
        "drop it next to the {object}",
        "drop it so that it is close to the {object}",
    ),
    "left-of-object": (
        "move to the left of {object}",
        "place it to the left of {object}",
        "place to the left of {object}",
        "drop it left to the {object}",
        "move to the next of {object}",
        "place it to the next of {object}",
        "place to the next of {object}",
        "drop it next to the {object}",
        "drop it so that it is close to the {object}",
    ),
    "behind-of-object": (
        "move it to the behind of {object}",
        "place it to the behind of {object}",
        "place to the behind of {object}",
        "drop it to the behind of {object}",
    ),
    "in-front-of-object": (
        "move it to the in front of {object}",
        "place it to the in front of {object}",
        "place to the in front of {object}",
        "drop it in front of the {object}",
    ),
    "move-right": (
        "place it to the right",
        "move to the right",
        "move it to the right",
        "drop to the right",
        "move right",
    ),
    "move-left": (
        "place it to the left",
        "move to the left",
        "move it to the left",
        "drop to the left",
        "move left",
    ),
}

CONJUNCTION_TEXT_BANK = (
    " and ",
    " and then ",
    ", then ",
    "; ",
    ", ",
    ", finally ",
    ", and next ",
)

SIMPLE_PICK_TEXT_BANK = ("pick up the {object}",)
SIMPLE_PLACE_TEXT_BANK = {
    "next-to-object": ("place it to the next of the {object}",),
    "right-of-object": ("place it to the right of the {object}",),
    "left-of-object": ("place it to the left of the {object}",),
    "behind-of-object": ("place it to the behind of the {object}",),
    "in-front-of-object": ("place it to the in front of the {object}",),
    "move-right": ("place it to the right",),
    "move-left": ("place it to the left",),
}


def instruction_anotation(env):
    """Anotation of the instruction"""
    print("\n-----Starting automatic instruction anotation-----\n")

    target_obj_name = env.env.get_obj_lang()
    pick_text = f'{random.choice(PICK_TEXT_BANK).replace("{object}", target_obj_name)}'
    simple_pick_text = (
        f'{random.choice(SIMPLE_PICK_TEXT_BANK).replace("{object}", target_obj_name)}'
    )

    target_obj = env.env.objects["obj"]
    final_target_obj_pos = env.env.sim.data.body_xpos[
        env.env.obj_body_id[target_obj.name]
    ]

    min_dis_to_distractor = float("inf")
    min_dis_of_distractor = None
    # Get the object that is closet to the target
    for obj_name in env.env.objects:
        if obj_name != target_obj.name:
            # Get distance
            obj_pos = env.env.sim.data.body_xpos[env.env.obj_body_id[obj_name]]
            _min = np.linalg.norm(obj_pos - final_target_obj_pos)
            print(
                f"Object: {obj_name} {env.env.get_obj_lang(obj_name)} to the target {target_obj_name} with the distance: {_min}m"
            )
            if min_dis_to_distractor > _min:
                min_dis_to_distractor = _min
                min_dis_of_distractor = env.env.objects[obj_name]

    if min_dis_of_distractor is not None:
        has_distractor = True
    else:
        has_distractor = False

    if has_distractor:
        print(
            f"Object {env.env.get_obj_lang(min_dis_of_distractor.name)} is closet with distance of {min_dis_to_distractor}m"
        )

    # Now compute the location of the object to the robot base
    base_T_world = T.make_pose(
        env.env.sim.data.get_body_xpos("mobilebase0_base"),
        env.env.sim.data.get_body_xmat("mobilebase0_base"),
    )
    target_obj_T_world = T.make_pose(
        env.env.sim.data.get_body_xpos(target_obj.name + "_main"),
        env.env.sim.data.get_body_xmat(target_obj.name + "_main"),
    )
    if has_distractor:
        distractor_obj_T_world = T.make_pose(
            env.env.sim.data.get_body_xpos(min_dis_of_distractor.name + "_main"),
            env.env.sim.data.get_body_xmat(min_dis_of_distractor.name + "_main"),
        )
        final_distractor_obj_T_base_pos = (
            T.pose_inv(base_T_world) @ distractor_obj_T_world
        )[:3, 3]
    final_target_obj_T_base_pos = (T.pose_inv(base_T_world) @ target_obj_T_world)[:3, 3]

    # Get the inital object location
    init_target_obj_pos = env.env.object_placements[target_obj.name][0]
    init_target_obj_rot = env.env.object_placements[target_obj.name][1]
    init_target_obj_T_world = T.make_pose(
        init_target_obj_pos, T.quat2mat(init_target_obj_rot)
    )
    init_target_obj_T_base_pos = (T.pose_inv(base_T_world) @ init_target_obj_T_world)[
        :3, 3
    ]

    print(
        f"Location change of the target object: {init_target_obj_T_base_pos} -> {final_target_obj_T_base_pos}"
    )
    if has_distractor:
        print(f"Location of the closet distractor: {final_distractor_obj_T_base_pos}")

    #         X
    #         ^
    #         |
    # Y <-----|-----
    #         |
    #         |

    # Now generate the place text
    key_for_place_text = None

    rand_num = random.random()
    if rand_num < 0.15:
        print("Case 1: not adding the place text")
        # not adding the text
        pass
    elif rand_num < 0.35 or min_dis_to_distractor > 0.25 or (not has_distractor):
        # simple move right / move left in y if random number is smaller than 0.3
        # or the distance between the target and the distractor is large
        print("Case 2: adding the move left and right place text")
        delta_y = final_target_obj_T_base_pos[1] - init_target_obj_T_base_pos[1]
        if delta_y > 0:
            # move left
            key_for_place_text = "move-left"
        else:
            # move right
            key_for_place_text = "move-right"
    else:
        print("Case 3: adding the spatial place text")
        delta_xyz = final_distractor_obj_T_base_pos - final_target_obj_T_base_pos
        abs_delta_xy = abs(delta_xyz[0:-1])  # only consider x y, no z
        # delta is smaller in x, and difference in x and y is large enough
        # this means that the object is at left or right of the distractor
        if abs_delta_xy[0] * 5 < abs_delta_xy[1]:
            if delta_xyz[1] > 0:
                key_for_place_text = "right-of-object"
            else:
                key_for_place_text = "left-of-object"

        elif abs_delta_xy[0] > abs_delta_xy[1] * 5:
            if delta_xyz[0] > 0:
                key_for_place_text = "in-front-of-object"
            else:
                key_for_place_text = "behind-of-object"
        else:
            key_for_place_text = "next-to-object"

    if key_for_place_text in [
        "right-of-object",
        "left-of-object",
        "in-front-of-object",
        "behind-of-object",
        "next-to-object",
    ]:
        place_text = random.choice(PLACE_TEXT_BANK[key_for_place_text]).replace(
            "{object}", f"{env.env.get_obj_lang(min_dis_of_distractor.name)}"
        )
        conjunction_text = random.choice(CONJUNCTION_TEXT_BANK)
        final_text = f"{pick_text}{conjunction_text}{place_text}"

        simple_place_text = random.choice(
            SIMPLE_PLACE_TEXT_BANK[key_for_place_text]
        ).replace("{object}", f"{env.env.get_obj_lang(min_dis_of_distractor.name)}")
        simple_final_text = f"{simple_pick_text} and {simple_place_text}"
    elif key_for_place_text in ["move-left", "move-right"]:
        place_text = random.choice(PLACE_TEXT_BANK[key_for_place_text]).replace(
            "{object}", ""
        )
        conjunction_text = random.choice(CONJUNCTION_TEXT_BANK)
        final_text = f"{pick_text}{conjunction_text}{place_text}"

        simple_place_text = random.choice(
            SIMPLE_PLACE_TEXT_BANK[key_for_place_text]
        ).replace("{object}", "")
        simple_final_text = f"{simple_pick_text} and {simple_place_text}"
    else:
        final_text = f"{pick_text}"
        simple_final_text = f"{simple_pick_text}"

    print(
        f"Final free-formed/simple instructions: {final_text} / {simple_final_text} / with the key_for_place_text of {key_for_place_text}"
    )

    return final_text, simple_final_text
