import argparse
import json
import time
from collections import OrderedDict

import robosuite
from robosuite.controllers import load_composite_controller_config

import robocasa.macros as macros


def choose_option(
    options, option_name, show_keys=False, default=None, default_message=None
):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print("Here is a list of {}s:\n".format(option_name))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i, k, v))
        else:
            print("[{}] {}".format(i, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, or any other key for default ({}): ".format(
                len(options) - 1,
                default_message,
            )
        )
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))

    return choice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="task (choose among 100+ tasks)")
    parser.add_argument("--layout", type=int, help="kitchen layout (choose number 0-9)")
    parser.add_argument("--style", type=int, help="kitchen style (choose number 0-11)")
    parser.add_argument("--robot", type=str, help="robot")
    args = parser.parse_args()

    tasks = OrderedDict(
        [
            ("PnPCounterToCab", "pick and place from counter to cabinet"),
            ("PnPCounterToSink", "pick and place from counter to sink"),
            ("PnPMicrowaveToCounter", "pick and place from microwave to counter"),
            ("PnPStoveToCounter", "pick and place from stove to counter"),
            ("OpenSingleDoor", "open cabinet or microwave door"),
            ("CloseDrawer", "close drawer"),
            ("TurnOnMicrowave", "turn on microwave"),
            ("TurnOnSinkFaucet", "turn on sink faucet"),
            ("TurnOnStove", "turn on stove"),
            ("ArrangeVegetables", "arrange vegetables on a cutting board"),
            ("MicrowaveThawing", "place frozen food in microwave for thawing"),
            ("RestockPantry", "restock cans in pantry"),
            ("PreSoakPan", "prepare pan for washing"),
            ("PrepareCoffee", "make coffee"),
            ("PickPlaceCan", "pick and place"),
        ]
    )

    if args.task is None:
        args.task = choose_option(
            tasks, "task", default="PnPCounterToCab", show_keys=True
        )
    robots = OrderedDict([(0, "TMR_ROBOT"), (1, "GR1FloatingBody")])

    if args.robot is None:
        robot_choice = choose_option(
            robots, "robot", default=0, default_message="TMR_ROBOT"
        )
        args.robot = robots[robot_choice]

    print("THE ROBOT IS: ", args.robot)
    config = {
        "env_name": args.task,
        "robots": args.robot,
        "controller_configs": load_composite_controller_config(robot=args.robot),
        # "layout_ids": args.layout,
        # "style_ids": args.style,
        "translucent_robot": False,
    }

    # args.renderer = "mjviewer"

    print(colored(f"Initializing environment...", "yellow"))
    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="robot0_robotview",
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_heights=300,  # set camera height
        camera_widths=480,  # set camera width
        camera_names="robot0_robotview",  # use "agentview" camera
        camera_depths=True,
    )
    env_info = json.dumps(config)
    env.step()