from robocasa.environments.kitchen.kitchen import *

from robocasa.models.objects.kitch_min_obj import OBJ_CATEGORIES, OBJ_GROUPS
from collections import defaultdict
from scipy.spatial.transform import Rotation as R


class PnP(Kitchen):
    """
    Class encapsulating the atomic pick and place tasks.

    Args:
        obj_groups (str): Object groups to sample the target object from.

        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
    """

    def __init__(self, obj_groups="all", exclude_obj_groups=None, *args, **kwargs):
        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups

        super().__init__(*args, **kwargs)

    def _get_obj_cfgs(self):
        raise NotImplementedError


class PnPCounterToCab(PnP):
    """
    Class encapsulating the atomic counter to cabinet pick and place task

    Args:
        cab_id (str): The cabinet fixture id to place the object.

        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(
        self, cab_id=FixtureType.CABINET_TOP, obj_groups="all", *args, **kwargs
    ):

        self.cab_id = cab_id
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to cabinet pick and place task:
        The cabinet to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to cabinet pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the counter and place it in the cabinet"
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to cabinet pick and place task.
        Puts the target object in the front area of the counter. Puts a distractor object on the counter
        and the back area of the cabinet.

        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.60, 0.30),
                    pos=(0.0, -1.0),
                    offset=(0.0, -0.01),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the counter to cabinet pick and place task is successful.
        Checks if the object is inside the cabinet and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        # obj_inside_cab = OU.obj_inside_of(self, "obj", self.cab)
        # gripper_obj_far = OU.gripper_obj_far(self)
        # return obj_inside_cab and gripper_obj_far
        obj = self.objects["obj"]
        start_pos = self.sim.data.body_xpos[
            self.obj_body_id[self.objects["distr_counter"].name]
        ]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])
        obj_z = obj_pos[2]
        obj_lifted = obj_z >= 1.1

        return obj_lifted


class PnPCabToCounter(PnP):
    """
    Class encapsulating the atomic cabinet to counter pick and place task

    Args:
        cab_id (str): The cabinet fixture id to pick the object from.

        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(
        self, cab_id=FixtureType.CABINET_TOP, obj_groups="all", *args, **kwargs
    ):
        self.cab_id = cab_id
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the cabinet to counter pick and place task:
        The cabinet to pick object from and the counter to place it on
        """
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab",
            dict(id=self.cab_id),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.cab),
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        """
        Get the episode metadata for the cabinet to counter pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the cabinet and place it on the counter"
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the cabinet to counter pick and place task.
        Puts the target object in the front area of the cabinet. Puts a distractor object on the counter
        and the back area of the cabinet.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the cabinet to counter pick and place task is successful.
        Checks if the object is on the counter and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        gripper_obj_far = OU.gripper_obj_far(self)
        obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
        return obj_on_counter and gripper_obj_far


class PnPCounterToSink(PnP):
    """
    Class encapsulating the atomic counter to sink pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, obj_groups="all", *args, **kwargs):

        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to sink pick and place task:
        The sink to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to sink pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the counter and place it in the sink"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to sink pick and place task.
        Puts the target object in the front area of the counter. Puts a distractor object on the counter
        and the sink.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.counter,
                        loc="left_right",
                    ),
                    size=(0.30, 0.40),
                    pos=("ref", -1.01),
                    offset=(0.0, -0.01),
                ),
            )
        )
        exclude_cats = [self.obj_category] if hasattr(self, "obj_category") else []
        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups=self.obj_groups,
                exclude_obj_groups=exclude_cats,
                split="B",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.40),
                    pos=("ref", 0.1),
                    offset=(0.3, -0.001),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_counter_2",
                obj_groups=self.obj_groups,
                exclude_obj_groups=exclude_cats,
                split="B",
                placement=dict(
                    fixture=self.sink,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.40),
                    pos=(0.1, -0.1),
                    offset=(0.0, 0.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_counter_3",
                obj_groups=self.obj_groups,
                exclude_obj_groups=exclude_cats,
                split="B",
                graspable=False,
                placement=dict(
                    fixture=self.sink,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=(0.1, 0.3),
                    offset=(0.0, 0.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_sink",
                obj_groups="all",
                exclude_obj_groups=exclude_cats,
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.25, 0.25),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the counter to sink pick and place task is successful.
        Checks if the object is inside the sink and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        ######### LIFT SUCCESS CONDITION ###################
        # obj = self.objects["obj"]
        # start_pos = self.sim.data.body_xpos[
        #     self.obj_body_id[self.objects["distr_counter"].name]
        # ]
        # obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])
        # obj_z = obj_pos[2]
        # obj_lifted = obj_z >= 1.1

        # return obj_lifted
        ######################################################
        obj_in_sink = OU.obj_inside_of(self, "obj", self.sink, partial_check=True)
        # gripper_obj_far = OU.gripper_obj_far(self)
        return obj_in_sink


class PnPLift(PnP):
    """
    Class encapsulating the atomic counter to sink pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, obj_groups="all", *args, **kwargs):

        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to sink pick and place task:
        The sink to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )

        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to sink pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta["lang"] = f"pick the {obj_lang} from the counter"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to sink pick and place task.
        Puts the target object in the front area of the counter. Puts a distractor object on the counter
        and the sink.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                split="A",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left",
                    ),
                    size=(0.30, 0.40),
                    pos=("ref", -1.01),
                    offset=(0.0, -0.01),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups=self.obj_groups,
                split="B",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.40),
                    pos=("ref", 0.1),
                    offset=(0.3, -0.001),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_counter_2",
                obj_groups=self.obj_groups,
                split="B",
                placement=dict(
                    fixture=self.sink,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.40),
                    pos=(0.1, -0.1),
                    offset=(0.0, 0.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_counter_3",
                obj_groups=self.obj_groups,
                split="B",
                graspable=False,
                placement=dict(
                    fixture=self.sink,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=(0.1, 0.3),
                    offset=(0.0, 0.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_counter_4",
                obj_groups=self.obj_groups,
                split="B",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=(0.3, 0.1),
                    offset=(0.0, 0.30),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the counter to sink pick and place task is successful.
        Checks if the object is inside the sink and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj = self.objects["obj"]
        start_pos = self.sim.data.body_xpos[
            self.obj_body_id[self.objects["distr_counter"].name]
        ]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])
        obj_z = obj_pos[2]
        obj_lifted = obj_z >= self.counter.height + 0.15

        return obj_lifted


class PnPCounterTop(PnP):
    """
    Class encapsulating the atomic counter to sink pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, obj_groups="all", *args, **kwargs):

        self.actions_meta = defaultdict(list)

        super().__init__(obj_groups=obj_groups, *args, **kwargs)

        self.obj_move = False

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to sink pick and place task:
        The sink to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("wall", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )

        self.init_robot_base_pos = self.counter
        self.obj_up_once = False
        self.obj_move = False

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to sink pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta["lang"] = f"pick the {obj_lang} from the counter"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to sink pick and place task.
        Puts the target object in the front area of the counter. Puts a distractor object on the counter
        and the sink.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                split="A",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.counter,
                        loc="left",
                    ),
                    size=(0.60, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.00, 0.00),
                ),
            )
        )
        exclude_cats = [self.obj_category] if hasattr(self, "obj_category") else []
        # distractors
        if self.use_distractors:
            cfgs.append(
                dict(
                    name="distr_counter_left_1",
                    obj_groups=self.obj_groups,
                    split="B",
                    exclude_obj_groups=exclude_cats,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.counter,
                            loc="left",
                        ),
                        size=(0.60, 0.30),
                        pos=("ref", -1.0),
                        offset=(0.00, 0.0),
                    ),
                )
            )
            cfgs.append(
                dict(
                    name="distr_counter_right_1",
                    obj_groups=self.obj_groups,
                    exclude_obj_groups=exclude_cats,
                    split="B",
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.counter,
                            loc="right",
                        ),
                        size=(0.60, 0.30),
                        pos=("ref", -1.0),
                        offset=(0.00, 0.0),
                    ),
                )
            )
            cfgs.append(
                dict(
                    name="distr_counter_left_2",
                    obj_groups=self.obj_groups,
                    split="B",
                    exclude_obj_groups=exclude_cats,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left",
                        ),
                        size=(0.6, 0.3),
                        pos=("ref", -0.9),
                        offset=(0.00, 0.03),
                    ),
                )
            )
            cfgs.append(
                dict(
                    name="distr_counter_right_2",
                    obj_groups=self.obj_groups,
                    exclude_obj_groups=exclude_cats,
                    split="B",
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.counter,
                            loc="right",
                        ),
                        size=(0.60, 0.30),
                        pos=("ref", -0.9),
                        offset=(0.00, 0.03),
                    ),
                )
            )
            cfgs.append(
                dict(
                    name="distr_counter_sink_1",
                    obj_groups=self.obj_groups,
                    exclude_obj_groups=exclude_cats,
                    split="B",
                    graspable=False,
                    placement=dict(
                        fixture=self.sink,
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(0.30, 0.30),
                        pos=(0.0, 0.0),
                        offset=(0.0, 0.0),
                    ),
                )
            )

        return cfgs

    def _check_success(self):
        """
        Check if the counter to sink pick and place task is successful.
        Checks if the object is inside the sink and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        robot_final_pos = np.array(
            self.sim.data.body_xpos[self.sim.model.body_name2id("mobilebase0_base")]
        )
        # print(f"self.robots[0].base_pos {self.robots[0].base_pos}")

        obj = self.objects["obj"]

        object_x = self.object_placements["obj"][0][1]
        object_ort = self.object_placements["obj"][1]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])
        obj_z = obj_pos[2]
        if not hasattr(self, "obj_initial_height"):
            self.obj_initial_height = obj_z
        obj_ort = np.array(self.sim.data.body_xquat[self.obj_body_id[obj.name]])
        hand_pos = np.array(
            self.sim.data.body_xpos[
                self.sim.model.body_name2id(self.robots[0].gripper["right"].root_body)
            ]
        )
        robot_pos = np.array(
            self.sim.data.body_xpos[self.sim.model.body_name2id("mobilebase0_base")]
        )
        robot_quat = np.array(
            self.sim.data.body_xquat[self.sim.model.body_name2id("mobilebase0_base")]
        )
        obj_moveing_threshold = 0.1
        # during teleop, it was 0.25 before
        # relax this value so that we might more success traj
        if self.robot_rot[2] > 3.15 or self.robot_rot[2] < -3.15:
            obj_x = obj_pos[0]
            obj_y = obj_pos[1]
            obj_z = obj_pos[2]
            object_x = self.object_placements["obj"][0][0]
            object_y = self.object_placements["obj"][0][1]
            euclidean_distance = np.linalg.norm([object_x - obj_x, object_y - obj_y])
            obj_move = euclidean_distance >= obj_moveing_threshold
        else:
            obj_x = obj_pos[0]
            obj_y = obj_pos[1]
            obj_z = obj_pos[2]
            object_x = self.object_placements["obj"][0][0]
            object_y = self.object_placements["obj"][0][1]
            euclidean_distance = np.linalg.norm([object_x - obj_x, object_y - obj_y])
            obj_move = euclidean_distance >= obj_moveing_threshold

        print(f"Object has moved: {euclidean_distance}m")

        obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
        drop_condition = self.check_contact(
            self.robots[0].gripper["right"], self.objects["obj"]
        )
        if obj_pos[2] >= self.obj_initial_height + 0.05:
            obj_up = True
        else:
            obj_up = False

        # Track if the object has ever been up at least once
        if not hasattr(self, "obj_up_once"):
            self.obj_up_once = False

        if obj_up and drop_condition:
            print("Object has been lifted at least once")
            self.obj_up_once = True

        self.actions_meta["obj_pos"].append(self.object_placements["obj"][0])
        self.actions_meta["obj_ort"].append(self.object_placements["obj"][1])
        self.actions_meta["eef_pose"].append(hand_pos)
        self.actions_meta["obj_eef_contact"].append(drop_condition)
        self.actions_meta["obj_counter_contact"].append(obj_on_counter)
        self.actions_meta["mobile_base_pos"].append(robot_pos)
        self.actions_meta["mobile_base_quat"].append(robot_quat)
        success = (
            self.obj_up_once and obj_move and obj_on_counter and not drop_condition
        )
        self.actions_meta["success"].append(success)

        if not self.obj_up_once:
            return False

        if self.obj_up_once and obj_move:
            print("Condition is satisfied for L2 distance")

        if not obj_move:
            return False

        self.obj_move = True  # object has been moved 0.25 meter

        if not hasattr(self, "obj_on_counter_frames"):
            self.obj_on_counter_frames = 0

        if self.mode == 1:
            if obj_on_counter:
                self.obj_on_counter_frames += 1
            else:
                self.obj_on_counter_frames = 0
            print("obj_on_counter times:", self.mode, self.obj_on_counter_frames)
            # Require obj_on_counter to be True for 1000 consecutive frames
            if self.obj_on_counter_frames < 100:
                return False
        else:
            if obj_on_counter:
                self.obj_on_counter_frames += 1
            else:
                self.obj_on_counter_frames = 0
            print("obj_on_counter times:", self.mode, self.obj_on_counter_frames)
            # Require obj_on_counter to be True for 1000 consecutive frames
            if self.obj_on_counter_frames < 15:
                return False

        if drop_condition:
            print("Right gripper contact with object")
            return False

        robot_final_pos = np.array(
            self.sim.data.body_xpos[self.sim.model.body_name2id("mobilebase0_base")]
        )
        robot_final_mat = np.array(
            self.sim.data.body_xmat[self.sim.model.body_name2id("mobilebase0_base")]
        )

        robot_final_mat = robot_final_mat.reshape(3, 3)
        robot_initial_mat = self.robots[0].base_ori
        rfinal_pos = np.array([robot_final_pos[0], robot_final_pos[1]])
        rinitial_pos = np.array(
            [self.robots[0].base_pos[0], self.robots[0].base_pos[1]]
        )
        check_robot_motion = self.check_robot_movement(
            rinitial_pos, rfinal_pos, robot_initial_mat, robot_final_mat
        )

        if not check_robot_motion:
            return False

        return True

    def actions_meta(self):
        return self.actions_meta

    @staticmethod
    def check_robot_movement(initial_pos, final_pos, initial_mat, final_mat):
        initial_euler = R.from_matrix(initial_mat).as_euler("xyz", degrees=False)
        final_euler = R.from_matrix(final_mat).as_euler("xyz", degrees=False)
        norm_pos = np.linalg.norm(final_pos - initial_pos)
        # check diff in euler only in Yaw after converting to RPY
        diff_euler = final_euler[2] - initial_euler[2]
        diff_euler = (diff_euler + np.pi) % (2 * np.pi) - np.pi
        if norm_pos < 0.01 and diff_euler < 0.087:
            return True

        print(f"Robot has moved with {norm_pos}m and {diff_euler}rad")
        return False


class PnPSinkToCounter(PnP):
    """
    Class encapsulating the atomic sink to counter pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, obj_groups="food", *args, **kwargs):

        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the sink to counter pick and place task:
        The sink to pick object from and the counter to place it on
        """
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        """
        Get the episode metadata for the sink to counter pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the sink and place it on the {cont_lang} located on the counter"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the sink to counter pick and place task.
        Puts the target object in the sink. Puts a distractor object on the counter
        and places a container on the counter for the target object to be placed on.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.25, 0.25),
                    pos=(0.0, 1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="container",
                obj_groups="container",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.35, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.30),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the sink to counter pick and place task is successful.
        Checks if the object is in the container, the container on the counter, and the gripper far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_in_recep = OU.check_obj_in_receptacle(self, "obj", "container")
        recep_on_counter = self.check_contact(self.objects["container"], self.counter)
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_in_recep and recep_on_counter and gripper_obj_far


class PnPCounterToMicrowave(PnP):
    # exclude layout 8 because the microwave is far from counters
    EXCLUDE_LAYOUTS = [8]
    """
    Class encapsulating the atomic counter to microwave pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to microwave pick and place task:
        The microwave to place object on, the counter to initialize it/the container on, and a distractor counter
        """
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.distr_counter = self.register_fixture_ref(
            "distr_counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_pos = self.microwave

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.microwave.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to microwave pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the counter and place it in the microwave"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to microwave pick and place task.
        Puts the target object in a container on the counter. Puts a distractor object on the distractor
        counter and places another container in the microwave.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                microwavable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )
        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate"),
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.distr_counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the counter to microwave pick and place task is successful.
        Checks if the object is inside the microwave and on the container and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj = self.objects["obj"]
        container = self.objects["container"]

        obj_container_contact = self.check_contact(obj, container)
        container_micro_contact = self.check_contact(container, self.microwave)
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_container_contact and container_micro_contact and gripper_obj_far


class PnPMicrowaveToCounter(PnP):
    # exclude layout 8 because the microwave is far from counters
    EXCLUDE_LAYOUTS = [8]
    """
    Class encapsulating the atomic microwave to counter pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, obj_groups="food", *args, **kwargs):

        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the microwave to counter pick and place task:
        The microwave to pick object from, the counter to place it on, and a distractor counter
        """
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.distr_counter = self.register_fixture_ref(
            "distr_counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_pos = self.microwave

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.microwave.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def get_ep_meta(self):
        """
        Get the episode metadata for the microwave to counter pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the microwave and place it on {cont_lang} located on the counter"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the microwave to counter pick and place task.
        Puts the target object in a container in the microwave. Puts a distractor object on the distractor
        counter and places another container on the counter."""
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                microwavable=True,
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="container",
                ),
            )
        )
        cfgs.append(
            dict(
                name="container",
                obj_groups=("container"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.distr_counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the microwave to counter pick and place task is successful.
        Checks if the object is inside the container and the gripper far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_container_contact = OU.check_obj_in_receptacle(self, "obj", "container")
        gripper_obj_far = OU.gripper_obj_far(self)
        return obj_container_contact and gripper_obj_far


class PnPCounterToStove(PnP):
    """
    Class encapsulating the atomic counter to stove pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to stove pick and place task:
        The stove to place object on and the counter to initialize it/container on
        """
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )
        self.init_robot_base_pos = self.stove

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to stove pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the plate and place it in the {cont_lang}"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to stove pick and place task.
        Puts the target object in a container on the counter and places pan on the stove.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="container",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                cookable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the counter to stove pick and place task is successful.
        Checks if the object is on the pan and the gripper far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_in_container = OU.check_obj_in_receptacle(self, "obj", "container", th=0.07)
        gripper_obj_far = OU.gripper_obj_far(self)

        return obj_in_container and gripper_obj_far


class PnPStoveToCounter(PnP):
    """
    Class encapsulating the atomic stove to counter pick and place task
    """

    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the stove to counter pick and place task:
        The counter to place object/container on and the stove to initialize it/the pan on
        """
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )
        self.init_robot_base_pos = self.stove

    def get_ep_meta(self):
        """
        Get the episode metadata for the stove to counter pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        obj_cont_lang = self.get_obj_lang(obj_name="obj_container")
        cont_lang, preposition = self.get_obj_lang(
            obj_name="container", get_preposition=True
        )
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the {obj_cont_lang} and place it {preposition} the {cont_lang}"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the stove to counter pick and place task.
        Puts the target object in a pan on the stove and places a container on the counter.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                cookable=True,
                max_size=(0.15, 0.15, None),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                    try_to_place_in="pan",
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate", "bowl"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the stove to counter pick and place task is successful.
        Checks if the object is inside the container on the counter and the gripper far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_in_container = OU.check_obj_in_receptacle(self, "obj", "container", th=0.07)
        gripper_obj_far = OU.gripper_obj_far(self)

        return obj_in_container and gripper_obj_far
