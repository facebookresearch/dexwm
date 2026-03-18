import random
from copy import deepcopy
import json
from robocasa.models.objects.kitchen_objects import (
    OBJ_GROUPS as OBJ_GROUPS,
)
from robocasa.models.objects.kitchen_objects import (
    OBJ_CATEGORIES as OBJ_CATEGORIES,
)
import os
import robocasa

BASE_JSON_PATH = os.path.dirname(robocasa.models.objects.__file__)


def seed_shuffle(seed_val):

    TEST_SET = {}
    TRAIN_SET = {}
    random.seed(seed_val)

    all_objs = list(OBJ_CATEGORIES.keys())

    random.shuffle(all_objs)
    split_idx = int(len(all_objs) * 0.8)
    TRAIN_SET = all_objs[:split_idx]
    TEST_SET = all_objs[split_idx:]
    return TRAIN_SET, TEST_SET


def filter_categories_and_groups(obj_names, obj_categories, obj_groups):
    filtered_categories = {}
    filtered_groups = {}

    # Filter categories
    for name in obj_names:
        if name in obj_categories:
            filtered_categories[name] = deepcopy(obj_categories[name])

    # Filter groups: only keep objects in obj_names
    for group_name, group_list in obj_groups.items():
        filtered_groups[group_name] = [obj for obj in group_list if obj in obj_names]

    return filtered_categories, filtered_groups


TRAIN_SET, TEST_SET = seed_shuffle(42)
TRAIN_CATEGORIES, TRAIN_GROUPS = filter_categories_and_groups(
    TRAIN_SET, OBJ_CATEGORIES, OBJ_GROUPS
)
TEST_CATEGORIES, TEST_GROUPS = filter_categories_and_groups(
    TEST_SET, OBJ_CATEGORIES, OBJ_GROUPS
)
with open(os.path.join(BASE_JSON_PATH, "train_categories.json"), "w") as f:
    json.dump(TRAIN_CATEGORIES, f, indent=4)

with open(os.path.join(BASE_JSON_PATH, "train_groups.json"), "w") as f:
    json.dump(TRAIN_GROUPS, f, indent=4)

with open(os.path.join(BASE_JSON_PATH, "test_categories.json"), "w") as f:
    json.dump(TEST_CATEGORIES, f, indent=4)

with open(os.path.join(BASE_JSON_PATH, "test_groups.json"), "w") as f:
    json.dump(TEST_GROUPS, f, indent=4)
