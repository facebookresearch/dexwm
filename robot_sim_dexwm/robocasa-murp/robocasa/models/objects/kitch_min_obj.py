import os
import robocasa

BASE_ASSET_ZOO_PATH = os.path.join(robocasa.models.assets_root, "objects")


# Constant that contains information about each object category. These will be used to generate the ObjCat classes for each category
OBJ_CATEGORIES = dict(
    liquor=dict(
        types=("drink", "alcohol"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            model_folders=["aigen_objs/alcohol"],
            scale=1.0,
        ),
        objaverse=dict(
            model_folders=["objaverse/alcohol"],
            scale=1.0,
        ),
    ),
    apple=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    avocado=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    banana=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.10,
        ),
        objaverse=dict(
            scale=1.55,
        ),
    ),
    bottled_water=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(
            scale=1.10,
            exclude=[
                "bottled_water_0",  # minor hole at top
                "bottled_water_5",  # causing error. eigenvalues of mesh inertia violate A + B >= C
            ],
        ),
    ),
    bowl=dict(
        types=("receptacle", "stackable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(
            scale=2.0,
            exclude=[
                "bowl_21",  # can see through from bottom of bowl
            ],
        ),
    ),
    bell_pepper=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.75,
        ),
    ),
    boxed_food=dict(
        types=("packaged_food"),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(
            scale=1.1,
            exclude=[
                "boxed_food_5",  # causing error. eigenvalues of mesh inertia violate A + B >= C
            ],
            # exclude=[
            #     "boxed_food_5",
            #     "boxed_food_3", "boxed_food_1", "boxed_food_6", "boxed_food_11", "boxed_food_10", "boxed_food_8", "boxed_food_9", "boxed_food_7", "boxed_food_2", # self turning due to single collision geom
            # ],
        ),
    ),
    can=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(),
        objaverse=dict(
            exclude=[
                "can_10",  # hole on bottom
                "can_5",  # causing error: faces of mesh have inconsistent orientation.
            ],
        ),
    ),
    cereal=dict(
        types=("packaged_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            # exclude=[
            #     "cereal_2", "cereal_5", "cereal_13", "cereal_3", "cereal_9", "cereal_0", "cereal_7", "cereal_4", "cereal_8", "cereal_12", "cereal_11", "cereal_1", "cereal_6", "cereal_10", # self turning due to single collision geom
            # ]
        ),
    ),
    wine_glass=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=True,
        types=("receptacle"),
    ),
    donut=dict(
        types=("sweets", "pastry"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            scale=1.15,
        ),
    ),
    lemon=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(),
    ),
    lime=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        objaverse=dict(
            scale=1.0,
        ),
        aigen=dict(
            scale=0.90,
        ),
    ),
    pot=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=2.25,
        ),
        objaverse=dict(
            model_folders=["objaverse/pan"],
            scale=1.70,
            exclude=list(
                set([f"pan_{i}" for i in range(25)])
                - set(["pan_0", "pan_12", "pan_17", "pan_22"])
            ),
        ),
    ),
    jam=dict(
        types=("packaged_food"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.05,
        ),
        objaverse=dict(
            scale=0.90,
        ),
    ),
    jug=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            scale=1.5,
        ),
    ),
    ketchup=dict(
        types=("condiment"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "ketchup_5"  # causing error: faces of mesh have inconsistent orientation.
            ]
        ),
    ),
    milk=dict(
        types=("dairy", "drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "milk_6"  # causing error: eigenvalues of mesh inertia violate A + B >= C
            ]
        ),
    ),
    spray=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(
            scale=1.75,
        ),
    ),
    soap_dispenser=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.7,
        ),
        objaverse=dict(
            exclude=[
                # "soap_dispenser_4", # can see thru body but that's fine if this is glass
            ]
        ),
    ),
    teapot=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(
            scale=1.25,
            exclude=[
                "teapot_9",  # hole on bottom
            ],
        ),
    ),
    yogurt=dict(
        types=("dairy", "packaged_food"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.95,
        ),
    ),
    lemonade=dict(
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("drink"),
    ),
    coffee_cup=dict(
        types=("drink"),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "coffee_cup_18",  # can see thru top
                "coffee_cup_5",  # can see thru from bottom side
                "coffee_cup_19",  # can see thru from bottom side
            ],
        ),
    ),
    coconut=dict(
        aigen=dict(
            scale=2.0,
        ),
        objaverse=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        types=("fruit"),
    ),
    pineapple=dict(
        aigen=dict(
            scale=2.0,
        ),
        objaverse=dict(
            exclude=[
                "pineapple_6",
            ],
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    squash=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            exclude=[
                "squash_10",  # hole at bottom
            ],
        ),
    ),
    watermelon=dict(
        aigen=dict(
            scale=2.5,
        ),
        objaverse=dict(
            scale=0.75,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("fruit"),
    ),
)


def get_cats_by_type(types, obj_registries=None):
    """
    Retrieves a list of item keys from the global `OBJ_CATEGORIES` dictionary based on the specified types.

    Args:
        types (list): A list of valid types to filter items by. Only items with a matching type will be included.
        obj_registries (list): only consider categories belonging to these object registries

    Returns:
        list: A list of keys from `OBJ_CATEGORIES` where the item's types intersect with the provided `types`.
    """
    types = set(types)

    res = []
    for key, val in OBJ_CATEGORIES.items():
        # check if category is in one of valid object registries
        if obj_registries is not None:
            if isinstance(obj_registries, str):
                obj_registries = [obj_registries]
            if any([reg in val for reg in obj_registries]) is False:
                continue

        if "types" in val:
            cat_types = val["types"]
        else:
            cat_types = list(val.values())[0].types
        if isinstance(cat_types, str):
            cat_types = [cat_types]
        cat_types = set(cat_types)
        # Access the "types" key in the dictionary using the correct syntax
        if len(cat_types.intersection(types)) > 0:
            res.append(key)

    return res


### define all object categories ###
OBJ_GROUPS = dict(
    all=list(OBJ_CATEGORIES.keys()),
)

for k in OBJ_CATEGORIES:
    OBJ_GROUPS[k] = [k]

all_types = set()
# populate all_types
for (cat, cat_meta_dict) in OBJ_CATEGORIES.items():
    # types are common to both so we only need to examine one
    cat_types = cat_meta_dict["types"]
    if isinstance(cat_types, str):
        cat_types = [cat_types]
    all_types = all_types.union(cat_types)

for t in all_types:
    OBJ_GROUPS[t] = get_cats_by_type(types=[t])

OBJ_GROUPS["food"] = get_cats_by_type(
    [
        "vegetable",
        "fruit",
        "sweets",
        "dairy",
        "meat",
        "bread_food",
        "pastry",
        "cooked_food",
    ]
)
OBJ_GROUPS["in_container"] = get_cats_by_type(
    [
        "vegetable",
        "fruit",
        "sweets",
        "dairy",
        "meat",
        "bread_food",
        "pastry",
        "cooked_food",
    ]
)

# custom groups
OBJ_GROUPS["container"] = ["bowl"]
OBJ_GROUPS["kettle"] = ["kettle_electric", "kettle_non_electric"]
OBJ_GROUPS["cookware"] = ["pan", "pot", "kettle_non_electric"]
OBJ_GROUPS["pots_and_pans"] = ["pan", "pot"]
OBJ_GROUPS["food_set1"] = [
    "apple",
    "baguette",
    "banana",
    "carrot",
    "cheese",
    "cucumber",
    "egg",
    "lemon",
    "orange",
    "potato",
]
OBJ_GROUPS["group1"] = ["apple", "carrot", "banana", "bowl", "can"]
OBJ_GROUPS["container_set2"] = ["bowl"]
