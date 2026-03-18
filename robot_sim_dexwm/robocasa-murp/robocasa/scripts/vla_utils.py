import os
import sys
# Modify the path here for your robot-skills folder
# VLA_PATH = "/home/jimmytyyang/research/robot-skills"
# sys.path.append(VLA_PATH)

# try:
#     # force the model to point to the new robot-skills
#     sys.path.remove("__editable__.robot_skills-0.1.0.finder.__path_hook__")
# except:
#     pass
# Modify the path here for your transformers cache file
import json
import inspect
import random
import pickle
import einops
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from omegaconf import OmegaConf
from PIL import Image
from src.model.vla.pizero import PiZeroInference, MultiTaskPiZero
import hydra
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
from src.model.vla.processing import VLAProcessor
from transformers import AutoTokenizer
import json
from scripts.utils import hydra_solver_num_images_given_load_depth

# Set the random seed
random.seed(42)

# Only register once
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "num_images_given_load_depth", hydra_solver_num_images_given_load_depth
)

# Data for clip the action and sensor
CLIP_MIN = -1.0
CLIP_MAX = 1.0
# (height, width) (numpy array format) with the different stage of the transformation size
# first to (300, 480), and then to (224, 224). This is to resemble how we transform the data
# from the initial hdf5 to tfds to the image size used during data loader.
TARGET_SIZE = [(300, 480), (224, 224)]
# The mapping of the plot
NAME_MAPPING = {
    0: "ee pos x",
    1: "ee pos y",
    2: "ee pos z",
    3: "ee rot mat (0 0)",
    4: "ee rot mat (0 1)",
    5: "ee rot mat (0 2)",
    6: "ee rot mat (1 0)",
    7: "ee rot mat (1 1)",
    8: "ee rot mat (1 2)",
    9: "finger 0",
    10: "finger 1",
    11: "finger 2",
    12: "finger 3",
    13: "finger 4",
    14: "finger 5",
    15: "finger 6",
    16: "finger 7",
    17: "finger 8",
    18: "finger 9",
    19: "finger 10",
    20: "finger 11",
    21: "finger 12",
    22: "finger 13",
    23: "finger 14",
    24: "finger 15",
}
# Pick instructions
TEXT_BANK = (
    "pick up {object}",
    "hey robot, could you please pick up the {object}",
    "I want to get {object}",
    "could you fetch {object}",
    "I want to find {object}",
    "return {object} to me",
    "find the {object} to me",
    "pass {object} to me",
    "please hand over {object} to me",
    "hey murp robot, pick up {object} for me",
    "where could I find {object}",
    "is there a {object}? if yes, could you please fetch it for me",
    "get the {object}",
    "pick the {object} up",
    "grasp a {object}",
    "grab the {object} up",
    "grab the {object}",
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
    "please do pick and place the {object}",
    "pick up {object} and place it randomly",
    "hey murp, could you please pick up {object} for me",
    "get {object} for me",
    "hand over {object} to me",
    "perform the action of picking {object} up and place somewhere on the table",
    "you are a helpful robot that follows my command: could you please fetch {object} for me",
)
# The number of repetition for the flow-matching policy action output
REPEAT_BATCH = 50


def move_target_object_given_global_xyz(
    env, xyz=np.array([0, 0, 0]), target_obj_name="obj"
):
    """Useful function to move the object given x,y,z location"""
    for obj_pos, obj_quat, obj in env.object_placements.values():
        if obj.name == target_obj_name:
            current_pos = np.array(
                env.sim.data.body_xpos[env.obj_body_id[target_obj_name]]
            ).copy()
            env.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([current_pos + xyz, np.array(obj_quat)]),
            )


def process_input(
    vla_config, data, cam="robot0_robotview_2", image_process_mode="resize"
):
    # Determine the numebr of the input images
    if vla_config.get("image_cond_steps", None) != None:
        image_cond_steps = vla_config.image_cond_steps
    else:
        image_cond_steps = vla_config.cond_steps

    # Unscale version of the data
    past_vs = []
    # Get the past
    for j in range(1, image_cond_steps):
        if len(data) < j + 1:
            break
        past_v = data[-j - 1]
        single_step_img = np.expand_dims(past_v[cam], axis=0)
        past_vs.append(single_step_img)

    if len(past_vs) > 0:
        past_vs = np.concatenate(past_vs)
    else:
        # Add dummy
        if image_process_mode == "resize":
            past_vs = np.zeros((0, 600, 960, 3))
        else:
            past_vs = np.zeros((0, 600, 960, 3))

    # Get the current
    v = data[-1]
    single_step_img = np.expand_dims(v[cam], axis=0)  # size = (1, 224, 224, 3)

    # Concate
    single_step_img = np.concatenate((past_vs, single_step_img), axis=0)

    # Repeat the first element if the length is smaller
    if single_step_img.shape[0] < image_cond_steps:
        first_obs = single_step_img[[0]]
        repeat_obs = np.concatenate(
            [first_obs] * (image_cond_steps - single_step_img.shape[0])
        )
        single_step_img = np.concatenate((repeat_obs, single_step_img), axis=0)
    return single_step_img


def process_lang(text):
    """A helpful function to make the input text more free-formed"""
    text = text.split("and")[0].strip()
    text = text.split(" ")
    from_i = 0
    the_i = 0
    for index, v in enumerate(text):
        if v == "the":
            the_i = index
        elif v == "from":
            from_i = index
            break
    object_name = " ".join(text[the_i + 1 : from_i])
    # Add an action prompt
    return f'action {random.choice(TEXT_BANK).replace("{object}", object_name)}'


def add_img_to_obs(env, obs, camera_height, camera_width):
    """Add rgb images into observations"""
    # This is torso cam
    torso_img = obs["robot0_robotview_image"][::-1]  # flip the image
    # image = Image.fromarray(torso_img.astype(np.uint8))
    # image.save(f"torso_img.png")

    # This is head
    head_img = obs["robot0_robotview_2_image"][::-1]
    # image = Image.fromarray(head_img.astype(np.uint8))
    # image.save(f"head_img.png")

    wrist_img = obs["gripper0_right_right_eye_in_hand_image"][::-1]
    # image = Image.fromarray(wrist_img.astype(np.uint8))
    # image.save(f"wrist_img.png")

    # Rename
    obs["robot0_robotview"] = torso_img
    obs["robot0_robotview_2"] = head_img
    obs["gripper0_right_right_eye_in_hand"] = wrist_img

    return obs


def crop_image(image):
    """Crop the image"""
    # Get the dimensions of the image
    height, width, _ = image.shape
    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2
    # Calculate the coordinates for the crop
    start_x = max(center_x - 112, 0)
    start_y = max(center_y - 112, 0)
    end_x = start_x + 224
    end_y = start_y + 224
    # Ensure the crop is within the image bounds
    end_x = min(end_x, width)
    end_y = min(end_y, height)
    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image


def get_vla_obs(
    env,
    camera_names,
    camera_height,
    camera_width,
    vla_config,
    obs_list,
    lang,
    time_index=0,
    pad_zero_dim=0,
):
    """Get the observation for the input of VLA"""
    if not vla_config.data.train.get("load_wrist", False):
        torso_img = process_input(vla_config, obs_list, cam="robot0_robotview")
        rgb_img = torch.tensor(np.tile(torso_img, (REPEAT_BATCH, 1, 1, 1, 1)))
        # image = Image.fromarray(torso_img[0].astype(np.uint8))
        # image.save(f"vla_image_{time_index}.png")
    else:
        torso_img = process_input(vla_config, obs_list, cam="robot0_robotview")
        wrist_img = process_input(
            vla_config, obs_list, cam="gripper0_right_right_eye_in_hand"
        )
        rgb_img = np.concatenate((torso_img, wrist_img), axis=0)
        rgb_img = torch.tensor(np.tile(rgb_img, (REPEAT_BATCH, 1, 1, 1, 1)))

    # Get the proprio
    past_vs = []
    for j in range(1, vla_config.cond_steps):
        if len(obs_list) < j + 1:
            break
        past_v = obs_list[-j - 1]
        proprios = np.expand_dims(
            np.concatenate(
                (
                    past_v["robot0_right_eef_T_right_base_pos"],
                    T.quat2mat(past_v["robot0_right_eef_T_right_base_quat_xyzw"])[
                        :2
                    ].flatten(),
                    past_v["robot0_right_gripper_qpos"],
                ),
                axis=0,
            ),
            axis=0,
        )
        past_vs.append(proprios)

    if len(past_vs) > 0:
        past_vs = np.concatenate(past_vs)
    else:
        # Add dummy
        past_vs = np.zeros((0, 25 + pad_zero_dim))

    # Get the current
    proprios = np.expand_dims(
        np.concatenate(
            (
                obs_list[-1]["robot0_right_eef_T_right_base_pos"],
                T.quat2mat(obs_list[-1]["robot0_right_eef_T_right_base_quat_xyzw"])[
                    :2
                ].flatten(),
                obs_list[-1]["robot0_right_gripper_qpos"],
            ),
            axis=0,
        ),
        axis=0,
    )

    # Concate
    proprios = np.concatenate((past_vs, proprios), axis=0)

    # Repeat the first element if the length is smaller
    if proprios.shape[0] < vla_config.cond_steps:
        first_obs = proprios[[0]]
        repeat_obs = np.concatenate(
            [first_obs] * (vla_config.cond_steps - proprios.shape[0])
        )
        proprios = np.concatenate((repeat_obs, proprios), axis=0)

    proprios = torch.tensor(np.tile(proprios, (REPEAT_BATCH, 1, 1)))

    obs = {
        "img": rgb_img,
        "proprio": proprios,
        "text": lang,
    }
    return obs


def debug_plot(
    vla_action_list,
    gt_action_list,
    obs_list,
    infer_timestep_list,
    save_name,
    mode="vla",
):
    """Simple function to plot the visual"""

    # Extract the sensor information
    pred = np.stack(vla_action_list)
    gt = np.stack(gt_action_list)
    ee_pos = np.stack([v["robot0_right_eef_T_right_base_pos"] for v in obs_list])
    ee_rot_mat_6d = np.stack(
        [
            T.quat2mat(v["robot0_right_eef_T_right_base_quat_xyzw"])[:2].flatten()
            for v in obs_list
        ]
    )
    hand_finger = np.stack([v["robot0_right_gripper_qpos"] for v in obs_list])
    proprio = np.concatenate((ee_pos, ee_rot_mat_6d, hand_finger), axis=-1)

    num_sub_plot = len(NAME_MAPPING)

    fig, axes = plt.subplots(
        num_sub_plot, figsize=(6, 40), height_ratios=[1] * num_sub_plot
    )
    # Display each image in a subplot
    for i in range(num_sub_plot):
        axes[i].plot(
            np.arange(pred.shape[0]).tolist(),
            pred[:, i].tolist(),
            color="blue",
            label="Action prediction",
        )
        axes[i].plot(
            np.arange(gt.shape[0]).tolist(),
            gt[:, i].tolist(),
            color="red",
            label="Action GT",
        )
        axes[i].plot(
            np.arange(proprio[:, i].shape[0]).tolist(),
            proprio[:, i].tolist(),
            color="green",
            label="Proprio observation",
        )
        axes[i].plot(
            infer_timestep_list,
            pred[infer_timestep_list, i].tolist(),
            color="black",
            marker="o",
            label="Inference time index",
            linestyle="None",
        )
        axes[i].set_title(f"{NAME_MAPPING[i]}")

    # Show legend
    for i in range(num_sub_plot):
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(save_name)


def rot_mat_six_dim_to_axisangle(rot_mat_six_dim):
    # Reconstruct the full 3x3 rotation matrix
    # Assume six_dim = [r11, r12, r13, r21, r22, r23]
    r11, r12, r13, r21, r22, r23 = rot_mat_six_dim
    r31 = r12 * r23 - r13 * r22
    r32 = r13 * r21 - r11 * r23
    r33 = r11 * r22 - r12 * r21
    rotation_matrix = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    # Use scipy to convert the rotation matrix to axis-angle
    rotation = R.from_matrix(rotation_matrix)
    axis_angle = rotation.as_rotvec()
    return axis_angle


def plot_images(imgs, text):
    # imgs: [batch, channel, H, W]
    img = einops.rearrange(imgs[0], "C H W -> H W C") * 255
    img = img.to(torch.uint8)
    img = Image.fromarray(img.numpy())
    img.save(f"img_time_idx_{text}.png")


def load_vla_skill_state_dict(data, model):
    """load to cpu first, then move to gpu"""
    data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
    model.load_state_dict(data["model"], strict=True)
    return model


def load_vla_skill(vla_skill_path, main_config, cuda_device="cuda:0"):
    """Load ckpt skill"""

    # Define the config path
    # config = OmegaConf.load(
    #     VLA_PATH + "/config/train/" + yaml_name,
    # )
    data = torch.load(vla_skill_path, weights_only=False, map_location="cpu")

    training_stats = data["training_stats"]
    config = data["training_config"]

    # Determine the model type
    dtype = torch.bfloat16 if config.get("use_bf16", True) else torch.float32

    # Initial the model and check if we want to distribute the model
    # to gpus
    model_type = "PiZero"

    if (
        "distribute_model_in_gpus" in main_config
        and main_config["distribute_model_in_gpus"]
    ):
        gpu_ids = []
        for i in range(torch.cuda.device_count()):
            gpu_ids.append(torch.device(f"cuda:{i}"))
        model = MultiTaskPiZero(config, use_ddp=False, gpu_ids=gpu_ids)
    else:
        if "_model_" in config:
            # Use hydra to import the model
            model_class = hydra.utils.get_class(config._model_)
            model = model_class(
                config, use_ddp=False, gpu_ids=[int(cuda_device.split(":")[-1])]
            )
            model_type = str(config._model_)
        else:
            model = MultiTaskPiZero(config, use_ddp=False)

    # Load the actual checkpoint
    model = load_vla_skill_state_dict(
        data,
        model,
    )

    # Freeze the weights
    model.freeze_all_weights()

    if "distribute_model_in_gpus" in main_config:
        if main_config["distribute_model_in_gpus"]:
            pass
        else:
            model.to(cuda_device)
    else:
        model.to(cuda_device)

    # To dtype
    model.to(dtype)

    # Compile the model
    # model = torch.compile(
    #     model,
    #     mode="default",
    # )
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_path, padding_side="right"
    )
    if "act" not in model_type:
        # Processor
        # Check the function signature
        vla_processor_var = inspect.signature(VLAProcessor).parameters.keys()
        if "use_meta_clip" in vla_processor_var:
            processor = VLAProcessor(
                tokenizer,
                config.vision.config.num_image_tokens,
                config.max_seq_len,
                config.tokenizer_padding,
                dinov2_hf=config.get("dinov2_hf", False),
                use_meta_clip=config.get("use_meta_clip", False),
            )
        elif "processor_name" in vla_processor_var:
            processor = VLAProcessor(
                tokenizer,
                config.vision.config.num_image_tokens,
                config.max_seq_len,
                config.tokenizer_padding,
                dinov2_hf=config.get("dinov2_hf", False),
                processor_name=config.get("processor_name", False),
            )
        else:
            processor = VLAProcessor(
                tokenizer,
                config.vision.config.num_image_tokens,
                config.max_seq_len,
                config.tokenizer_padding,
                dinov2_hf=config.get("dinov2_hf", False),
            )
    else:
        processor = None

    return model, processor, config, training_stats


def process_imgs(imgs, is_rgb=True, resize_rgb=True):
    if is_rgb:
        # Resize the image here
        # TODO: the image is not being resized
        target_size_height, target_size_width = TARGET_SIZE[-1]
        rgbs_process = torch.zeros(
            (imgs.shape[0], imgs.shape[1], target_size_height, target_size_width, 3)
        )
        for i, rgbs in enumerate(imgs):
            for j, rgb in enumerate(rgbs):
                if resize_rgb:
                    img = Image.fromarray(rgb.cpu().detach().numpy().astype(np.uint8))
                    for target_size in TARGET_SIZE:
                        img = img.resize((target_size[1], target_size[0]))
                else:
                    img = crop_image(rgb.cpu().detach().numpy().astype(np.uint8))
                rgb = torch.as_tensor(np.array(img))
                rgbs_process[i, j] = rgb

        return rgbs_process
    else:
        assert False, "depth image code computation is wrong. TODO here!!!!!"
        imgs = einops.repeat(imgs, "B H W 1 -> B H W 3")
        depthes_process = torch.zeros((imgs.shape[0], target_size, target_size, 3))
        for i, depth in enumerate(imgs):
            img = Image.fromarray(depth.cpu().detach().numpy())
            img = img.resize((480, 300))
            img = img.resize((target_size, target_size))
            depth = torch.as_tensor(np.array(img))
            depthes_process[i] = depth
        return depthes_process


def normalize(data, p01, p99):
    """normalize the sensor: joint and action"""
    normalize_value = 2 * (data - p01) / (p99 - p01 + 1e-8) - 1
    return torch.clip(normalize_value, CLIP_MIN, CLIP_MAX)


def dennormalize(data, p01, p99):
    """denormalize the sensor: joint and action"""
    return (data - CLIP_MIN) / (CLIP_MAX - CLIP_MIN) * (p99 - p01) + p01


def action_chunk_rot_6d_to_mat(rot_6d):
    """Make the action chunk rot 6d to mat"""
    action_chunk_size = rot_6d.shape[1]
    r11 = rot_6d[:, :, 0]
    r12 = rot_6d[:, :, 1]
    r13 = rot_6d[:, :, 2]
    r21 = rot_6d[:, :, 3]
    r22 = rot_6d[:, :, 4]
    r23 = rot_6d[:, :, 5]
    r31 = r12 * r23 - r13 * r22
    r32 = r13 * r21 - r11 * r23
    r33 = r11 * r22 - r12 * r21
    stacked = torch.stack([r11, r12, r13, r21, r22, r23, r31, r32, r33], axis=2)
    reshaped = torch.reshape(stacked, [-1, action_chunk_size, 3, 3])
    return reshaped


def infer_action(
    vla_skill,
    vla_processor,
    vla_config,
    obs,
    training_stats,
    training_stats_key=None,
    cuda_device="cuda:0",
    save_path="data.npz",
):
    # Determine the model type
    if "_model_" in vla_config:
        # Use hydra to import the model
        model_type = str(vla_config._model_)
    else:
        model_type = "PiZero"

    # Determine where to do normalization of the observation and the actions
    normalization_mode = "within_whole_traj"
    if vla_config.data.train.get("compute_stats_after_traj_transform", False):
        normalization_mode = "within_prediction_chunk"
        if not vla_config.data.train.get("use_rot_mat_sub_post_chunk", True):
            normalization_mode = "within_prediction_chunk_inv_multi"

    assert normalization_mode in [
        "within_whole_traj",
        "within_prediction_chunk",
        "within_prediction_chunk_inv_multi",
    ], f"Use the wrong normalization mode: {normalization_mode}."

    # RGB
    images = []
    _image = process_imgs(
        obs["img"], is_rgb=True, resize_rgb=vla_config.data.train.get("resize_rgb", True)
    )
    images = _image.to(torch.uint8)  # torch.Size([1, cond_step, 224, 224, 3])
    batch_size = images.shape[0]
    num_img_per_step = images.shape[1]

    # Proprios
    proprios = obs["proprio"].clone()

    # Normalize
    # Use the first key for the dataset
    if training_stats_key is None:
        training_stats_key = list(training_stats.keys())[0]

    if normalization_mode in [
        "within_prediction_chunk",
        "within_prediction_chunk_inv_multi",
    ]:
        proprio_normalization_mask = training_stats[training_stats_key]["proprio"][
            "mask"
        ]
        proprios[:, :, proprio_normalization_mask] = normalize(
            proprios[:, :, proprio_normalization_mask],
            training_stats[training_stats_key]["proprio"]["p01"][
                :, proprio_normalization_mask
            ],
            training_stats[training_stats_key]["proprio"]["p99"][
                :, proprio_normalization_mask
            ],
        )
    else:
        proprio_normalization_mask = training_stats[training_stats_key]["proprio"]
        if "mask" in proprio_normalization_mask:
            proprio_normalization_mask = proprio_normalization_mask["mask"]
        else:
            # forget to save in robot-skills, use default one that normalizes everything
            proprio_normalization_mask = [True] * 3 + [True] * 6 + [True] * 16
        proprios[:, :, proprio_normalization_mask] = normalize(
            proprios[:, :, proprio_normalization_mask],
            training_stats[training_stats_key]["proprio"]["p01"][
                proprio_normalization_mask
            ],
            training_stats[training_stats_key]["proprio"]["p99"][
                proprio_normalization_mask
            ],
        )

    # Get type
    dtype = torch.bfloat16 if vla_config.get("use_bf16", True) else torch.float32

    images = einops.rearrange(
        images, "B T H W C -> (B T) C H W"
    )  # remove num_img_per_step dimension

    # plot_images(images / 255, f"test")
    if vla_processor is None:
        # This is for ACT models
        model_inputs = {}
        model_inputs["pixel_values"] = images
        model_inputs["pixel_values"] = einops.rearrange(
            model_inputs["pixel_values"],
            "(B T) C H W -> B T C H W",
            B=batch_size,
            T=num_img_per_step,
        )
        # Pad the tensor
        padded_tensor = F.pad(proprios, (0, 7))
        inputs = {
            "pixel_values": model_inputs["pixel_values"].to(dtype),
            "proprios": padded_tensor.to(dtype),
        }
    elif "PiZero" in model_type:
        model_inputs = vla_processor(text=[obs["text"]] * batch_size, images=images)
        model_inputs["pixel_values"] = einops.rearrange(
            model_inputs["pixel_values"],
            "(B T) C H W -> B T C H W",
            B=batch_size,
            T=num_img_per_step,
        )

        (
            causal_mask,
            vlm_position_ids,
            proprio_position_ids,
            action_position_ids,
        ) = vla_skill.build_causal_mask_and_position_ids(
            model_inputs["attention_mask"],
            dtype,
        )

        inputs = {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"].to(dtype),
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": proprios.to(dtype),
        }

        # For evaluation mode
        split_mask = True
        if split_mask:
            (
                image_text_proprio_mask,
                action_mask,
            ) = vla_skill.split_full_mask_into_submasks(causal_mask)
            inputs["image_text_proprio_mask"] = image_text_proprio_mask
            inputs["action_mask"] = action_mask
        else:
            inputs["causal_mask"] = causal_mask
    else:
        # LBM case
        images = einops.rearrange(
            images,
            "(B T) C H W -> B T C H W",
            B=batch_size,
            T=num_img_per_step,
        )

        # rgb = einops.rearrange(
        #     images[0],
        #     "C H W -> H W C",
        # )

        # Concate the images since the HF model cannot take more than one image
        images = torch.cat([images[:, i] for i in range(images.size(1))], dim=-1)
        model_inputs = vla_processor(text=[obs["text"]] * batch_size, images=images)
        inputs = {
            "input_ids": model_inputs["input_ids"],  # torch.Size([2, 256, 4608])
            "attention_mask": model_inputs[
                "attention_mask"
            ],  # torch.Size([2, 256, 4608])
            "pixel_values": model_inputs["pixel_values"],  # torch.Size([2, 256, 4608])
            "proprios": proprios.to(dtype),
        }

    # Move the input to the main device
    inputs = {k: v.to(cuda_device) if type(v) != list else v for k, v in inputs.items()}

    # Add dataset name
    # inputs["dataset_name"] = [["action_delta_arm_joints","proprio_arm_related_8"]] * proprios.shape[0]

    # Finally, get the action
    # [batch_size, horizen, dim] = torch.Size([1, 1, 23])
    with torch.inference_mode():
        preds = vla_skill.infer_action(**inputs)

    # The action was dennormalized
    preds = preds.cpu().detach().to(torch.float32)
    preds_processed = preds.clone()
    action_normalization_mask = training_stats[training_stats_key]["action"]["mask"]
    if normalization_mode in [
        "within_prediction_chunk",
        "within_prediction_chunk_inv_multi",
    ]:
        preds_processed[:, :, action_normalization_mask] = dennormalize(
            preds_processed[:, :, action_normalization_mask],
            training_stats[training_stats_key]["action"]["p01"][
                :, :, action_normalization_mask
            ],
            training_stats[training_stats_key]["action"]["p99"][
                :, :, action_normalization_mask
            ],
        ).to(torch.float32)

        if normalization_mode == "within_prediction_chunk":
            # Add back the current proprio
            preds_processed += obs["proprio"]
        else:
            # Get ee pos
            ee_pos = preds_processed[:, :, 0:3] + obs["proprio"][:, :, 0:3]
            action_chunk_size = ee_pos.shape[1]
            # Get rot 6d
            ee_rot_mat_pred = action_chunk_rot_6d_to_mat(preds_processed[:, :, 3:9])
            ee_rot_mat_init = action_chunk_rot_6d_to_mat(obs["proprio"][:, :, 3:9])
            ee_rot_mat_init = ee_rot_mat_init.repeat(1, action_chunk_size, 1, 1)
            ee_rot_mat = torch.matmul(ee_rot_mat_init.float(), ee_rot_mat_pred)[
                :, :, :2
            ]
            ee_rot_6d = ee_rot_mat.view(batch_size, action_chunk_size, 6)
            # Get hand
            hand_joint = preds_processed[:, :, 9:] + obs["proprio"][:, :, 9:]
            preds_processed = torch.cat((ee_pos, ee_rot_6d, hand_joint), dim=-1)
    else:
        preds_processed[:, :, action_normalization_mask] = dennormalize(
            preds_processed[:, :, action_normalization_mask],
            training_stats[training_stats_key]["action"]["p01"][
                action_normalization_mask
            ],
            training_stats[training_stats_key]["action"]["p99"][
                action_normalization_mask
            ],
        ).to(torch.float32)

    return preds_processed
