# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import numpy as np
import random
import PIL
import torch
from PIL import ImageEnhance, ImageFilter, Image
import math

def to_pil(im):
    if isinstance(im, PIL.Image.Image):
        return im
    elif isinstance(im, torch.Tensor):
        return PIL.Image.fromarray(np.asarray(im))
    elif isinstance(im, np.ndarray):
        return PIL.Image.fromarray(im)
    else:
        raise ValueError('Type not supported', type(im))

class PillowRGBAugmentation:
    def __init__(self, pillow_fn, p, factor_interval):
        self._pillow_fn = pillow_fn
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, im, mask, obs):
        im = to_pil(im)
        if random.random() <= self.p:
            im = self._pillow_fn(im).enhance(factor=random.uniform(*self.factor_interval))
        #im.save('./BRIGHT.png')
        return im, mask, obs

class PillowSharpness(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0., 50.)):
        super().__init__(pillow_fn=ImageEnhance.Sharpness,
                         p=p,
                         factor_interval=factor_interval)


class PillowContrast(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.2, 50.)):
        super().__init__(pillow_fn=ImageEnhance.Contrast,
                         p=p,
                         factor_interval=factor_interval)


class PillowBrightness(PillowRGBAugmentation):
    def __init__(self, p=0.5, factor_interval=(0.1, 6.0)):
        super().__init__(pillow_fn=ImageEnhance.Brightness,
                         p=p,
                         factor_interval=factor_interval)


class PillowColor(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.0, 20.0)):
        super().__init__(pillow_fn=ImageEnhance.Color,
                         p=p,
                         factor_interval=factor_interval)

def occlusion_aug(bbox, img_shape, min_area=0.0, max_area=0.3, max_try_times=5):
    # xmin, ymin, _, _ = bbox
    # xmax = bbox[2]
    # ymax = bbox[3]
    imght, imgwidth = img_shape
    xmin, ymin, xmax, ymax = 0,0,imgwidth,imght
    counter = 0
    while True:
        # force to break if no suitable occlusion
        if counter > max_try_times: # 5
            print('No suitable occlusion')
            return 0, 0, 0, 0
        counter += 1

        area_min = min_area # 0.0
        area_max = max_area # 0.3
        synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)
        
        ratio_min = 0.5
        ratio_max = 1 / 0.5
        synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

        if(synth_ratio*synth_area<=0):
            print(synth_area,xmax,xmin,ymax,ymin)
            print(synth_ratio,ratio_max,ratio_min)           
        synth_h = math.sqrt(synth_area * synth_ratio)
        synth_w = math.sqrt(synth_area / synth_ratio)
        synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
        synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

        if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
            synth_xmin = int(synth_xmin)
            synth_ymin = int(synth_ymin)
            synth_w = int(synth_w)
            synth_h = int(synth_h)
            break
    return synth_ymin, synth_h, synth_xmin, synth_w

def apply_color_jitter(rgb):
    color_factor=2*random.random()
    c_high = 1 + color_factor
    c_low = 1 - color_factor
    rgb=rgb.copy()
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
    rgb[:, :, 1] = np.clip(rgb[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2] * random.uniform(c_low, c_high), 0, 255)
    rgb = Image.fromarray(rgb)
    return rgb

def apply_occlusion(rgb):
    rgb=np.array(rgb)
    h,w,_ = rgb.shape
    synth_ymin, synth_h, synth_xmin, synth_w = occlusion_aug(None,np.array([h,w]), min_area=0.0, max_area=0.3, max_try_times=5)
    rgb = rgb.copy()
    rgb[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
    rgb = Image.fromarray(rgb)
    return rgb

def apply_rgb_aug(rgb):
    augSharpness = PillowSharpness(p=0.6, factor_interval=(0., 50.)) #0.3
    augContrast = PillowContrast(p=0.6, factor_interval=(0.7, 1.8)) #0.3
    augBrightness = PillowBrightness(p=0.6, factor_interval=(0.7, 1.8)) #0.3
    augColor = PillowColor(p=0.6, factor_interval=(0., 4.)) #0.3
    mask = None
    state = None
    rgb, mask, state = augSharpness(rgb, mask, state)
    rgb, mask, state = augContrast(rgb, mask, state)
    rgb, mask, state = augBrightness(rgb, mask, state)
    rgb, mask, state = augColor(rgb, mask, state)  
    rgb = np.array(rgb)
    rgb = Image.fromarray(rgb)
    return rgb