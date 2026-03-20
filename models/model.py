# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import copy
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F
from typing import Type
from functools import partial
from einops import rearrange, repeat
from transformers import AutoModel
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from einops import rearrange
from .decoder_vit import SimpleViTDecoder



def blockwise_spatial_mask(b, h, q_idx, kv_idx, num_frames=5, num_tokens=196):
    n = num_tokens

    # Iterate over frames
    for i in range(num_frames):

        start_i = i * n
        end_i = (i+1) * n
        q_mask_1 = start_i <= q_idx
        q_mask_2 = q_idx < end_i
        q_mask = q_mask_1 &  q_mask_2

        kv_idx_1 = start_i <= kv_idx
        kv_idx_2 = kv_idx < end_i
        kv_mask = kv_idx_1 & kv_idx_2

        curr_m = q_mask * kv_mask

        if i == 0:
            m = curr_m
        else:
            m = m | curr_m
    return m

def blockwise_spatial_mask_eval(b, h, q_idx, kv_idx, num_frames=5, num_tokens=196):
    n = num_tokens
    i = num_frames - 1
    start_i = i * n
    end_i = (i+1) * n
    q_mask_1 = start_i <= q_idx
    q_mask_2 = q_idx < end_i
    q_mask = q_mask_1 &  q_mask_2

    kv_idx_1 = start_i <= kv_idx
    kv_idx_2 = kv_idx < end_i
    kv_mask = kv_idx_1 & kv_idx_2

    return q_mask * kv_mask

def blockwise_temporal_mask(b, h, q_idx, kv_idx, num_frames=5, num_tokens=196):
    n = num_tokens
    # Iterate over frames
    m = None
    for i in range(num_frames):
        start_i = i * n
        end_i = start_i + n
        q_mask_1 = start_i <= q_idx
        q_mask_2 = q_idx < end_i
        q_mask = q_mask_1 &  q_mask_2
        kv_mask = kv_idx < start_i
        curr_m = q_mask * kv_mask
        if m is None:
            m = curr_m
        else:
            m = m | curr_m
    return m

def blockwise_temporal_mask_eval(b, h, q_idx, kv_idx, num_frames=5, num_tokens=196):
    n = num_tokens
    # Only last frame cross attends to past frames
    m = None
    i = num_frames - 1
    start_i = i * n
    end_i = start_i + n
    q_mask_1 = start_i <= q_idx
    q_mask_2 = q_idx < end_i
    q_mask = q_mask_1 &  q_mask_2
    kv_mask = kv_idx < start_i
    curr_m = q_mask * kv_mask
    if m is None:
        m = curr_m
    else:
        m = m | curr_m
    return m

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)

class MHA(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        block_mask=None,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = num_heads
        self.dropout = dropout
        self.packed_proj = nn.Linear(hidden_size, hidden_size * 3, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
        assert hidden_size % num_heads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = hidden_size // num_heads
        self.bias = bias
        if block_mask is not None:
            self.block_mask = block_mask


    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if query is key and key is value:
            result = self.packed_proj(query)
            query, key, value = torch.chunk(result, 3, dim=-1)
        else:
            q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
            if self.bias:
                q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
            else:
                q_bias, k_bias, v_bias = None, None, None
            query, key, value = F.linear(query, q_weight, q_bias), F.linear(key, k_weight, k_bias), F.linear(value, v_weight, v_bias)

        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        attn_output = flex_attention(query, key, value, block_mask=self.block_mask.to(query.device))

        attn_output = attn_output.transpose(1, 2).flatten(-2)
        attn_output = self.out_proj(attn_output)

        return attn_output

class CDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, adain_input_size, mlp_ratio=4.0, spatial_mask=None, temporal_mask=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MHA(hidden_size, num_heads=num_heads, bias=True, block_mask=spatial_mask, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = MHA(hidden_size, num_heads=num_heads, bias=True, block_mask=temporal_mask, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adain_input_size, 11 * hidden_size, bias=True)
        )

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c, num_cond, x_clean):
        shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(11, dim=-1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa).flatten(1,2)
        x = x + gate_msa.unsqueeze(2) * self.attn(query=x_norm, key=x_norm, value=x_norm).unflatten(1, (num_cond + 1, -1))

        x_cond_norm = modulate(self.norm_cond(x_clean), shift_ca_xcond, scale_ca_xcond).flatten(1,2)
        x_norm = modulate(self.norm2(x), shift_ca_x, scale_ca_x).flatten(1,2)
        x = x + gate_ca_x.unsqueeze(2) * self.cttn(query=x_norm, key=x_cond_norm, value=x_cond_norm).unflatten(1, (num_cond + 1, -1))

        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(2) * self.mlp(x_norm)

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, adain_input_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adain_input_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class HeatmapModel(nn.Module):
    def __init__(
        self,
        encoder_dim=1024,
        decoder_dim=256,
        target_resolution=224,
        patch_size=14,
        num_layers=6,
        num_heads=16,
        mlp_ratio=4.0,
        dropout=0.0,
        output_channels=12
    ):
        super().__init__()
        self.vit_decoder = SimpleViTDecoder(encoder_dim=encoder_dim,        # Input encoder dimension (raw pixel patches)
                                            decoder_dim=decoder_dim,       # Decoder hidden dimension
                                            num_layers=num_layers,          # Number of transformer layers
                                            num_heads=num_heads,           # Attention heads
                                            mlp_ratio=mlp_ratio,          # MLP expansion ratio
                                            target_resolution=target_resolution,  # Output image resolution
                                            patch_size=patch_size,          # Output patch size
                                            dropout=dropout,             # Dropout
                                            output_channels=output_channels,
                                            do_tanh=False,
                                            use_torch_attn=False
                                            )

    def forward(self, x, hw):
        x = self.vit_decoder(x, hw_input=hw)
        return x

class DexWM(nn.Module):
    def __init__(
        self,
        backbone_name,
        num_patches=256,
        patch_size=14,
        hidden_dim=1024,
        action_dim = 42,
        depth=12,
        num_heads=16,
        mlp_ratio=2.0,
        num_context=4,
        is_eval=False,
        emb_loss_fn=nn.MSELoss(),
        use_gradient_checkpointing=True,
        use_fsdp=False,
        num_keypoints=12,
        heatmap_dim=256,
        heatmap_layers=6
    ):
        super().__init__()

        if backbone_name=='vjepa2_vitg_256':
            # self.image_embedder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
            self.image_embedder = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
            encoder_dim = 1408
        if backbone_name=='dinov2':
            self.image_embedder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            encoder_dim = 1024
        if backbone_name=='dinov3_large':
            self.image_embedder = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
            encoder_dim = 1024
        if backbone_name=='siglip2_so400m_patch14_384':
            self.image_embedder = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384")
            encoder_dim = 1152
            # Fix for torch.compile: Move CPU-pinned position_ids buffers to CUDA
            # SigLIP has position_ids as persistent buffers on CPU which breaks compile
            if hasattr(self.image_embedder, 'vision_model'):
                if hasattr(self.image_embedder.vision_model.embeddings, 'position_ids'):
                    self.image_embedder.vision_model.embeddings.register_buffer(
                        'position_ids',
                        self.image_embedder.vision_model.embeddings.position_ids.cuda(),
                        persistent=False
                    )
                if hasattr(self.image_embedder.text_model.embeddings, 'position_ids'):
                    self.image_embedder.text_model.embeddings.register_buffer(
                        'position_ids',
                        self.image_embedder.text_model.embeddings.position_ids.cuda(),
                        persistent=False
                    )
        if backbone_name=='webssl_vitl':
            self.image_embedder = AutoModel.from_pretrained("facebook/webssl-dino300m-full2b-224")
            encoder_dim = 1024

        for p in self.image_embedder.parameters():
            p.requires_grad=False

        if hidden_dim != encoder_dim:
            self.input_proj = nn.Linear(encoder_dim, hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, encoder_dim)

        self.encoder_dim = encoder_dim
        self.backbone_name = backbone_name
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.is_eval = is_eval
        self.num_context = num_context
        self.action_dim=action_dim  # num_joints * 3 + 6 (for camera)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_fsdp = use_fsdp

        inference_context_size = num_context
        window_size = num_patches*(inference_context_size + 1)

        if self.is_eval == 1:
            blockwise_spatial_partial = partial(blockwise_spatial_mask_eval, num_frames=inference_context_size + 1, num_tokens=num_patches)
            blockwise_temporal_partial = partial(blockwise_temporal_mask_eval, num_frames=inference_context_size + 1, num_tokens=num_patches)
        else:
            blockwise_spatial_partial = partial(blockwise_spatial_mask, num_frames=inference_context_size + 1, num_tokens=num_patches)
            blockwise_temporal_partial = partial(blockwise_temporal_mask, num_frames=inference_context_size + 1, num_tokens=num_patches)
        spatial_mask = create_block_mask(blockwise_spatial_partial, B=1, H=1, Q_LEN=window_size, KV_LEN=window_size)
        temporal_mask = create_block_mask(blockwise_temporal_partial, B=1, H=1, Q_LEN=window_size, KV_LEN=window_size)
        self.temporal_mask = temporal_mask

        self.blocks = nn.ModuleList([CDiTBlock(hidden_dim, num_heads, self.action_dim, mlp_ratio=mlp_ratio, spatial_mask=spatial_mask, temporal_mask=temporal_mask) for _ in range(depth)])

        self.final_layer = FinalLayer(hidden_dim, adain_input_size=self.action_dim)
        self.pos_embed = nn.Parameter(torch.zeros(self.num_context + 1, self.num_patches, hidden_dim), requires_grad=True)


        self.emb_loss_fn = emb_loss_fn

        heatmap_patch_size = patch_size

        self.kp_layer = HeatmapModel(encoder_dim=encoder_dim,
                                decoder_dim=heatmap_dim,
                                target_resolution=224,
                                patch_size=heatmap_patch_size,
                                num_layers=heatmap_layers,
                                num_heads=16,
                                mlp_ratio=4.0,
                                dropout=0.0,
                                output_channels=num_keypoints)

        self.kp_loss_fn = torch.nn.MSELoss(reduction='none')

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        nn.init.normal_(self.pos_embed, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def encode_image(self, x):
        B, T, C, H, W = x.shape
        # print(x.shape)
        with torch.no_grad():
            if 'vjepa' in self.backbone_name:
                x = x.repeat_interleave(2, dim=1)
                x = x.reshape(B, T, 2, C, H, W)
                x = x.reshape(-1, 2, C, H, W)
                x = self.image_embedder(x).last_hidden_state
                x = torch.nn.functional.layer_norm(x, x.shape[-1:], weight=None, bias=None)
            elif 'dinov2' in self.backbone_name:
                x = x.contiguous()
                x = x.view(-1, C, H, W)
                x = self.image_embedder.forward_features(x)['x_norm_patchtokens']   # (B*T, 448, 1024)
            elif 'dinov3' in self.backbone_name:
                x = x.contiguous()
                x = x.view(-1, C, H, W)
                x = self.image_embedder.forward(x).last_hidden_state[:, 1 + self.image_embedder.config.num_register_tokens:]
            elif 'webssl' in self.backbone_name:
                x = x.contiguous()
                x = x.view(-1, C, H, W)
                x = self.image_embedder.forward(x).last_hidden_state[:, 1:]
            elif 'siglip2' in self.backbone_name:
                x = x.contiguous()
                x = x.view(-1, C, H, W)
                x = self.image_embedder.vision_model(x, interpolate_pos_encoding=True).last_hidden_state   # (B*T, 448, 1152)
            _, P, F = x.shape
            x = x.view(B, T, P, F)
        return x

    def prepare_actions(self, all_uvz, action_diff):
        if not action_diff:
            d = all_uvz.shape[2] // 2
            curr_xyz = all_uvz[:,:,:d]
            next_xyz = all_uvz[:,:,d:]
            xyz = next_xyz - curr_xyz
            xyz_norm = xyz
            xyz_norm[:,:,-1] = xyz_norm[:,:,-1]%(2*np.pi) # assuming last one is a rotation angle
        else:
            xyz_norm = all_uvz
        actions = torch.cat([xyz_norm.flatten(2)], dim=-1)
        return actions

    def forward_kp(self, x_emb, cam_pose, gt_kps, valid_kp):
        x_goal = x_emb.clone()
        if self.patch_size==14:
            hw = [16, 28]
        if self.patch_size==16:
            hw = [14, 24]

        kps_heatmap = self.kp_layer(x_goal[:,1:], hw)
        B, T, C, H, W = kps_heatmap.shape
        kps_heatmap = kps_heatmap.view(B*T*C,H,W)
        if gt_kps is None:
            kp_loss = None
        else:
            gt_kps = gt_kps.view(B*T*C,H,W)
            valid_kp = valid_kp.view(B*T*C)
            kp_loss = self.kp_loss_fn(kps_heatmap, gt_kps).mean([1,2])
            kp_loss = (kp_loss*valid_kp).mean()
            kp_loss = kp_loss
        return kps_heatmap, kp_loss

    def forward(self, x, actions=None, rel_t=None, prev_emb=None, action_diff=False,
                cam_pose=None, gt_kps=None, only_kp=False, valid_kp=None):
        """
        x: (B, context+1, C, H, W)
        goals: (B, C, H, W)
        actions: (B, num_joints, 3)
        rel_t: (B,)
        """
        x_emb = self.encode_image(x)

        if only_kp:
            return self.forward_kp(x_emb, cam_pose, gt_kps, valid_kp)
        if prev_emb is not None:   # this is used for multistep prediction
            # print(x_emb.shape, prev_emb.shape)
            prev_emb = prev_emb[:,-8:]  # when making long predictions, we want to still only consider the last 8 frames
            T_in_pred = prev_emb.shape[1]
            x_emb = torch.cat([
                    x_emb[:, : -T_in_pred-1],
                    prev_emb,
                    x_emb[:, -1:]
                ], dim=1)
        x_goal = x_emb.clone()

        # CDiT-B / whenever Transformer dimension doesn't match encoder dimension
        if hasattr(self, 'input_proj'):
            x_emb = self.input_proj(x_emb)
        x_emb = x_emb + self.pos_embed[:self.num_context+1]

        x_context = x_emb.clone()

        x_emb_clone = x_emb[:, :-1].clone()  # Clone the slice to ensure no shared memory
        x_emb[:, 1:] = x_emb_clone

        # x_emb represents the “future” embeddings at successive timesteps (as in CDiT’s noisy future embeddings).
        # Since we do not use a noising/denoising process in DexWM, we initialize each future step by shifting the
        # embeddings by one timestep and reusing the previous step:
        #   original: [T, T+1, T+2, ..., T+N]
        #   shifted : [T, T,   T+1, ..., T+(N-1)]
        # The model predicts the next-step targets [T+1, T+2, ..., T+N] (i.e., a one-step-ahead prediction).
        # The prediction at index 0 is excluded from the loss, so x_emb[0] is effectively unused; it is kept
        # only to preserve the existing tensor shapes.

        c = self.prepare_actions(actions, action_diff)
        c = torch.cat([torch.zeros_like(c)[:,:1], c], dim=1)  # because x_emb has timesteps [T, T, T+1, ...], therefore, 0th action is 0

        num_cond = self.num_context
        for idx, block in enumerate(self.blocks):
            if self.is_eval or not self.use_gradient_checkpointing or self.use_fsdp:
                # if FSDP checkpoint block is registered
                x_emb = block(x_emb, c, num_cond, x_context)
            else:
                x_emb = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x_emb, c, num_cond, x_context, use_reentrant=False)       # (N, T, D)

        x_pred = self.final_layer(x_emb, c)

        if hasattr(self, 'output_proj'):
            x_pred_for_emb_loss = self.output_proj(x_pred)
        else:
            x_pred_for_emb_loss = x_pred

        emb_loss = self.emb_loss_fn(x_pred_for_emb_loss[:,1:], x_goal[:,1:])

        pred_kps, kp_loss = self.forward_kp(x_pred_for_emb_loss, cam_pose, gt_kps, valid_kp)

        return x_pred_for_emb_loss, x_goal, pred_kps, emb_loss, kp_loss
