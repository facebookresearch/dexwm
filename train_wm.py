import numpy as np
import argparse
import yaml
import torch
import torch.nn as nn
import sys
import os
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
    CheckpointImpl,
)
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig, FullOptimStateDictConfig

from distributed import init_distributed, cleanup
from datasets.egodex import EgoDexDataset
from datasets.robocasa_random_movement import RobocasaRandomDataset
from datasets.droid import DroidDataset
from datasets.egodex_and_droid import EgodexDroidDataset
from models.model import DexWM, CDiTBlock
import matplotlib.pyplot as plt
from functools import partial
import wandb
import torch._dynamo

def get_latest_checkpoint(args):
    latest_ckpt = None
    if hasattr(args, 'job_dir'):
        job_dir = args.job_dir
    else:
        job_dir = args['job_dir']

    if os.path.exists(f'{job_dir}/checkpoints'):
        ckpts = os.listdir(f'{job_dir}/checkpoints')
        if len(ckpts) > 0:
            ckpts = sorted(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_ckpt = ckpts[-1].replace('.pth.tar', '')
            print(f'Resuming from {latest_ckpt}')

    return latest_ckpt

def get_patch_size_from_backbone(backbone_name):
    if 'dinov2' in backbone_name or 'siglip' in backbone_name or 'webssl' in backbone_name:
        patch_size = 14
        num_patches = 448
    elif 'dinov3' in backbone_name or 'vjepa' in backbone_name:
        patch_size = 16
        num_patches = 336
    else:
        raise ValueError(f'Backbone {backbone_name} not supported')
    return patch_size, num_patches

def main(args_temp, args):
    dataset_name = args['dataset']
    max_context_len = args['data']['max_context_len']
    num_context = args['data']['num_context']
    img_size = args['data']['img_size']
    aug = args['data']['aug']
    keys = args['data'].get('keys', 'all')
    full_seq = args['data'].get('full_seq', False)
    var_time = args['data'].get('var_time', False)

    backbone_name = args['model']['backbone_name']
    hidden_dim = args['model']['hidden_dim']
    action_dim = args['model']['action_dim']
    depth = args['model']['depth']
    num_heads = args['model']['num_heads']
    mlp_ratio = args['model']['mlp_ratio']
    do_compile = args['model']['do_compile']

    batch_size = args['train']['batch_size']
    num_workers = args['train']['num_workers']
    epochs = args['train']['epochs']
    global_seed = args['train']['global_seed']
    save_name = args['train']['save_name']
    resume = args['train']['resume']
    do_eval = args['train']['do_eval']
    eval_freq = args['train']['eval_freq']
    kp_weight = args['train'].get('kp_weight', 1)

    patch_size, num_patches = get_patch_size_from_backbone(backbone_name)

    if args_temp is not None:
        use_fsdp = args_temp.use_fsdp
    else:
        use_fsdp = False

    _, rank, device, _ = init_distributed()

    if rank == 0:
        if args['wandb']['do_wandb']:   # set to false during debugging to avoid recording results
            wandb.init(project=args['wandb']['project'],
                    entity=args['wandb']['entity'],
                    name=args['wandb']['name'],
                    config=args)

    seed = global_seed # (DistributedSampler + model init need consistent seed across ranks) global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}, device={device}", flush=True)

    # Set default number of keypoints (will be overridden for baseline codes)
    num_keypoints = 12

    if dataset_name=='egodex':
        root_folder = args['data']['root_folder']
        train_subset = EgoDexDataset(root_folder=root_folder, max_context_len=max_context_len, num_context=num_context, patch_size=patch_size,
                                    backbone_name=backbone_name, img_size=img_size, aug=aug, train=True, keys=keys, var_time=var_time)
        # Get the number of keypoints for the model based on the dataset configuration
        num_keypoints = len(train_subset.hand_and_tip_keys)

        val_subset = EgoDexDataset(root_folder=root_folder, max_context_len=max_context_len, num_context=num_context, patch_size=patch_size,
                                    backbone_name=backbone_name, img_size=img_size, aug=False, train=False, keys=keys, var_time=var_time)

    elif dataset_name=='robocasa_random':
        root_folder = args['data']['root_folder']
        train_subset = RobocasaRandomDataset(root_folder=root_folder, max_context_len=max_context_len, num_context=num_context, patch_size=patch_size,
                                    backbone_name=backbone_name, img_size=img_size, aug=aug, train=True, full_seq=full_seq)

        val_subset = RobocasaRandomDataset(root_folder=root_folder, max_context_len=max_context_len, num_context=num_context, patch_size=patch_size,
                                    backbone_name=backbone_name, img_size=img_size, aug=False, train=False, full_seq=full_seq)

    elif dataset_name=='droid':
        root_folder = args['data']['root_folder']
        num_keypoints = 22
        train_subset = DroidDataset(root_folder=root_folder, max_context_len=max_context_len, num_context=num_context, patch_size=patch_size,
                                    backbone_name=backbone_name, img_size=img_size, aug=aug, train=True, var_time=var_time, num_keypoints=num_keypoints)

        val_subset = DroidDataset(root_folder=root_folder, max_context_len=max_context_len, num_context=num_context, patch_size=patch_size,
                                    backbone_name=backbone_name, img_size=img_size, aug=False, train=False, var_time=var_time, num_keypoints=num_keypoints)

    elif dataset_name=='egodex_and_droid':
        egodex_root_folder = args['data']['egodex_root_folder']
        droid_root_folder = args['data']['droid_root_folder']
        train_subset = EgodexDroidDataset(egodex_root_folder=egodex_root_folder, droid_root_folder=droid_root_folder,
                                    max_context_len=max_context_len, num_context=num_context, patch_size=patch_size,
                                    backbone_name=backbone_name, img_size=img_size, aug=aug, train=True, keys=keys, var_time=var_time)
        val_subset = EgodexDroidDataset(egodex_root_folder=egodex_root_folder, droid_root_folder=droid_root_folder,
                                    max_context_len=max_context_len, num_context=num_context, patch_size=patch_size,
                                    backbone_name=backbone_name, img_size=img_size, aug=False, train=False, keys=keys, var_time=var_time)



    train_sampler = DistributedSampler(train_subset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=global_seed)
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False,
        shuffle=False,
        drop_last=True,
        sampler=train_sampler)

    val_sampler = DistributedSampler(val_subset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=global_seed)
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False,
        shuffle=False,
        drop_last=True,
        sampler=val_sampler)

    dexwm = DexWM(backbone_name=backbone_name,
                num_patches=num_patches,
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_context = num_context,
                emb_loss_fn=nn.MSELoss(reduction='mean'),
                use_gradient_checkpointing=True,
                use_fsdp=use_fsdp,
                num_keypoints=num_keypoints)


    if not use_fsdp:
        dexwm.to(device)
        dexwm = DDP(dexwm, device_ids=[device])
    else:
        # Use FSDP 1 since 2 is buggy
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32
        )

        dexwm = FSDP(
            dexwm,
            auto_wrap_policy=ModuleWrapPolicy({CDiTBlock}),
            sync_module_states=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision_policy,
            use_orig_params=True,
            # ignored_modules=[dexwm.image_embedder],
            device_id=device
        )

        if dexwm.use_gradient_checkpointing:
            non_reentrant_wrapper = partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                dexwm, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=lambda submodule: isinstance(submodule, (CDiTBlock, ResBlock))
            )

    if do_compile:
        dexwm = torch.compile(dexwm)

    if rank == 0:
        print(dexwm)

    if hasattr(dexwm, 'module'):
        model_for_optim = dexwm.module
    else:
        model_for_optim = dexwm

    optimizer = torch.optim.AdamW(model_for_optim.parameters(), lr=float(args['train']['optim']['lr']),
                                    weight_decay=float(args['train']['optim']['weight_decay']))
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        pct_start=0.0,
        final_div_factor=float(args['train']['scheduler']['final_div_factor']),
        max_lr=float(args['train']['scheduler']['max_lr']),
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        cycle_momentum=False
    )

    last_epoch = 0
    start_step = 0
    scaler = torch.amp.GradScaler('cuda')

    latest_ckpt = get_latest_checkpoint(args)
    # Always prefer the latest checkpoint if it exists (handles preemption properly)
    # Only use the specified resume checkpoint if no latest checkpoint is found
    if latest_ckpt is not None:
        resume = latest_ckpt
        print(f"Found latest checkpoint, using: {resume}")
    elif resume is not None:
        print(f"No latest checkpoint found, using specified resume: {resume}")
    else:
        print("No checkpoint to resume from, starting fresh")

    if resume is not None:
        if '/checkpoints/' in resume:
            checkpoint_path = resume
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint_dir = f"{args['job_dir']}/checkpoints"
            checkpoint_path = f"{checkpoint_dir}/{resume}.pth.tar"
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        pretrained_dict = checkpoint["model"]

        if use_fsdp:
            dexwm.load_state_dict(pretrained_dict)
            if ('robocasa' not in dataset_name) or ('robocasa' in resume):   # load optimizer for robocasa only when resuming from a robocasa savepoint
                sharded_optim_state_dict = FSDP.optim_state_dict_to_load(
                    dexwm, optimizer, checkpoint["opt"]
                )
                optimizer.load_state_dict(sharded_optim_state_dict)
        else:
            dexwm.module.load_state_dict(pretrained_dict)
            if ('robocasa' not in dataset_name) or ('robocasa' in resume):
                optimizer.load_state_dict(checkpoint["opt"])
        if ('robocasa' not in dataset_name) or ('robocasa' in resume):
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
            last_epoch = checkpoint["epoch"]
            start_step = checkpoint["train_steps"] + 1
            scaler.load_state_dict(checkpoint["scaler"])

            # Reseed sampler
            train_loader.sampler.seed = seed + start_step
            print(f"Reseeding with {seed + start_step}")



    def train_fn(model, data_loader, optimizer, lr_scheduler, train, epoch_num, start_step):
        if train:
            model.train()
        else:
            model.eval()
        tot_emb_loss = 0
        tot_kp_loss = 0
        if rank==0:
            iterator = tqdm(data_loader, dynamic_ncols=True, leave=True, file=sys.stdout)
        else:
            iterator = data_loader

        if start_step > 0 and rank == 0:
            print(f'Skipping {start_step} steps')

        num_steps = len(iterator) - start_step
        i = start_step
        for (curr_frames, actions, rel_t, heatmaps, valid_kp, metadata) in iterator:
            if i >= num_steps:
                # Update LR too
                lr_scheduler.step()
                break
            optimizer.zero_grad()
            model.zero_grad()
            curr_frames = curr_frames.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            valid_kp = valid_kp.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                goal_pred, goal_tgt, pred_kp, emb_loss, kp_loss = model(curr_frames, actions, rel_t, gt_kps=heatmaps, valid_kp=valid_kp)
                loss = emb_loss + kp_weight*kp_loss

                if use_fsdp:
                    # FSDP mixed precision policy handles loss scaling already
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                lr_scheduler.step()
            tot_emb_loss += emb_loss.item()
            tot_kp_loss += kp_loss.item()
            # Calculate the correct global step for WandB
            global_step = epoch_num * len(data_loader) + i
            if rank==0:
                iterator.set_postfix({'emb_loss': f'{emb_loss.detach().cpu().numpy():.3f}',
                                        'kp_loss': f'{kp_loss.detach().cpu().numpy():.5f}'})
                if args['wandb']['do_wandb']:
                    wandb.log({'train_emb_loss': emb_loss.item(),
                            'kp_loss': kp_loss.item(),
                            'lr': lr_scheduler.get_last_lr()[0]}, step=global_step)

            if (i+1) % eval_freq == 0:
                if do_eval:
                    if rank==0:
                        print('Validating')
                    val_fn(model, val_loader, global_step)
                model.train()

                checkpoint_dir = f"{args['job_dir']}/checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{checkpoint_dir}/{save_name}_{e}.pth.tar"

                if use_fsdp:
                    FSDP.set_state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                        FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
                    )
                    model_state_dict = model.state_dict()
                    optimizer_state_dict = FSDP.optim_state_dict(model, optimizer)
                else:
                    optimizer_state_dict = optimizer.state_dict()
                    if hasattr(model, 'module'):
                        model_state_dict = model.module.state_dict()
                    else:
                        model_state_dict = model.state_dict()

                if rank==0:
                    checkpoint = {
                            "model": model_state_dict,
                            "opt": optimizer_state_dict,
                            "scheduler": lr_scheduler.state_dict(),
                            "args": args,
                            "epoch": e,
                            "train_steps": i,
                            "scaler": scaler.state_dict()
                        }
                    torch.save(checkpoint, checkpoint_path)

            i += 1

    def val_fn(model, data_loader, global_step):
        model.eval()
        tot_emb_loss = 0
        tot_kp_loss = 0
        if rank==0:
            iterator = tqdm(data_loader, dynamic_ncols=True, leave=True, file=sys.stdout)
        else:
            iterator = data_loader
        for i, (curr_frames, actions, rel_t, heatmaps, valid_kp, metadata) in enumerate(iterator):
            optimizer.zero_grad()
            model.zero_grad()
            curr_frames = curr_frames.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            valid_kp = valid_kp.to(device, non_blocking=True)

            with torch.inference_mode():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    goal_pred, goal_tgt, pred_kp, emb_loss, kp_loss = model(curr_frames, actions, rel_t, gt_kps=heatmaps, valid_kp=valid_kp)
            tot_emb_loss += emb_loss.item()
            tot_kp_loss += kp_loss.item()
            if rank==0:
                iterator.set_postfix({'avg_emb_loss': f'{(tot_emb_loss/(i+1)):.3f}',
                                        'avg_kp_loss': f'{(tot_kp_loss/(i+1)):.5f}'})
        if rank==0:
            if args['wandb']['do_wandb']:
                wandb.log({'val_emb_loss': tot_emb_loss/len(data_loader),
                        'val_kp_loss': tot_kp_loss/len(data_loader)}, step=global_step)



    for e in range(last_epoch, epochs):
        if rank==0:
            print('Epoch', e)
        train_sampler.set_epoch(e)
        train_fn(dexwm, train_loader, optimizer, lr_scheduler, train=True, epoch_num=e, start_step=start_step)
        # Reset start_step for next epoch
        start_step = 0

    cleanup()


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_dir", default="experiments/temp", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument('--config', type=str, default='configs/egodex_and_droid.yaml', help='Path to the config file')
    args = parser.parse_args()
    fname = parser.parse_args().config


    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    os.makedirs(args.job_dir, exist_ok=True)
    params['job_dir'] = args.job_dir
    main(None, params)
