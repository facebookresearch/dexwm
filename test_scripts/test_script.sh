python test_scripts/test_models.py \
    --config configs/egodex.yaml \
    --model_checkpoint /checkpoint/amaia/video/raktimgg/HOWM/experiments/egodex_and_droid_heatmap_loss_aug_weight100_no_flip_var_time2/checkpoints/egodex_and_droid_heatmap_loss_weight100_39.pth.tar \
    --kp_config configs/egodex_kp_layer.yaml \
    --kp_checkpoint /checkpoint/amaia/video/raktimgg/HOWM/experiments/egodex_kp_layer_heatmap_large/checkpoints/egodex_kp_layer_39.pth.tar \
    --output_dir debug
