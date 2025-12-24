#!/bin/bash

# This script is used for Causal VAE Training
# It undergoes a two-stage training
# Stage-1: image and video mixed training
# Stage-2: pure video training, using context parallel to load video with more video frames (up to 257 frames)

# Bid dataset use the `kl_weight = 1e-4`

GPUS=1  # The gpu number
VAE_MODEL_PATH=PATH/vae_ckpt   # The vae model dir
LPIPS_CKPT=PATH/vgg_lpips.pth    # The LPIPS VGG CKPT path, used for calculating the lpips loss
# OUTPUT_DIR=/PATH/output_dir    # The checkpoint saving dir
OUTPUT_DIR=/content/drive/MyDrive/output_dir    # The checkpoint saving dir

VIDEO_ANNO=annotation/video_data_files_path.jsonl   # The video annotation file path
RESOLUTION=256     # The training resolution, default is 256
NUM_FRAMES=9     # x * 8 + 1, the number of video frames
BATCH_SIZE=1


# # Update the Stage-1 training.
# torchrun --nproc_per_node $GPUS \
#     train/train_video_vae.py \
#     --num_workers 6 \
#     --model_path $VAE_MODEL_PATH \
#     --model_dtype bf16 \
#     --lpips_ckpt $LPIPS_CKPT \
#     --output_dir $OUTPUT_DIR \
#     --video_anno $VIDEO_ANNO \
#     --resolution $RESOLUTION \
#     --max_frames $NUM_FRAMES \
#     --disc_start 0.0 \
#     --kl_weight 1e-6 \
#     --pixelloss_weight 1.0 \
#     --perceptual_weight 1.0 \
#     --disc_weight 0.5 \
#     --batch_size $BATCH_SIZE \
#     --opt adamw \
#     --opt_betas 0.9 0.95 \
#     --seed 42 \
#     --weight_decay 1e-3 \
#     --clip_grad 1.0 \
#     --lr 1e-5 \
#     --lr_disc 1e-5 \
#     --warmup_epochs 1 \
#     --epochs 100 \
#     --iters_per_epoch 2000 \
#     --print_freq 40 \
#     --save_ckpt_freq 1 \
#     --add_discriminator



# Update the Stage-1 training.
torchrun --nproc_per_node $GPUS \
    train/train_video_vae.py \
    --num_workers 6 \
    --model_path $VAE_MODEL_PATH \
    --model_dtype bf16 \
    --lpips_ckpt $LPIPS_CKPT \
    --output_dir $OUTPUT_DIR \
    --video_anno $VIDEO_ANNO \
    --image_mix_ratio 0.0 \
    --resolution $RESOLUTION \
    --max_frames $NUM_FRAMES \
    --disc_start 5000 \          # CHANGE 1: Delayed start (was 2000). Let pixel loss stabilize first.
    --kl_weight 1e-6 \           # CHANGE 2: Increased stability (was 1e-12). 
    --pixelloss_weight 1.0 \     # Keep this at 1.0 (Good).
    --perceptual_weight 1.0 \
    --disc_weight 0.5 \
    --batch_size $BATCH_SIZE \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --seed 42 \
    --weight_decay 1e-3 \
    --clip_grad 1.0 \
    --lr 1e-5 \                  # CHANGE 3: Lowered 10x (was 1e-4). This is the most critical fix.
    --lr_disc 1e-5 \             # CHANGE 4: Lowered 10x (was 1e-4).
    --warmup_epochs 1 \
    --epochs 100 \
    --iters_per_epoch 2000 \
    --print_freq 40 \
    --save_ckpt_freq 1 \
    --add_discriminator
