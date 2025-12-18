import sys
import os
sys.path.append(os.path.abspath('.'))

import datetime
import numpy as np
import time
import torch

import json
import random
from pathlib import Path




from dataset import (
    VideoDataset,
    create_mixed_dataloaders
)

from trainer_misc import (
    NativeScalerWithGradNormCount,
    create_optimizer,
    train_one_epoch,
    auto_load_model,
    save_model,
    cosine_scheduler,
)

from video_vae import CausalVideoVAELossWrapper
from args import get_args


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True





def build_model(args):
    model_dtype = args.model_dtype
    model_path = args.model_path

    print(f"Load the base VideoVAE checkpoint from path: {model_path}, using dtype {model_dtype}")

    model = CausalVideoVAELossWrapper(
        model_path,
        model_dtype='fp32',      # For training, we used mixed training
        disc_start=args.disc_start,
        logvar_init=args.logvar_init,
        kl_weight=args.kl_weight,
        pixelloss_weight=args.pixelloss_weight,
        perceptual_weight=args.perceptual_weight,
        disc_weight=args.disc_weight,
        interpolate=False,
        add_discriminator=args.add_discriminator,
        freeze_encoder=args.freeze_encoder,
        load_loss_module=True,
        lpips_ckpt=args.lpips_ckpt,
    )

    if args.pretrained_vae_weight:
        pretrained_vae_weight = args.pretrained_vae_weight
        print(f"Loading the vae checkpoint from {pretrained_vae_weight}")
        model.load_checkpoint(pretrained_vae_weight)

    return model


def main(args):
    

    # print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(args)
    
    num_training_steps_per_epoch = args.iters_per_epoch
    log_writer = None



    video_dataset = VideoDataset(args.video_anno)

    # print(video_dataset)
    # data_loader_train = DataLoader(video_dataset, 
    #                               batch_size=args.batch_size)

    data_loader_train = create_mixed_dataloaders(
        video_dataset,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        epoch=args.seed
    )
    
    

    model.to(device)
    model_without_ddp = model

    n_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fix_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            print(name)
    print(f'total number of learnable params: {n_learnable_parameters / 1e6} M')
    print(f'total number of fixed params in : {n_fix_parameters / 1e6} M')

    total_batch_size = args.batch_size
    print("LR = %.8f" % args.lr)
    print("Min LR = %.8f" % args.min_lr)
    print("Weigth Decay = %.8f" % args.weight_decay)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % (num_training_steps_per_epoch * args.epochs))
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model_without_ddp.vae)
    optimizer_disc = create_optimizer(args, model_without_ddp.loss.discriminator) if args.add_discriminator else None

    loss_scaler = NativeScalerWithGradNormCount(enabled=True if args.model_dtype == "fp16" else False)
    loss_scaler_disc = NativeScalerWithGradNormCount(enabled=True if args.model_dtype == "fp16" else False) if args.add_discriminator else None

    
    model_without_ddp = model

    print("Use step level LR & WD scheduler!")

    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    lr_schedule_values_disc = cosine_scheduler(
        args.lr_disc, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    ) if args.add_discriminator else None

    auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, 
        loss_scaler=loss_scaler, optimizer_disc=optimizer_disc,
    )
    
    print(f"Start training for {args.epochs} epochs, the global iterations is {args.global_step}")
    start_time = time.time()
   
            
    for epoch in range(args.start_epoch, args.epochs):
        
        train_stats = train_one_epoch(
            model, 
            args.model_dtype,
            data_loader_train,
            optimizer, 
            optimizer_disc,
            device, 
            epoch, 
            loss_scaler,
            loss_scaler_disc,
            args.clip_grad, 
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            lr_schedule_values_disc=lr_schedule_values_disc,
            args=args,
            print_freq=args.print_freq,
            iters_per_epoch=num_training_steps_per_epoch,
        )

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq, optimizer_disc=optimizer_disc
                )
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': epoch, 'n_parameters': n_learnable_parameters}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
