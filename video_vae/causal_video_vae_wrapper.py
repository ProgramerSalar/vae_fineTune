import torch
import os
import torch.nn as nn
from collections import OrderedDict
from .modeling_causal_vae import CausalVideoVAE
from .modeling_loss import LPIPSWithDiscriminator
from einops import rearrange
from PIL import Image
from IPython import embed




class CausalVideoVAELossWrapper(nn.Module):
    """
        The causal video vae training and inference running wrapper
    """
    def __init__(self, model_path, model_dtype='fp32', disc_start=0, logvar_init=0.0, kl_weight=1.0, 
        pixelloss_weight=1.0, perceptual_weight=1.0, disc_weight=0.5, interpolate=True, 
        add_discriminator=True, freeze_encoder=False, load_loss_module=False, lpips_ckpt=None, **kwargs,
    ):
        super().__init__()

        if model_dtype == 'bf16':
            torch_dtype = torch.bfloat16
        elif model_dtype == 'fp16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.vae = CausalVideoVAE.from_pretrained(model_path, torch_dtype=torch_dtype, interpolate=False)
        self.vae_scale_factor = self.vae.config.scaling_factor

        if freeze_encoder:
            print("Freeze the parameters of vae encoder")
            for parameter in self.vae.encoder.parameters():
                parameter.requires_grad = False
            for parameter in self.vae.quant_conv.parameters():
                parameter.requires_grad = False

        self.add_discriminator = add_discriminator
        self.freeze_encoder = freeze_encoder

        # Used for training
        if load_loss_module:
            self.loss = LPIPSWithDiscriminator(disc_start, logvar_init=logvar_init, kl_weight=kl_weight,
                pixelloss_weight=pixelloss_weight, perceptual_weight=perceptual_weight, disc_weight=disc_weight, 
                add_discriminator=add_discriminator, using_3d_discriminator=False, disc_num_layers=4, lpips_ckpt=lpips_ckpt)
        else:
            self.loss = None

        self.disc_start = disc_start

    def load_checkpoint(self, checkpoint_path, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']

        vae_checkpoint = OrderedDict()
        disc_checkpoint = OrderedDict()

        for key in checkpoint.keys():
            if key.startswith('vae.'):
                new_key = key.split('.')
                new_key = '.'.join(new_key[1:])
                vae_checkpoint[new_key] = checkpoint[key]
            if key.startswith('loss.discriminator'):
                new_key = key.split('.')
                new_key = '.'.join(new_key[2:])
                disc_checkpoint[new_key] = checkpoint[key]

        vae_ckpt_load_result = self.vae.load_state_dict(vae_checkpoint, strict=False)
        print(f"Load vae checkpoint from {checkpoint_path}, load result: {vae_ckpt_load_result}")

        if self.add_discriminator:
            disc_ckpt_load_result = self.loss.discriminator.load_state_dict(disc_checkpoint, strict=False)
            print(f"Load disc checkpoint from {checkpoint_path}, load result: {disc_ckpt_load_result}")

    def forward(self, x, step, identifier=['video']):
        xdim = x.ndim
        if xdim == 4:
            x = x.unsqueeze(2)   #  (B, C, H, W) -> (B, C, 1, H , W)

        if 'video' in identifier:
            # The input is video
            assert 'image' not in identifier
        else:
            # The input is image
            assert 'video' not in identifier
            # We arrange multiple images to a 5D Tensor for compatibility with video input
            # So we needs to reformulate images into 1-frame video tensor 
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = x.unsqueeze(2)  # [(b t) c 1 h w]

        
        batch_x = x

        posterior, reconstruct = self.vae(batch_x, freeze_encoder=self.freeze_encoder, 
                    is_init_image=True, temporal_chunk=False,)

        # The reconstruct loss
        reconstruct_loss, rec_log = self.loss(
            batch_x, reconstruct, posterior, 
            optimizer_idx=0, global_step=step, last_layer=self.vae.get_last_layer(),
        )

        if step < self.disc_start:
            return reconstruct_loss, None, rec_log

        # The loss to train the discriminator
        gan_loss, gan_log = self.loss(batch_x, reconstruct, posterior, optimizer_idx=1, 
            global_step=step, last_layer=self.vae.get_last_layer(),
        )

        loss_log = {**rec_log, **gan_log}

        return reconstruct_loss, gan_loss, loss_log

    

    

    

    
    
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype