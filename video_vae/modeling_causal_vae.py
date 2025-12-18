from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)

from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin

from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from .modeling_enc_dec import (
    DecoderOutput, DiagonalGaussianDistribution, 
    CausalVaeDecoder, CausalVaeEncoder,
)
from .modeling_causal_conv import CausalConv3d




class CausalVideoVAE(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # encoder related parameters
        encoder_in_channels: int = 3,
        encoder_out_channels: int = 4,
        encoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 2),
        encoder_down_block_types: Tuple[str, ...] = (
            "DownEncoderBlockCausal3D",
            "DownEncoderBlockCausal3D",
            "DownEncoderBlockCausal3D",
            "DownEncoderBlockCausal3D",
        ),
        encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        encoder_spatial_down_sample: Tuple[bool, ...] = (True, True, True, False),
        encoder_temporal_down_sample: Tuple[bool, ...] = (True, True, True, False),
        encoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        encoder_act_fn: str = "silu",
        encoder_norm_num_groups: int = 32,
        encoder_double_z: bool = True,
        encoder_type: str = 'causal_vae_conv',
        # decoder related
        decoder_in_channels: int = 4,
        decoder_out_channels: int = 3,
        decoder_layers_per_block: Tuple[int, ...] = (3, 3, 3, 3),
        decoder_up_block_types: Tuple[str, ...] = (
            "UpDecoderBlockCausal3D",
            "UpDecoderBlockCausal3D",
            "UpDecoderBlockCausal3D",
            "UpDecoderBlockCausal3D",
        ),
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        decoder_spatial_up_sample: Tuple[bool, ...] = (True, True, True, False),
        decoder_temporal_up_sample: Tuple[bool, ...] = (True, True, True, False),
        decoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        decoder_act_fn: str = "silu",
        decoder_norm_num_groups: int = 32,
        decoder_type: str = 'causal_vae_conv',
        sample_size: int = 256,
        scaling_factor: float = 0.18215,
        add_post_quant_conv: bool = True,
        interpolate: bool = False,
        downsample_scale: int = 8,
    ):
        super().__init__()

        print(f"The latent dimmension channes is {encoder_out_channels}")
        # pass init params to Encoder

        self.encoder = CausalVaeEncoder(
            in_channels=encoder_in_channels,
            out_channels=encoder_out_channels,
            down_block_types=encoder_down_block_types,
            spatial_down_sample=encoder_spatial_down_sample,
            temporal_down_sample=encoder_temporal_down_sample,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            act_fn=encoder_act_fn,
            norm_num_groups=encoder_norm_num_groups,
            double_z=True,
            block_dropout=encoder_block_dropout,
        )

        # pass init params to Decoder
        self.decoder = CausalVaeDecoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            up_block_types=decoder_up_block_types,
            spatial_up_sample=decoder_spatial_up_sample,
            temporal_up_sample=decoder_temporal_up_sample,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            norm_num_groups=decoder_norm_num_groups,
            act_fn=decoder_act_fn,
            interpolate=interpolate,
            block_dropout=decoder_block_dropout,
        )

        self.quant_conv = CausalConv3d(2 * encoder_out_channels, 2 * encoder_out_channels, kernel_size=1, stride=1)
        self.post_quant_conv = CausalConv3d(encoder_out_channels, encoder_out_channels, kernel_size=1, stride=1)
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size

        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / downsample_scale) 
        self.encode_tile_overlap_factor = 1 / 4
        self.decode_tile_overlap_factor = 1 / 4
        self.downsample_scale = downsample_scale

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (self.encoder, self.decoder)):
            module.gradient_checkpointing = value

    def get_last_layer(self):
        return self.decoder.conv_out.conv.weight


    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
        freeze_encoder: bool = False,
        is_init_image=True, 
        temporal_chunk=False,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample

        # with torch.no_grad():
        h = self.encoder(x, is_init_image=True, temporal_chunk=False)
        moments = self.quant_conv(h, is_init_image=True, temporal_chunk=False)
        posterior = DiagonalGaussianDistribution(moments)
            
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        
        dec = self.decode(z, is_init_image=True).sample
            
        return posterior, dec
    

    def decode(self, z: torch.FloatTensor, is_init_image=True, temporal_chunk=False, 
            return_dict: bool = True, window_size: int = 2, tile_sample_min_size: int = 256,) -> Union[DecoderOutput, torch.FloatTensor]:
        
        z = self.post_quant_conv(z, is_init_image=is_init_image, temporal_chunk=False)
        dec = self.decoder(z, is_init_image=is_init_image, temporal_chunk=False)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
        



