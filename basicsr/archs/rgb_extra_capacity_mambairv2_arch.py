import torch
from torch import nn

from basicsr.archs.mambairv2_arch import MambaIRv2
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class RGBExtraCapacityMambaIRv2(MambaIRv2):
    """Stage-1 B0+: RGB-only capacity control exactly matched to B1.

    B0+ change note: a parameter-free RGB-to-luma projection feeds an extra
    1-channel RGB branch. Its 3x3 convolution and the following 1x1 fusion have
    exactly the same shapes as B1's depth branch and RGB-depth fusion. Thus B0+
    and B1 have identical trainable parameter counts and learned convolution
    MACs, while B0+ receives no information outside the LR RGB image.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rgb_in_chans = int(kwargs.get('in_chans', 3))
        if rgb_in_chans != 3:
            raise ValueError('Stage-1 B0+ requires a three-channel RGB input.')

        # B0+ change: these two layers deliberately mirror B1 depth_conv and
        # rgb_depth_fusion in shape, parameter count, and initialization order.
        self.extra_rgb_conv = nn.Conv2d(1, self.embed_dim, 3, 1, 1)
        self.rgb_capacity_fusion = nn.Conv2d(
            self.embed_dim * 2, self.embed_dim, 1, 1)

    @staticmethod
    def _mirror_pad_to_window(x, target_h, target_w):
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :target_h, :]
        return torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :target_w]

    @staticmethod
    def _rgb_to_luma(rgb):
        # B0+ change: fixed BT.601-style coefficients add no trainable
        # parameters and ensure the extra branch contains RGB-derived data only.
        return (
            rgb[:, 0:1] * 0.2989
            + rgb[:, 1:2] * 0.5870
            + rgb[:, 2:3] * 0.1140)

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(
                f'RGBExtraCapacityMambaIRv2 expects BCHW RGB input, got {tuple(rgb.shape)}.')

        h_ori, w_ori = rgb.shape[-2:]
        h = ((h_ori + self.window_size - 1) // self.window_size) * self.window_size
        w = ((w_ori + self.window_size - 1) // self.window_size) * self.window_size
        rgb = self._mirror_pad_to_window(rgb, h, w)

        # Match B1 input scaling: the main RGB path uses the original MambaIRv2
        # normalization, whereas the 1-channel auxiliary path remains in [0, 1].
        extra_rgb = self._rgb_to_luma(rgb)
        self.mean = self.mean.type_as(rgb)
        normalized_rgb = (rgb - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(rgb.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        rgb_feature = self.conv_first(normalized_rgb)
        extra_feature = self.extra_rgb_conv(extra_rgb.type_as(normalized_rgb))
        fused = self.rgb_capacity_fusion(
            torch.cat([rgb_feature, extra_feature], dim=1))
        body = self.conv_after_body(self.forward_features(fused, params)) + fused

        if self.upsampler == 'pixelshuffle':
            output = self.conv_last(self.upsample(self.conv_before_upsample(body)))
        elif self.upsampler == 'pixelshuffledirect':
            output = self.upsample(body)
        elif self.upsampler == 'nearest+conv':
            output = self.conv_before_upsample(body)
            output = self.lrelu(self.conv_up1(
                torch.nn.functional.interpolate(output, scale_factor=2, mode='nearest')))
            output = self.lrelu(self.conv_up2(
                torch.nn.functional.interpolate(output, scale_factor=2, mode='nearest')))
            output = self.conv_last(self.lrelu(self.conv_hr(output)))
        else:
            output = normalized_rgb + self.conv_last(body)

        output = output / self.img_range + self.mean
        return output[..., :h_ori * self.upscale, :w_ori * self.upscale]
