import logging
from math import ceil

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

from ..model_utils import is_divisible_elementwise, pad_2d, pad_3d
from .earth_3d_specifics import EarthSpecificLayer

__all__ = ["PanguModel"]

log = logging.getLogger(__name__)


class PanguModel(nn.Module):
    """
    Implementing https://github.com/198808xc/Pangu-Weather
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        patch_size: tuple[int, int, int],
        window_size: tuple[int, int, int],
        upper_levels: int,
        upper_channels: int,
        surface_channels: int,
        embed_dim: int,
        heads: list[int],
        depths: list[int],
        max_drop_path_ratio: float,
        dropout_rate: float,
        smoothing_kernel_size: int | None = None,
        const_mask_paths: list[str] | None = None,
    ) -> None:
        assert len(depths) == 2  # only two layers allowed
        assert len(heads) == len(depths)
        super().__init__()

        drop_path_list = np.linspace(0, max_drop_path_ratio, sum(depths)).tolist()
        image_shape = [upper_levels] + image_shape
        inp_Z, inp_H, inp_W = [ceil(x / y) for x, y in zip(image_shape, patch_size)]
        inp_Z += 1  # surface

        # TODO: smoothing layer
        if smoothing_kernel_size is None:
            self.smoothing_layer = Identity()

        self.patch_embed = PatchEmbedding(
            img_shape=image_shape,
            patch_shape=patch_size,
            upper_channels=upper_channels,
            surface_channels=surface_channels,
            dim=embed_dim,
            const_mask_paths=const_mask_paths,
        )

        self.layer1 = EarthSpecificLayer(
            input_shape=(inp_Z, inp_H, inp_W),
            dim=embed_dim,
            heads=heads[0],
            depth=depths[0],
            drop_path_ratio_list=drop_path_list[: depths[0]],
            dropout_rate=dropout_rate,
            window_size=window_size,
        )

        # self.downsample =


class Identity(nn.Module):
    """
    Identity layer. Replace smoothing layer when smoothing_kernel_size is None.
    """

    def forward(
        self, x_upper: torch.Tensor, x_surface: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x_upper, x_surface


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_shape: tuple[int, int, int],
        patch_shape: tuple[int, int, int],
        upper_channels: int,
        surface_channels: int,
        dim: int,
        constant_mask_paths: list[str] | None,
    ) -> None:
        """
        Convert input fields to patches and linearly embed them.

        Args:
            img_shape (tuple[int, int, int]): The shape of unpatched data which is (img_Z, img_H, img_W).
            patch_shape (tuple[int, int, int]): Size of the patch (pat_Z, pat_H, pat_W).
            upper_channels (int): The number of channels in the upper_air data.
            surface_channels (int): The number of channels in the surface data.
            dim (int): The dimension of the embedded output.
            constant_mask_paths (list[str] | None): The paths to constant masks.

        Raises:
            NotImplementedError: If constant masks are provided.

        Returns:
            None
        """
        super().__init__()

        # TODO: constant mask
        if constant_mask_paths is not None:
            raise NotImplementedError("Constant mask is not implemented yet.")
        else:
            land_mask, soil_mask, topography_mask = None, None, None
            additional_channels = 0
        self.register_buffer("land_mask", land_mask)
        self.register_buffer("soil_mask", soil_mask)
        self.register_buffer("topography_mask", topography_mask)

        # Use convolution to partition data into cubes
        self.patch_shape = patch_shape
        self.conv_upper = nn.Conv3d(
            in_channels=upper_channels,
            out_channels=dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )

        self.conv_surface = nn.Conv2d(
            in_channels=surface_channels + additional_channels,
            out_channels=dim,
            kernel_size=patch_shape[1:],
            stride=patch_shape[1:],
        )

        if not is_divisible_elementwise(img_shape, patch_shape):
            log.warning(
                f"Input shape {img_shape} is not divisible by patch "
                f"shape {patch_shape}, padding is applied."
            )
        self.upper_pad = pad_3d(img_shape, patch_shape)
        self.surface_pad = pad_2d(img_shape[1:], patch_shape[1:])

    def forward(
        self, input_upper: torch.Tensor, input_surface: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_upper (torch.Tensor): Tensor of shape (B, img_Z, img_H, img_W, Ch_upper).
            input_surface (torch.Tensor): Tensor of shape (B, 1, img_H, img_W, Ch_surface).
        Returns:
            torch.Tensor: Tensor of shape (B, inp_Z*inp_H*inp_W, dim).
        """
        if (
            self.land_mask is not None
            and self.soil_mask is not None
            and self.topography_mask is not None
        ):
            batch_size = input_surface.shape[0]
            input_surface = torch.cat(
                [
                    input_surface,
                    repeat(self.land_mask, "h w -> b 1 h w 1", b=batch_size),
                    repeat(self.soil_mask, "h w -> b 1 h w 1", b=batch_size),
                    repeat(self.topography_mask, "h w -> b 1 h w 1", b=batch_size),
                ],
                dim=-1,
            )

        # Pad the input to make it divisible by patch_shape
        input_upper = self.upper_pad(rearrange(input_upper, "b z h w c -> b c z h w"))
        input_surface = self.surface_pad(
            rearrange(input_surface, "b z h w c -> (b z) c h w")
        )

        # shape: (B, dim, inp_Z-1, inp_H, inp_W)
        embedding_upper = self.conv_upper(input_upper)
        # shape: (B, dim, inp_H, inp_W)
        embedding_surface = self.conv_surface(input_surface)
        # shape: (B, dim, inp_Z, inp_H, inp_W)
        x = torch.cat([embedding_upper, embedding_surface[:, :, None]], dim=2)
        # shape: (B, inp_Z*inp_H*inp_W, dim)
        x = rearrange(x, "b c z h w -> b (z h w) c")

        return x


class DownSample(nn.Module):
    def __init__(self, inp_shape: tuple[int, int, int], inp_channels: int, oup_channels: int):
        """
        Implementation of `SwinPatchMerging`. Reduces the lateral resolution by a factor of 2.

        Args:
            inp_shape (tuple[int, int, int]): Shape of the input tensor (inp_Z, inp_H, inp_W).
            inp_channels (int): Number of input channels.
            oup_channels (int): Number of output channels.
        """
        super().__init__()
        self.linear = nn.Linear(4*inp_channels, oup_channels, bias=False)
        self.norm = nn.LayerNorm(4*inp_channels)
        self.Z, self.H, self.W = inp_shape
        
        # Pad to make H and W divisible by 2
        self.pad = pad_2d((self.H, self.W), (2, 2))
