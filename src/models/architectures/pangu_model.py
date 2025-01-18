import logging
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from sklearn.preprocessing import MinMaxScaler

from ...const import LAND_SEA_MASK_PATH, TOPOGRAPHY_MASK_PATH
from ..model_utils import (
    crop_pad_2d,
    crop_pad_3d,
    is_divisible_elementwise,
    pad_2d,
    pad_3d,
)
from .earth_3d_specifics import EarthSpecificLayer
from .smoothing import SegmentedSmoothingV2

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
        surface_input_channels: int,
        surface_output_channels: int,
        embed_dim: int,
        heads: list[int],
        depths: list[int],
        max_drop_path_ratio: float,
        dropout_rate: float,
        smoothing_kernel_size: int | None = None,
        segmented_smooth_boundary_width: int | None = None,
    ) -> None:
        assert len(depths) == 2  # only two layers allowed
        assert len(heads) == len(depths)
        super().__init__()

        hierarchy = len(depths)
        image_shape = [upper_levels] + image_shape
        inp_Z, inp_H, inp_W = [ceil(x / y) for x, y in zip(image_shape, patch_size)]
        inp_Z += 1  # surface
        drop_path_list = np.linspace(0, max_drop_path_ratio, sum(depths)).tolist()
        embed_dim = [(2**i) * embed_dim for i in range(hierarchy)]

        if smoothing_kernel_size is None:
            self.smoothing_layer = Identity()
        else:
            assert smoothing_kernel_size % 2 == 1
            if segmented_smooth_boundary_width:
                smoothing_func = SegmentedSmoothingV2(
                    kernel_size=smoothing_kernel_size,
                    boundary_width=segmented_smooth_boundary_width,
                )
            else:
                smoothing_func = nn.AvgPool3d(
                    kernel_size=(1, smoothing_kernel_size, smoothing_kernel_size),
                    stride=(1, 1, 1),
                    padding=(0, smoothing_kernel_size // 2, smoothing_kernel_size // 2),
                    count_include_pad=False,
                )

            self.smoothing_layer = SmoothingBlock(smoothing_func=smoothing_func)

        # ===== Left Side of Unet =====#
        self.patch_embed = PatchEmbedding(
            img_shape=image_shape,
            patch_size=patch_size,
            upper_channels=upper_channels,
            surface_channels=surface_input_channels,
            dim=embed_dim[0],
        )

        for i, depth in enumerate(depths):
            slice_range = slice(sum(depths[:i]), sum(depths[: i + 1]))
            self.__setattr__(
                "layer{}".format(i + 1),
                EarthSpecificLayer(
                    input_shape=(inp_Z, ceil(inp_H / (2**i)), ceil(inp_W / (2**i))),
                    dim=embed_dim[i],
                    heads=heads[i],
                    depth=depth,
                    drop_path_ratio_list=drop_path_list[slice_range],
                    dropout_rate=dropout_rate,
                    window_size=window_size,
                ),
            )

        self.downsample = DownSample(inp_shape=(inp_Z, inp_H, inp_W), dim=embed_dim[0])

        # ===== Right Side of Unet =====#
        self.patch_recover = PatchRecovery(
            img_shape=image_shape,
            inp_shape=(inp_Z, inp_H, inp_W),
            patch_size=patch_size,
            upper_channels=upper_channels,
            surface_channels=surface_output_channels,
            dim=embed_dim[1],  # skip connection
        )

        for i, depth in enumerate(reversed(depths), start=1):
            j = hierarchy - i  # reversed index
            slice_range = slice(sum(depths[:j]), sum(depths[: j + 1]))
            self.__setattr__(
                "layer{}".format(i + hierarchy),
                EarthSpecificLayer(
                    input_shape=(inp_Z, ceil(inp_H / (2**j)), ceil(inp_W / (2**j))),
                    dim=embed_dim[j],
                    heads=heads[j],
                    depth=depth,
                    drop_path_ratio_list=drop_path_list[slice_range],
                    dropout_rate=dropout_rate,
                    window_size=window_size,
                ),
            )

        self.upsample = UpSample(oup_shape=(inp_Z, inp_H, inp_W), oup_dim=embed_dim[0])

    def forward(
        self, input_upper: torch.Tensor, input_surface: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unet structure.

        Args:
            input_upper (torch.Tensor): Tensor of shape (B, img_Z, img_H, img_W, Ch_upper).
            input_surface (torch.Tensor): Tensor of shape (B, 1, img_H, img_W, Ch_surface).
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of upper-air data and surface data.
        """
        # left side
        x = self.patch_embed(input_upper, input_surface)
        x = self.layer1(x)
        skip = x
        x = self.downsample(x)
        x = self.layer2(x)
        # right side
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)
        x = torch.cat([skip, x], dim=-1)
        output_upper, output_surface = self.patch_recover(x)
        output_upper, output_surface = self.smoothing_layer(
            output_upper, output_surface
        )
        return output_upper, output_surface


class Identity(nn.Module):
    """
    Identity layer. Replace smoothing layer when smoothing_kernel_size is None.
    """

    def forward(
        self, x_upper: torch.Tensor, x_surface: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x_upper, x_surface


class SmoothingBlock(nn.Module):
    def __init__(
        self,
        smoothing_func: nn.Module,
    ) -> None:
        """
        Smooths horizontal dimensions of the upper and surface data with average pooling
        """
        super().__init__()
        self.smoothing_func = smoothing_func

    def forward(
        self, x_upper: torch.Tensor, x_surface: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_upper = rearrange(x_upper, "b z h w c -> b c z h w")
        x_surface = rearrange(x_surface, "b 1 h w c -> b c 1 h w")

        x_upper = self.smoothing_func(x_upper)
        x_surface = self.smoothing_func(x_surface)

        x_upper = rearrange(x_upper, "b c z h w -> b z h w c")
        x_surface = rearrange(x_surface, "b c 1 h w -> b 1 h w c")
        return x_upper, x_surface


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        upper_channels: int,
        surface_channels: int,
        dim: int,
    ) -> None:
        """
        Convert input fields to patches and linearly embed them.

        Args:
            img_shape (tuple[int, int, int]): The shape of unpatched data which is (img_Z, img_H, img_W).
            patch_size (tuple[int, int, int]): Size of the patch (pat_Z, pat_H, pat_W).
            upper_channels (int): The number of channels in the upper_air data.
            surface_channels (int): The number of channels in the surface data.
            dim (int): The dimension of the embedded output.

        Raises:
            NotImplementedError: If constant masks are provided.

        Returns:
            None
        """
        super().__init__()

        if Path(LAND_SEA_MASK_PATH).exists() and Path(TOPOGRAPHY_MASK_PATH).exists():
            land_mask = torch.from_numpy(np.load(LAND_SEA_MASK_PATH).astype(np.float32))
            topography_mask = torch.from_numpy(
                np.load(TOPOGRAPHY_MASK_PATH).astype(np.float32)
            )
            # Scale and shift to the range of [0, 1]
            scaler = MinMaxScaler().fit(topography_mask.reshape(-1, 1))
            scale = scaler.scale_.astype(np.float32)
            min = scaler.min_.astype(np.float32)
            topography_mask = topography_mask * scale + min
            additional_channels = 2
        else:
            land_mask, topography_mask = None, None
            additional_channels = 0
        self.register_buffer("land_mask", land_mask)
        self.register_buffer("topography_mask", topography_mask)

        # Use convolution to partition data into cubes
        self.conv_upper = nn.Conv3d(
            in_channels=upper_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.conv_surface = nn.Conv2d(
            in_channels=surface_channels + additional_channels,
            out_channels=dim,
            kernel_size=patch_size[1:],
            stride=patch_size[1:],
        )

        if not is_divisible_elementwise(img_shape, patch_size):
            log.warning(
                f"Input shape {img_shape} is not divisible by patch "
                f"shape {patch_size}, padding is applied."
            )
        self.upper_pad = pad_3d(img_shape, patch_size)
        self.surface_pad = pad_2d(img_shape[1:], patch_size[1:])

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
        if self.land_mask is not None and self.topography_mask is not None:
            batch_size = input_surface.shape[0]
            input_surface = torch.cat(
                [
                    input_surface,
                    repeat(self.land_mask, "h w -> b 1 h w 1", b=batch_size),
                    repeat(self.topography_mask, "h w -> b 1 h w 1", b=batch_size),
                ],
                dim=-1,
            )

        # Pad the input to make it divisible by patch_size
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


class PatchRecovery(nn.Module):
    def __init__(
        self,
        img_shape: tuple[int, int, int],
        inp_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        upper_channels: int,
        surface_channels: int,
        dim: int,
    ) -> None:
        """
        Recover the output fields from patches.

        Args:
            img_shape (tuple[int, int, int]): The shape of the unpatched image (img_Z, img_H, img_W).
            inp_shape (tuple[int, int, int]): The shape of the patched input tensor (inp_Z, inp_H, inp_W).
            patch_size (tuple[int, int, int]): The size of the patches (patch_Z, patch_H, patch_W).
            upper_channels (int): The number of channels in the upper_air data.
            surface_channels (int): The number of channels in the surface data.
            dim (int): The dimension of the patched input tensor.

        Returns:
            None
        """
        super().__init__()
        # Hear we use two transposed convolutions to recover data
        self.conv_upper = nn.ConvTranspose3d(
            in_channels=dim,
            out_channels=upper_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.conv_surface = nn.ConvTranspose2d(
            in_channels=dim,
            out_channels=surface_channels,
            kernel_size=patch_size[1:],
            stride=patch_size[1:],
        )
        self.inp_shape = inp_shape
        self.upper_crop = crop_pad_3d(img_shape, patch_size)
        self.surface_crop = crop_pad_2d(img_shape[1:], patch_size[1:])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, inp_Z*inp_H*inp_W, dim).
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of upper-air data (B, img_Z, img_H, img_W, Ch_upper)
                and surface data (B, 1, img_H, img_W, Ch_surface)
        """
        inp_Z, inp_H, inp_W = self.inp_shape
        x = rearrange(
            x, "b (z h w) c -> b c z h w", z=inp_Z, h=inp_H, w=inp_W
        ).contiguous()

        # Deconvolve to original size
        output_upper = self.conv_upper(x[:, :, :-1, :, :])
        output_surface = self.conv_surface(x[:, :, -1, :, :])

        # Crop the output to remove zero-paddings
        output_upper = output_upper[
            :, :, self.upper_crop[0], self.upper_crop[1], self.upper_crop[2]
        ]
        output_surface = output_surface[
            :, :, self.surface_crop[0], self.surface_crop[1]
        ]

        # shape: (B, img_Z, img_H, img_W, Ch)
        output_upper = rearrange(output_upper, "b c z h w -> b z h w c")
        output_surface = rearrange(output_surface, "b c h w -> b 1 h w c")
        return output_upper, output_surface


class DownSample(nn.Module):
    def __init__(self, inp_shape: tuple[int, int, int], dim: int):
        """
        Implementation of `SwinPatchMerging`. Reduces the lateral resolution by a factor of 2.

        Args:
            inp_shape (tuple[int, int, int]): Shape of the input tensor (inp_Z, inp_H, inp_W).
            dim (int): Number of input channels.
        """
        super().__init__()
        self.inp_shape = inp_shape
        self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

        # Pad to make H and W divisible by 2
        self.pad = pad_2d((inp_shape[1], inp_shape[2]), (2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, inp_Z*inp_H*inp_W, dim).
        Returns:
            torch.Tensor: Tensor of shape (B, inp_Z*inp_H/2*inp_W/2, 2*dim).
        """
        x = rearrange(
            x,
            "b (z h w) c -> b c z h w",
            z=self.inp_shape[0],
            h=self.inp_shape[1],
            w=self.inp_shape[2],
        )
        x = self.pad(x)
        x = rearrange(
            x,
            "b c z (hh n1) (ww n2) -> b (z hh ww) (n1 n2 c)",
            n1=2,
            n2=2,
        )
        x = self.linear(self.norm(x))
        return x


class UpSample(nn.Module):
    def __init__(self, oup_shape: tuple[int, int, int], oup_dim: int):
        """
        Increases the lateral resolution by a factor of 2.

        Args:
            oup_shape (tuple[int, int, int]): Shape of the output tensor (inp_Z, inp_H, inp_W).
            oup_dim (int): Number of input channels AFTER the upsampling.
        """
        super().__init__()
        self.linear1 = nn.Linear(2 * oup_dim, 4 * oup_dim, bias=False)
        self.linear2 = nn.Linear(oup_dim, oup_dim, bias=False)
        self.norm = nn.LayerNorm(oup_dim)
        self.oup_shape = oup_shape

        # Rollback to the original resolution without padding
        self.crop = crop_pad_2d(oup_shape[1:], (2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, inp_Z*inp_H/2*inp_W/2, 2*dim).
        Returns:
            torch.Tensor: Tensor of shape (B, inp_Z*inp_H*inp_W, dim).
        """
        x = self.linear1(x)
        x = rearrange(
            x,
            "b (z hh ww) (n1 n2 c) -> b z (hh n1) (ww n2) c",
            hh=ceil(self.oup_shape[1] / 2),
            ww=ceil(self.oup_shape[2] / 2),
            n1=2,
            n2=2,
        )
        x = x[:, :, self.crop[0], self.crop[1], :]
        x = rearrange(x, "b z h w c -> b (z h w) c")
        x = self.linear2(self.norm(x))
        return x
