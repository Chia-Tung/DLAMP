from importlib import import_module
from math import ceil, floor

import torch
import torch.nn as nn
from einops import rearrange


def get_builder(model_name: str):
    return getattr(import_module("..", __name__), "{}Builder".format(model_name))


def window_partition_3d(
    input_feature: torch.Tensor, window_size: tuple[int, int, int]
) -> torch.Tensor:
    """
    Partitions the given input into windows.

    Args:
        input_feature (tensor): (B, Z, H, W, C)
        window_size (tuple[int, int, int]): attention window's shape (wZ, wH, wW)
    Returns:
        torch.Tensor: Tensor of shape (B * num_windows, wZ, wH, wW, C)
    """
    wZ, wH, wW = window_size
    windows = rearrange(
        input_feature,
        "b (nZ wZ) (nH wH) (nW wW) c -> (b nZ nH nW) wZ wH wW c",
        wZ=wZ,
        wH=wH,
        wW=wW,
    )
    return windows


def window_reverse_3d(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    orig_img_size: tuple[int, int, int],
) -> torch.Tensor:
    """
    Merges windows to produce higher resolution features.

    Args:
        windows (torch.Tensor): Tensor of shape (B * num_windows, wZ, wH, wW, C)
        window_size (tuple[int, int, int]): window size (wZ, wH, wW)
        orig_img_size (tuple[int, int, int]): original image size (Z, H, W)
    Returns:
        torch.Tensor: Tensor of shape (B, Z, H, W, C)
    """
    Z, H, W = orig_img_size
    wZ, wH, wW = window_size
    nZ, nH, nW = map(lambda x, y: x // y, orig_img_size, window_size)
    orig_img = rearrange(
        windows,
        "(b nZ nH nW) wZ wH wW c -> b (nZ wZ) (nH wH) (nW wW) c",
        nZ=nZ,
        nH=nH,
        nW=nW,
    )
    return orig_img


def gen_3d_attn_mask(
    img_shape: tuple[int, int, int],
    window_size: tuple[int, int, int],
) -> torch.Tensor:
    """
    Generates a 3D attention mask tensor based on the given image shape and window size.

    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.

    Args:
        img_shape (tuple[int, int, int]): The shape of the image tensor.
        window_size (tuple[int, int, int]): The size of the sliding window.

    Returns:
        torch.Tensor: The 3D attention mask tensor.
    """
    shift_size = tuple(i // 2 for i in window_size)
    img_mask = torch.zeros(1, img_shape[0], img_shape[1], img_shape[2], 1)
    z_slices = (
        slice(0, -window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    )
    h_slices = (
        slice(0, -window_size[1]),
        slice(-window_size[1], -shift_size[1]),
        slice(-shift_size[1], None),
    )
    w_slices = (
        slice(0, -window_size[2]),
        slice(-window_size[2], -shift_size[2]),
        slice(-shift_size[2], None),
    )
    cnt = 0
    for z in z_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, z, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition_3d(img_mask, window_size)
    mask_windows = mask_windows.reshape(
        -1, window_size[0] * window_size[1] * window_size[2]
    )
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def pad_3d(
    img_shape: tuple[int, int, int], sub_shape: tuple[int, int, int]
) -> nn.ZeroPad3d:
    """
    Get nn.ZeroPad3d for padding the input to be divisible by sub_shape.

    Args:
        img_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
        sub_shape (tuple[int, int, int]): Shape of the sub tensor (z, h, w).
    Returns:
        nn.ZeroPad3d: Padding layer.
    """
    Z, H, W = img_shape
    z, h, w = sub_shape
    pad = nn.ZeroPad3d(
        (
            floor((-W % w) / 2),
            ceil((-W % w) / 2),
            floor((-H % h) / 2),
            ceil((-H % h) / 2),
            floor((-Z % z) / 2),
            ceil((-Z % z) / 2),
        )
    )
    return pad


def pad_2d(img_shape: tuple[int, int], sub_shape: tuple[int, int]) -> nn.ZeroPad2d:
    """
    Get nn.ZeroPad2d for padding the input to be divisible by sub_shape.

    Args:
        img_shape (tuple[int, int]): Shape of the input tensor (H, W).
        sub_shape (tuple[int, int]): Shape of the sub tensor (h, w).
    Returns:
        nn.ZeroPad2d: Padding layer.
    """
    H, W = img_shape
    h, w = sub_shape
    pad = nn.ZeroPad2d(
        (
            floor((-W % w) / 2),
            ceil((-W % w) / 2),
            floor((-H % h) / 2),
            ceil((-H % h) / 2),
        )
    )
    return pad


def crop_pad_3d(
    img_shape: tuple[int, int, int], sub_shape: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    """
    Get the index for reversing padding via GetPad3D.

    Args:
        img_shape (tuple[int, int, int]): Shape of the input tensor (Z, H, W).
        sub_shape (tuple[int, int, int]): Shape of the sub tensor (z, h, w).
    Returns:
        tuple[slice, slice, slice]: Crop index.
    """
    Z, H, W = img_shape
    z, h, w = sub_shape
    return (
        slice(floor((-Z % z) / 2), -ceil((-Z % z) / 2)) if Z % z != 0 else slice(None),
        slice(floor((-H % h) / 2), -ceil((-H % h) / 2)) if H % h != 0 else slice(None),
        slice(floor((-W % w) / 2), -ceil((-W % w) / 2)) if W % w != 0 else slice(None),
    )


def crop_pad_2d(
    img_shape: tuple[int, int], sub_shape: tuple[int, int]
) -> tuple[slice, slice]:
    """
    Get the index for reversing padding via GetPad2D.

    Args:
        img_shape (tuple[int, int]): Shape of the input tensor (H, W).
        sub_shape (tuple[int, int]): Shape of the sub tensor (h, w).
    Returns:
        tuple[slice, slice]: Crop index.
    """
    H, W = img_shape
    h, w = sub_shape
    return (
        slice(floor((-H % h) / 2), -ceil((-H % h) / 2)) if H % h != 0 else slice(None),
        slice(floor((-W % w) / 2), -ceil((-W % w) / 2)) if W % w != 0 else slice(None),
    )
