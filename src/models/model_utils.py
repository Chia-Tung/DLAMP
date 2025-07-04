from importlib import import_module
from math import ceil, floor

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR


def get_builder(model_name: str):
    return getattr(import_module("..", __name__), "{}Builder".format(model_name))


def window_partition_3d(
    input_feature: torch.Tensor,
    window_size: tuple[int, int, int],
    combine_img_dim: bool = False,
) -> torch.Tensor:
    """
    Partitions the given input into windows.

    Args:
        input_feature (tensor): (B, Z, H, W, C)
        window_size (tuple[int, int, int]): attention window's shape (wZ, wH, wW)
        combine_img_dim (bool): whether to combine the image dimensions into one channel
    Returns:
        torch.Tensor: Tensor of shape (B * num_windows, wZ, wH, wW, C) or (B * num_windows, wZ*wH*wW, C)
            if `combine_img_dim` is True
    """
    arg = "(b nZ nH nW) (wZ wH wW) c" if combine_img_dim else "(b nZ nH nW) wZ wH wW c"
    windows = rearrange(
        input_feature,
        "b (nZ wZ) (nH wH) (nW wW) c -> " + arg,
        wZ=window_size[0],
        wH=window_size[1],
        wW=window_size[2],
    )
    return windows


def window_reverse_3d(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    orig_img_size: tuple[int, int, int],
    from_combine_dim: bool = False,
) -> torch.Tensor:
    """
    Merges windows to produce higher resolution features.

    Args:
        windows (torch.Tensor): Tensor of shape (B * num_windows, wZ, wH, wW, C) or (B * num_windows, wZ*wH*wW, C)
            if `from_combined_img_dim` is True
        window_size (tuple[int, int, int]): window size (wZ, wH, wW)
        orig_img_size (tuple[int, int, int]): original image size (Z, H, W)
        from_combine_dim (bool): whether the input tensor is the same shape as `combine_img_dim=True` from
            `window_partition_3d`
    Returns:
        torch.Tensor: Tensor of shape (B, Z, H, W, C)
    """
    arg = "(b nZ nH nW) (wZ wH wW) c" if from_combine_dim else "(b nZ nH nW) wZ wH wW c"
    nZ, nH, nW = map(lambda x, y: x // y, orig_img_size, window_size)
    orig_img = rearrange(
        windows,
        arg + " -> b (nZ wZ) (nH wH) (nW wW) c",
        nZ=nZ,
        nH=nH,
        nW=nW,
        wZ=window_size[0],
        wH=window_size[1],
        wW=window_size[2],
    )
    return orig_img


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


def is_divisible_elementwise(list1: list[int], list2: list[int]) -> bool:
    """
    Check if each element in list1 is divisible by the corresponding element in list2.

    Args:
        list1 (list[int]): List of integers.
        list2 (list[int]): List of integers.

    Returns:
        bool: True if all elements are divisible element-wise, False otherwise.
    """
    assert len(list1) == len(list2)
    return all(x % y == 0 for x, y in zip(list1, list2))


def get_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    schedule_type: str,
    training_steps: int,
    warmup_steps: int | None = None,
    warmup_epochs: int | None = None,
    cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    def cosine_decay(current_step):
        # Warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # decadence
        progress = (current_step - warmup_steps) / max(1, training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(cycles) * 2.0 * progress)))

    def constant(current_step):
        return 1.0

    def constant_warmup(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        return 1.0

    def linear_decay(current_epoch):
        if current_epoch <= warmup_epochs:
            lr_scale = 1.0
        else:
            overflow = current_epoch - warmup_epochs
            lr_scale = 0.995**overflow
            if lr_scale < 0.05:
                lr_scale = 0.05
        return lr_scale

    match schedule_type:
        case "cosine":
            return LambdaLR(optimizer, cosine_decay, last_epoch)
        case "constant":
            return LambdaLR(optimizer, constant, last_epoch)
        case "constant_warmup":
            return LambdaLR(optimizer, constant_warmup, last_epoch)
        case "linear_decay":
            return LambdaLR(optimizer, linear_decay, last_epoch)
        case _:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")


class RunningAverage:
    def __init__(self):
        self._N = 0
        self._sum = 0

    def get(self):
        if self._N > 0:
            return self._sum / self._N

        return None

    def reset(self):
        self._N = 0
        self._sum = 0

    def add(self, val, n):
        self._N += n
        self._sum += val


def restruct_dimension(x_upper, x_surface, is_numpy=False, device=None):
    """
    Args:
        x_upper (torch.Tensor): Tensor of shape (B, Lv, H, W, C1)
        x_surface (torch.Tensor): Tensor of shape (B, 1, H, W, C2)
        is_numpy (bool, optional): Whether the input is in numpy format
        device (torch.device, optional): Device of the output tensor

    Returns:
        torch.Tensor: Tensor of shape (B, Lv*C1+C2, H, W)
    """
    if is_numpy and device is not None:
        x_upper = torch.from_numpy(x_upper).to(device)
        x_surface = torch.from_numpy(x_surface).to(device)
    elif is_numpy and device is None:
        raise ValueError("If `is_numpy` is True, `device` must be provided.")

    x_upper = rearrange(x_upper, "b z h w c -> b (z c) h w")
    x_surface = rearrange(x_surface, "b 1 h w c -> b c h w")
    x = torch.cat([x_upper, x_surface], dim=1)
    return x


def deconstruct(
    x: torch.Tensor, upper_ch: int, surface_ch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Deconstructs a tensor `x` into two tensors `x_upper` and `x_surface`.

    Args:
        x (torch.Tensor): The input tensor of shape (B, Lv*C1+C2, H, W).
        upper_ch (int): The number of channels in the upper layer (C1).
        surface_ch (int): The number of channels in the surface layer (C2).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing `x_upper` and `x_surface`.
            - `x_upper` (torch.Tensor): The upper layer tensor of shape (B, Lv, H, W, C1).
            - `x_surface` (torch.Tensor): The surface layer tensor of shape (B, 1, H, W, C2).
    """
    x_upper, x_surface = x[:, :-surface_ch], x[:, -surface_ch:]
    x_upper = rearrange(x_upper, "b (z c) h w -> b z h w c", c=upper_ch)
    x_surface = rearrange(x_surface, "b c h w -> b 1 h w c")
    return x_upper, x_surface
