import torch
import torch.nn as nn
from einops import rearrange, repeat

from .earth_3d_specifics import EarthAttention3D

__all__ = ["PanguModel"]


class PanguModel(nn.Module):
    """
    Implementing https://github.com/198808xc/Pangu-Weather
    """
