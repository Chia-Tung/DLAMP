import torch
import torch.nn as nn
from einops import rearrange, repeat

from .earth_attention_3d import EarthAttention3D

__all__ = ["PanguModel"]


class PanguModel(nn.Module):
    """
    Implementing https://github.com/198808xc/Pangu-Weather
    """
