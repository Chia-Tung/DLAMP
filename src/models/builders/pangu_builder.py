import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from lightning import LightningModule
from torchvision.transforms.v2 import CenterCrop, Resize, ToDtype

from ...const import CHECKPOINT_DIR
from ...utils import DataCompose
from .. import PanguModel
from ..lightning_modules import PanguLightningModule
from .base_builder import BaseBuilder

__all__ = ["PanguBuilder"]


class PanguBuilder(BaseBuilder):
    def __init__(self, data_list: list[DataCompose], **kwargs):
        super().__init__(**kwargs)

        self.pressure_levels: list[str] = DataCompose.get_all_levels(
            data_list, only_upper=True, to_str=True
        )
        self.upper_vars: list[str] = DataCompose.get_all_vars(
            data_list, only_upper=True, to_str=True
        )
        self.surface_vars: list[str] = DataCompose.get_all_vars(
            data_list, only_surface=True, to_str=True
        )

        self.info_log(f"Input Image Shape: {self.kwargs.image_shape}")
        self.info_log(f"Patch Size: {self.kwargs.patch_size}")
        self.info_log(f"Window Size: {self.kwargs.window_size}")

    def _preprocess_layer(self) -> nn.Module:
        return nn.Sequential(
            Rearrange("b z h w c -> b c z h w"),
            CenterCrop([2 * x for x in self.kwargs.image_shape]),
            Resize(self.kwargs.image_shape),
            ToDtype(torch.float32),
            Rearrange("b c z h w -> b z h w c"),
        )

    def _backbone_model(self) -> nn.Module:
        return PanguModel(
            image_shape=self.kwargs.image_shape,
            patch_size=self.kwargs.patch_size,
            window_size=self.kwargs.window_size,
            upper_levels=len(self.pressure_levels),
            upper_channels=len(self.upper_vars),
            surface_channels=len(self.surface_vars),
            embed_dim=self.kwargs.embed_dim,
            heads=self.kwargs.heads,
            depths=self.kwargs.depths,
            max_drop_path_ratio=self.kwargs.max_drop_path_ratio,
            dropout_rate=self.kwargs.dropout_rate,
            smoothing_kernel_size=self.kwargs.smoothing_kernel_size,
            const_mask_paths=self.kwargs.const_mask_paths,
        )

    def build(self) -> LightningModule:
        return PanguLightningModule(
            checkpoint_dir=CHECKPOINT_DIR,
            preprocess_layer=self._preprocess_layer(),
            backbone_model=self._backbone_model(),
            upper_var_weights=None,
            surface_var_weights=None,
            surface_alpha=self.kwargs.surface_alpha,
            pressure_levels=self.pressure_levels,
            upper_vars=self.upper_vars,
            surface_vars=self.surface_vars,
            optim_config=self.kwargs.optim_config,
            lr_schedule=self.kwargs.lr_schedule,
        )
