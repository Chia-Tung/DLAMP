from omegaconf import OmegaConf

from .. import PanguModel
from .base_builder import BaseBuilder

__all__ = ["PanguBuilder"]


class PanguBuilder(BaseBuilder):
    def __init__(self, **kwargs):
        self.kwargs = OmegaConf.create(kwargs)

    def _preprocess_layer(self):
        pass

    def _backbone_model(self):
        pass
        # return PanguModel(
        #     data_spatial_shape=data_spatial_shape,
        #     upper_vars=len(upper_vars),
        #     surface_vars=len(surface_vars),
        #     depths=depths,
        #     heads=heads,
        #     embed_dim=embed_dim,
        #     patch_shape=patch_shape,
        #     window_size=window_shape,
        #     constant_mask_paths=constant_mask_paths,
        #     smoothing_kernel_size=smoothing_kernel_size,
        #     residual=residual,
        #     res_conn_after_smooth=res_conn_after_smooth,
        # )

    def build(self):
        pass
