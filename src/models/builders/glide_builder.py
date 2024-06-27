from pathlib import Path

import onnxruntime as ort
import torch.nn as nn
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from ...const import CHECKPOINT_DIR
from ...utils import DataCompose, convert_hydra_dir_to_timestamp
from ..architectures import GlideUNet
from ..lightning_modules import DiffusionLightningModule
from .base_builder import BaseBuilder

__all__ = ["GlideBuilder"]


class GlideBuilder(BaseBuilder):
    def __init__(self, hydra_dir: Path, data_list: list[DataCompose], **kwargs):
        super().__init__(**kwargs)

        self.time_stamp = convert_hydra_dir_to_timestamp(hydra_dir)
        self.input_channels = len(data_list)

        self.info_log(f"Input Image Shape: {self.kwargs.image_shape}")
        self.info_log(f"Glide Unet Layers: {len(self.kwargs.ch_mults)}")

    def _backbone_model(self) -> nn.Module:
        return GlideUNet(
            input_channels=self.input_channels,
            hidden_dim=self.kwargs.hidden_dim,
            ch_mults=self.kwargs.ch_mults,
            is_attn=self.kwargs.is_attn,
            attn_num_heads=self.kwargs.attn_num_heads,
        )

    def _regression_model(self) -> ort.InferenceSession:
        assert "CUDAExecutionProvider" in ort.get_available_providers()

        # An issue about onnxruntime for cuda12.x
        # ref: https://github.com/microsoft/onnxruntime/issues/8313#issuecomment-1486097717
        _default_session_options = ort.capi._pybind_state.get_default_session_options()

        def get_default_session_options_new():
            _default_session_options.inter_op_num_threads = 1
            _default_session_options.intra_op_num_threads = 1
            return _default_session_options

        ort.capi._pybind_state.get_default_session_options = (
            get_default_session_options_new
        )

        return ort.InferenceSession(
            self.kwargs.regression_onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def build_model(self, test_dataloader: DataLoader | None = None) -> LightningModule:
        return DiffusionLightningModule(
            test_dataloader=test_dataloader,
            backbone_model=self._backbone_model(),
            regression_model=self._regression_model(),
            timesteps=self.kwargs.timesteps,
            beta_start=self.kwargs.beta_start,
            beta_end=self.kwargs.beta_end,
            batch_size=self.kwargs.batch_size,
        )

    def build_trainer(self) -> Trainer:
        raise NotImplementedError

    def checkpoint_callback(self) -> ModelCheckpoint:
        return ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            filename=self.kwargs.model_name
            + f"_{self.time_stamp}"
            + "-{epoch:03d}-{val_loss_epoch:.4f}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss_epoch",
            mode="min",
        )

    def wandb_logger(self, save_dir: str = "./logs") -> WandbLogger:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        return WandbLogger(
            save_dir=save_dir,
            log_model=False,  # log W&B artifacts
            project="my-burdensome-project",
            name=self.kwargs.model_name + f"_{self.time_stamp}",
        )
