from pathlib import Path

import onnxruntime as ort
import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
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
            optim_config=self.kwargs.optim_config,
            lr_schedule=self.kwargs.lr_schedule,
        )

    def build_trainer(self, logger) -> Trainer:
        num_gpus = (
            torch.cuda.device_count()
            if self.kwargs.num_gpus is None
            else self.kwargs.num_gpus
        )

        callbacks = []
        callbacks.append(LearningRateMonitor())
        callbacks.append(self.checkpoint_callback())
        # if self.kwargs.log_image_every_n_steps is not None:
        #     callbacks.append(
        #         LogPredictionSamplesCallback(self.kwargs.log_image_every_n_steps)
        #     )
        if self.kwargs.early_stop_patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss_epoch", patience=self.kwargs.early_stop_patience
                )
            )

        return Trainer(
            num_sanity_val_steps=2,
            benchmark=True,
            fast_dev_run=self.kwargs.fast_dev_run,  # use n batch(es) to fast run through train/valid, no checkpoint, no max_epoch
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=self.kwargs.log_every_n_steps,  # only affect train_loss
            # -1: infinite epochs, None: default 1000 epochs
            max_epochs=getattr(self.kwargs, "max_epochs", None),
            # max_epoch must be valid, min_steps is prior to early stopping
            min_steps=getattr(self.kwargs, "min_steps", -1),
            limit_train_batches=getattr(self.kwargs, "limit_train_batches", None),
            limit_val_batches=getattr(self.kwargs, "limit_val_batches", None),
            accelerator="gpu",
            devices=[i for i in range(num_gpus)],
            strategy="auto" if num_gpus <= 1 else "ddp",
            callbacks=callbacks,
            profiler=PyTorchProfiler(
                dirpath="./profiler", filename=f"{self.__class__.__name__}"
            ),
        )

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
