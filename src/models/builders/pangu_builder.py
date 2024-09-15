from pathlib import Path

import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler
from torch.utils.data import DataLoader

from ...const import CHECKPOINT_DIR
from ...utils import DataCompose, convert_hydra_dir_to_timestamp
from .. import PanguModel
from ..callbacks import LogPredictionSamplesCallback
from ..lightning_modules import PanguLightningModule
from .base_builder import BaseBuilder

__all__ = ["PanguBuilder"]


class PanguBuilder(BaseBuilder):
    def __init__(self, hydra_dir: Path, data_list: list[DataCompose], **kwargs):
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

        self.time_stamp = convert_hydra_dir_to_timestamp(hydra_dir)

        self.info_log(f"Input Image Shape: {self.kwargs.image_shape}")
        self.info_log(f"Patch Size: {self.kwargs.patch_size}")
        self.info_log(f"Window Size: {self.kwargs.window_size}")

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
            segmented_smooth_boundary_width=self.kwargs.segmented_smooth_boundary_width,
        )

    def build_model(
        self,
        test_dataloader: DataLoader | None = None,
        predict_iters: int | None = None,
    ) -> LightningModule:
        return PanguLightningModule(
            test_dataloader=test_dataloader,
            backbone_model=self._backbone_model(),
            upper_var_weights=None,
            surface_var_weights=None,
            surface_alpha=self.kwargs.surface_alpha,
            pressure_levels=self.pressure_levels,
            upper_vars=self.upper_vars,
            surface_vars=self.surface_vars,
            optim_config=self.kwargs.optim_config,
            lr_schedule=self.kwargs.lr_schedule,
            predict_iters=predict_iters,
        )

    def build_trainer(self, logger) -> Trainer:
        num_gpus = (
            torch.cuda.device_count()
            if self.kwargs.num_gpus is None
            else self.kwargs.num_gpus
        )
        strategy = getattr(self.kwargs, "strategy", "auto")

        callbacks = []
        callbacks.append(LearningRateMonitor())
        callbacks.append(self.checkpoint_callback())
        if self.kwargs.log_image_every_n_steps is not None:
            callbacks.append(
                LogPredictionSamplesCallback(self.kwargs.log_image_every_n_steps)
            )
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
            # If min_steps > 0, max_epoch must be valid. And min_steps is prior to early stopping
            min_steps=getattr(self.kwargs, "min_steps", -1),
            limit_train_batches=getattr(self.kwargs, "limit_train_batches", None),
            limit_val_batches=getattr(self.kwargs, "limit_val_batches", None),
            accelerator="gpu",
            devices=[i for i in range(num_gpus)],
            strategy=strategy,
            callbacks=callbacks,
            # profiler=AdvancedProfiler(
            #     dirpath="./profiler", filename=f"{self.__class__.__name__}"
            # ),
            precision=self.kwargs.precision,
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
            project="my-awesome-project",
            name=self.kwargs.model_name + f"_{self.time_stamp}",
            offline=True,
        )
