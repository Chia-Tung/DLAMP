from pathlib import Path
from typing import Callable

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
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.strategies import FSDPStrategy
from torch.utils.data import DataLoader

from inference.infer_utils import init_ort_instance, load_pangu_model

from ...const import CHECKPOINT_DIR
from ...utils import DataCompose, convert_hydra_dir_to_timestamp
from ..architectures import GlideUNet
from ..callbacks import LogDiffusionPredSamplesCallback
from ..diffusion_process import DDIMProcess, DDPMProcess
from ..lightning_modules import create_diffusion_module
from .base_builder import BaseBuilder

__all__ = ["GlideBuilder"]


class GlideBuilder(BaseBuilder):
    def __init__(self, hydra_dir: Path, data_list: list[DataCompose], **kwargs):
        super().__init__(**kwargs)

        self.time_stamp = convert_hydra_dir_to_timestamp(hydra_dir)
        self.input_channels = len(data_list)
        self.data_list = data_list

        self.info_log(f"Input Image Shape: {self.kwargs.image_shape}")
        self.info_log(f"Glide Unet Layers: {len(self.kwargs.ch_mults)}")

    def _backbone_model(self) -> nn.Module:
        return GlideUNet(
            image_channels=self.input_channels,
            hidden_dim=self.kwargs.hidden_dim,
            ch_mults=self.kwargs.ch_mults,
            is_attn=self.kwargs.is_attn,
            n_blocks=self.kwargs.n_blocks,
        )

    def _regression_model(self) -> Callable[[torch.device], nn.Module]:
        if self.kwargs.regression_onnx_path:
            return init_ort_instance(onnx_path=self.kwargs.regression_onnx_path)
        elif self.kwargs.regressoin_ckpt_path:
            return load_pangu_model(
                ckpt_path=self.kwargs.regressoin_ckpt_path, data_list=self.data_list
            )
        else:
            raise ValueError(
                "Either regression_onnx_path or regressoin_ckpt_path must be provided."
            )

    def build_model(self, test_dataloader: DataLoader | None = None) -> LightningModule:
        if self.kwargs.diffusion_type == "DDPM":
            parent_class = DDPMProcess
        elif self.kwargs.diffusion_type == "DDIM":
            parent_class = DDIMProcess
        else:
            raise ValueError("Invalid base class name.")

        return create_diffusion_module(parent_class)(
            test_dataloader=test_dataloader,
            backbone_model_fn=self._backbone_model,
            regression_model_fn=self._regression_model(),
            timesteps=self.kwargs.timesteps,
            beta_start=self.kwargs.beta_start,
            beta_end=self.kwargs.beta_end,
            optim_config=self.kwargs.optim_config,
            warmup_epochs=self.kwargs.warmup_epochs,
            loss_factor=self.kwargs.loss_factor,
        )

    def build_trainer(self, logger) -> Trainer:
        # set number of GPUs
        num_gpus = (
            torch.cuda.device_count()
            if self.kwargs.num_gpus is None
            else self.kwargs.num_gpus
        )

        # distributed training strategy
        strategy = getattr(self.kwargs, "strategy", "auto")
        match strategy:
            case "FULL_SHARD":
                strategy = FSDPStrategy(
                    sharding_strategy="FULL_SHARD", state_dict_type="sharded"
                )
            case "SHARD_GRAD_OP":
                strategy = FSDPStrategy(
                    sharding_strategy="SHARD_GRAD_OP", state_dict_type="sharded"
                )
            case _:
                pass

        # set callbacks
        callbacks = []
        callbacks.append(LearningRateMonitor())
        callbacks.append(self.checkpoint_callback())
        if self.kwargs.log_image_every_n_steps is not None:
            callbacks.append(
                LogDiffusionPredSamplesCallback(self.kwargs.log_image_every_n_steps)
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
            + f"_{self.kwargs.diffusion_type}"
            + f"_{self.time_stamp}"
            + "-{epoch:03d}-{val_loss_epoch:.6f}",
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
            name=self.kwargs.model_name
            + f"_{self.kwargs.diffusion_type}"
            + f"_{self.time_stamp}",
            offline=True,
        )
