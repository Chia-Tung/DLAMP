from datetime import datetime, timedelta

import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback

from visual import VizRadar

from ...datasets import CustomDataset
from ...standardization import destandardization
from ...utils import DataGenerator


class LogPredictionSamplesCallback(Callback):
    def __init__(self, log_image_every_n_steps: int):
        super().__init__()
        self.log_freq = log_image_every_n_steps
        self.painter = VizRadar()
        self.global_step_record = 0
        self.already_load_data_for_plot = False
        #
        self.log_input_tensors = []
        self.log_target_imgs = []
        self.log_pred_imgs = []

    def on_validation_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self.already_load_data_for_plot == True:
            return

        # load axis
        custom_dataset: CustomDataset = pl_module.test_dataloader().dataset
        data_gnrt: DataGenerator = custom_dataset._data_gnrt
        self.data_lat, self.data_lon = data_gnrt.yield_data(
            datetime(2022, 10, 1, 0), {"Lat": ["NoRule"], "Lon": ["NoRule"]}
        )

        # choose cases from `src.const.EVAL_CASES`
        cases = [datetime(2022, 9, 12)]  # datetime(2022, 10, 16)
        for case in cases:
            internal_idx = custom_dataset.get_internal_index_from_dt(case)
            input, target = custom_dataset[internal_idx]

            # Input Data: (lv, H, W, C) -> (1, lv, H, W, C)
            for k in input.keys():
                input[k] = torch.from_numpy(input[k][None]).cuda()
            self.log_input_tensors.append(input)  # torch.Tensor

            # Target Data
            target_radar = np.squeeze(destandardization(target["surface"]))  # (H, W)
            fig_gt = self.painter.plot_1x1(self.data_lon, self.data_lat, target_radar)
            self.log_target_imgs.append(wandb.Image(fig_gt[0]))

        self.already_load_data_for_plot = True

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        global_step = trainer.global_step
        if global_step != 0 and global_step - self.global_step_record < self.log_freq:
            return

        wandb_logger = trainer.logger.experiment

        # no step slider for Table: https://github.com/wandb/wandb/issues/1826
        # table = wandb.Table(columns=["case ID", "pred", "target"])
        for idx, input in enumerate(self.log_input_tensors):
            _, oup_surface = pl_module(input["upper_air"], input["surface"])
            oup_surface = destandardization(
                oup_surface.cpu().numpy()
            )  # (B, 1, H, W, 1)
            oup_surface = np.squeeze(oup_surface)  # (H, W)
            fig_pd, _ = self.painter.plot_1x1(self.data_lon, self.data_lat, oup_surface)
            self.log_pred_imgs.append(wandb.Image(fig_pd))

            # table.add_data(idx, wandb.Image(fig_pd), wandb.Image(fig_gt))

        wandb_logger.log(
            {"ground truth": self.log_target_imgs, "predictions": self.log_pred_imgs}
        )
        # wandb_logger.log({"prediction_table": table})

        self.log_pred_imgs.clear()
        self.global_step_record = global_step
