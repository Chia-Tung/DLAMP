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
        self.fig_inputs = None
        self.fig_targets = None

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
        self.fig_inputs = []
        self.fig_targets = []
        for case in cases:
            internal_idx = custom_dataset.get_internal_index_from_dt(case)
            input, target = custom_dataset[internal_idx]
            # (lv, H, W, C) -> (1, lv, H, W, C)
            for k in input.keys():
                input[k] = torch.from_numpy(input[k][None]).cuda()
                target[k] = destandardization(target[k])
                target[k] = torch.from_numpy(target[k][None]).cuda()

            self.fig_inputs.append(input)
            self.fig_targets.append(target)

        self.already_load_data_for_plot = True

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        global_step = trainer.global_step
        if global_step != 0 and global_step - self.global_step_record < self.log_freq:
            return

        wandb_logger = trainer.logger.experiment
        fig_gt_list = []
        fig_pd_list = []

        # no step slider for Table: https://github.com/wandb/wandb/issues/1826
        # table = wandb.Table(columns=["case ID", "pred", "target"])
        for idx, (input, target) in enumerate(zip(self.fig_inputs, self.fig_targets)):
            _, oup_surface = pl_module(input["upper_air"], input["surface"])
            oup_surface = destandardization(oup_surface.cpu().numpy())
            oup_surface = np.squeeze(oup_surface)  # (H, W)
            tag_surface = np.squeeze(target["surface"].cpu().numpy())  # (H, W)

            fig_gt, _ = self.painter.plot_1x1(self.data_lon, self.data_lat, tag_surface)
            fig_pd, _ = self.painter.plot_1x1(self.data_lon, self.data_lat, oup_surface)

            fig_gt_list.append(wandb.Image(fig_gt))
            fig_pd_list.append(wandb.Image(fig_pd))

            # table.add_data(idx, wandb.Image(fig_pd), wandb.Image(fig_gt))

        wandb_logger.log({"ground truth": fig_gt_list, "predictions": fig_pd_list})
        # wandb_logger.log({"prediction_table": table})

        self.global_step_record = global_step
