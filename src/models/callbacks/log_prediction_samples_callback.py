from datetime import datetime

import lightning as L
import numpy as np
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import Callback

from visual import VizPressure

from ...const import DATA_CONFIG_PATH
from ...datasets import CustomDataset
from ...standardization import destandardization
from ...utils import DataCompose, DataGenerator


class LogPredictionSamplesCallback(Callback):
    def __init__(self, log_image_every_n_steps: int):
        super().__init__()
        self.log_freq = log_image_every_n_steps
        self.painter = VizPressure()
        self.global_step_record = 0
        self.already_load_data_for_plot = False

        # load config
        with open(DATA_CONFIG_PATH, "r") as stream:
            data_config = yaml.safe_load(stream)
        data_list = DataCompose.from_config(data_config["train_data"])
        self.sfc_dc = [dc for dc in data_list if dc.level.is_surface()]

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
        dc_lat, dc_lon = DataCompose.from_config({"Lat": ["NoRule"], "Lon": ["NoRule"]})
        self.data_lat = data_gnrt.yield_data(custom_dataset._init_time_list[0], dc_lat)
        self.data_lon = data_gnrt.yield_data(custom_dataset._init_time_list[0], dc_lon)

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
            target_data = np.squeeze(destandardization(target["surface"]))  # (H, W, C)
            (slp,) = DataCompose.from_config({"SLP": ["SeaSurface"]})
            target_slp = target_data[:, :, self.sfc_dc.index(slp)]
            fig_gt = self.painter.plot_1x1(self.data_lon, self.data_lat, target_slp)
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
            )  # (B, 1, H, W, C)
            oup_surface = np.squeeze(oup_surface)  # (H, W, C)
            (slp,) = DataCompose.from_config({"SLP": ["SeaSurface"]})
            oup_slp = oup_surface[:, :, self.sfc_dc.index(slp)]
            fig_pd, _ = self.painter.plot_1x1(self.data_lon, self.data_lat, oup_slp)
            self.log_pred_imgs.append(wandb.Image(fig_pd))

            # table.add_data(idx, wandb.Image(fig_pd), wandb.Image(fig_gt))

        wandb_logger.log(
            {"ground truth": self.log_target_imgs, "predictions": self.log_pred_imgs}
        )
        # wandb_logger.log({"prediction_table": table})

        self.log_pred_imgs.clear()
        self.global_step_record = global_step
