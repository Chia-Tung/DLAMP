from datetime import datetime, timedelta

import lightning as L
import numpy as np
import wandb
from lightning.pytorch.callbacks import Callback

from visual import VizRadar

from ...utils import DataCompose, DataType, Level, gen_data


class LogPredictionSamplesCallback(Callback):
    def __init__(self):
        super().__init__()
        self.fig_input = []
        self.fig_target = []
        self.painter = VizRadar()
        self.data_lat = gen_data(
            datetime(2022, 10, 1, 0), DataCompose(DataType.Lat, Level.Surface)
        )[1:-1:2, 1:-1:2]
        self.data_lon = gen_data(
            datetime(2022, 10, 1, 0), DataCompose(DataType.Lon, Level.Surface)
        )[1:-1:2, 1:-1:2]
        assert self.data_lat.shape == (224, 224)
        assert self.data_lon.shape == (224, 224)
        self.already_load_data_for_plot = False

    def on_validation_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self.already_load_data_for_plot == True:
            return

        # Pick cases from `src.const.EVAL_CASES`
        cases = [datetime(2022, 9, 12), datetime(2022, 10, 16)]
        custom_dataset = trainer.val_dataloaders.dataset
        for case in cases:
            # (lv, H, W, C)
            fig_input = custom_dataset._get_variables_from_dt(case, to_tensor=True)

            fig_target = custom_dataset._get_variables_from_dt(
                case + timedelta(hours=1), to_tensor=True
            )
            # (1, lv, H, W, C)
            for k in fig_input.keys():
                fig_input[k] = pl_module.preprocess_layer(fig_input[k][None]).cuda()
                fig_target[k] = pl_module.preprocess_layer(fig_target[k][None]).cuda()

            self.fig_input.append(fig_input)
            self.fig_target.append(fig_target)

        self.already_load_data_for_plot = True

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        wandb_logger = trainer.logger.experiment
        fig_gt_tmp = []
        fig_pd_tmp = []

        # no step slider for Table: https://github.com/wandb/wandb/issues/1826
        # table = wandb.Table(columns=["case ID", "pred", "target"])
        for idx, (input, target) in enumerate(zip(self.fig_input, self.fig_target)):
            _, oup_surface = pl_module(input["upper_air"], input["surface"])
            oup_surface = np.squeeze(oup_surface.cpu().numpy())  # (H, W)
            tag_surface = np.squeeze(target["surface"].cpu().numpy())  # (H, W)

            fig_gt, _ = self.painter.plot(self.data_lon, self.data_lat, tag_surface)
            fig_pd, _ = self.painter.plot(self.data_lon, self.data_lat, oup_surface)

            fig_gt_tmp.append(wandb.Image(fig_gt))
            fig_pd_tmp.append(wandb.Image(fig_pd))

            # table.add_data(idx, wandb.Image(fig_pd), wandb.Image(fig_gt))

        wandb_logger.log({"ground truth": fig_gt_tmp, "predictions": fig_pd_tmp})
        # wandb_logger.log({"prediction_table": table})
