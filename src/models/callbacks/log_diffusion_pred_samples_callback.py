import lightning as L
import numpy as np
import wandb
from lightning.pytorch.loggers import WandbLogger

from ...standardization import destandardization
from .log_prediction_samples_callback import LogPredictionSamplesCallback


class LogDiffusionPredSamplesCallback(LogPredictionSamplesCallback):
    def __init__(self, log_image_every_n_steps: int):
        super().__init__(log_image_every_n_steps)
        #
        self.first_guess = []
        self.first_guess_surface = []
        self.log_first_guess_imgs = []
        self.log_final_imgs = []

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        global_step = trainer.global_step
        if pl_module.global_rank != 0 or (
            global_step != 0 and global_step - self.global_step_record < self.log_freq
        ):
            return

        wandb_logger: WandbLogger = trainer.logger.experiment
        for idx, input in enumerate(self.log_input_tensors):
            upa_ch = input["upper_air"].shape[-1]
            sfc_ch = input["surface"].shape[-1]

            # one-time logging for first guess
            if len(self.log_first_guess_imgs) < len(self.log_input_tensors):
                regress = pl_module.inference_regression(
                    input["upper_air"], input["surface"], input["upper_air"].device
                )  # (B, Lv*C1+C2, H, W)
                regress_sfc = regress[:, -sfc_ch:].permute(0, 2, 3, 1)  # (B, H, W, C2)
                regress_sfc = regress_sfc.unsqueeze(1).cpu().numpy()  # (B, 1, H, W, C2)
                regress_sfc = np.squeeze(destandardization(regress_sfc))  # (H, W)
                fig_fg, _ = self.painter.plot_1x1(
                    self.data_lon, self.data_lat, regress_sfc
                )
                self.first_guess.append(regress)
                self.first_guess_surface.append(regress_sfc)
                self.log_first_guess_imgs.append(wandb.Image(fig_fg))

            # denoising process
            model_output = pl_module.denoising(
                self.first_guess[idx], input["upper_air"].device
            )
            model_output_radar = {}
            for step, output in model_output.items():
                # extract radar channel (B, 1, H, W, 1)
                output_surface = output[:, -1:, :, :, None]
                output_surface = output_surface.cpu().numpy()
                output_surface = np.squeeze(destandardization(output_surface))
                model_output_radar[f"step_{step}"] = output_surface

            # plot diffusion outputs
            fig_pd, _ = self.painter.plot_1xn(
                self.data_lon,
                self.data_lat,
                list(model_output_radar.values()),
                titles=list(model_output_radar.keys()),
            )
            fig_final, _ = self.painter.plot_1x1(
                self.data_lon,
                self.data_lat,
                self.first_guess_surface[idx] + model_output_radar["step_0"],
            )
            self.log_pred_imgs.append(wandb.Image(fig_pd))
            self.log_final_imgs.append(wandb.Image(fig_final))

        wandb_logger.log(
            {
                "ground truth": self.log_target_imgs,
                "first guess": self.log_first_guess_imgs,
                "diffusion": self.log_pred_imgs,
                "final output": self.log_final_imgs,
            }
        )
        self.global_step_record = global_step
        self.log_pred_imgs.clear()
        self.log_final_imgs.clear()
