import lightning as L
import numpy as np
import onnxruntime as ort
import wandb
from lightning.pytorch.loggers import WandbLogger

from ...standardization import destandardization
from .log_prediction_samples_callback import LogPredictionSamplesCallback


class LogDiffusionPredSamplesCallback(LogPredictionSamplesCallback):
    def __init__(self, log_image_every_n_steps: int):
        super().__init__(log_image_every_n_steps)

        self.log_first_guess_imgs = []
        self.log_final_imgs = []

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        global_step = trainer.global_step
        if pl_module.global_rank != 0 or (
            global_step != 0 and global_step - self.global_step_record < self.log_freq
        ):
            return

        wandb_logger: WandbLogger = trainer.logger.experiment
        regress_ort: ort.InferenceSession = pl_module.regress_ort

        for idx, input in enumerate(self.log_input_tensors):
            upper_ch = input["upper_air"].shape[-1]
            surface_ch = input["surface"].shape[-1]
            ort_inputs = {
                regress_ort.get_inputs()[0].name: input["upper_air"].cpu().numpy(),
                regress_ort.get_inputs()[1].name: input["surface"].cpu().numpy(),
            }
            # shape: (B, lv, H, W, C); type: np.ndarry
            first_guess_upper, first_guess_surface = regress_ort.run(None, ort_inputs)
            # shape: (B, Lv*C1+C2, H, W); type: torch.Tensor
            first_guess = pl_module.restruct_dimension(
                first_guess_upper,
                first_guess_surface,
                is_numpy=True,
                device=input["upper_air"].device,
            )

            # one-time logging for first guess w/ shape: (H, W)
            first_guess_surface = np.squeeze(destandardization(first_guess_surface))
            if len(self.log_first_guess_imgs) < len(self.log_input_tensors):
                fig_fg, _ = self.painter.plot_1x1(
                    self.data_lon, self.data_lat, first_guess_surface
                )
                self.log_first_guess_imgs.append(wandb.Image(fig_fg))

            # denoising process
            model_output = pl_module.denoising(first_guess, input["upper_air"].device)
            model_output_radar = {}
            for step, output in model_output.items():
                _, output_surface = pl_module.deconstruct(output, upper_ch, surface_ch)
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
                first_guess_surface + model_output_radar["step_0"],
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
