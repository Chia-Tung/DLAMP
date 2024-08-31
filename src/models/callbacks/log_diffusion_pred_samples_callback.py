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

        self.fig_gt_list = []
        self.fig_fg_list = []

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        global_step = trainer.global_step
        if pl_module.global_rank != 0 or (
            global_step != 0 and global_step - self.global_step_record < self.log_freq
        ):
            return

        wandb_logger: WandbLogger = trainer.logger.experiment
        regress_ort: ort.InferenceSession = pl_module.regress_ort
        fig_pd_list = []
        fig_final_list = []

        for idx, (input, target) in enumerate(zip(self.fig_inputs, self.fig_targets)):
            upper_ch = target["upper_air"].shape[-1]
            surface_ch = target["surface"].shape[-1]
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
                device=target["upper_air"].device,
            )
            # shape: (H, W); type: np.ndarry
            fgs_surface = np.squeeze(destandardization(first_guess_surface))
            # shape: (H, W); type: np.ndarry
            tag_surface = np.squeeze(target["surface"].cpu().numpy())
            # denoising process
            outputs = pl_module.denoising(first_guess, target["upper_air"].device)
            output_plot = []
            steps = []
            for step, output in outputs.items():
                _, output_surface = pl_module.deconstruct(output, upper_ch, surface_ch)
                output_surface = output_surface.cpu().numpy()
                output_surface = np.squeeze(destandardization(output_surface))
                output_plot.append(output_surface)
                steps.append(f"step_{step}")

            if len(self.fig_gt_list) <= idx:
                fig_gt, _ = self.painter.plot_1x1(
                    self.data_lon, self.data_lat, tag_surface
                )
                fig_fg, _ = self.painter.plot_1x1(
                    self.data_lon, self.data_lat, fgs_surface
                )
                self.fig_gt_list.append(wandb.Image(fig_gt))
                self.fig_fg_list.append(wandb.Image(fig_fg))

            fig_pd, _ = self.painter.plot_1xn(
                self.data_lon, self.data_lat, output_plot, titles=steps
            )
            fig_final, _ = self.painter.plot_1x1(
                self.data_lon, self.data_lat, fgs_surface + output_plot[-1]
            )
            fig_pd_list.append(wandb.Image(fig_pd))
            fig_final_list.append(wandb.Image(fig_final))

        wandb_logger.log(
            {
                "ground truth": self.fig_gt_list,
                "first guess": self.fig_fg_list,
                "diffusion": fig_pd_list,
                "final output": fig_final_list,
            }
        )
        self.global_step_record = global_step
