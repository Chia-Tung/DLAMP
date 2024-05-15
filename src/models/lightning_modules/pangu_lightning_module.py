import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.utilities.grads import grad_norm

from ..model_utils import get_scheduler_with_warmup

__all__ = ["PanguLightningModule"]


# TODO: weighted MAE loss
class PanguLightningModule(L.LightningModule):
    def __init__(self, *, backbone_model, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone_model"])
        self.backbone_model: nn.Module = backbone_model

        if kwargs["upper_var_weights"] is None or kwargs["surface_var_weights"] is None:
            self.weighted_loss = False
            self.criterion = nn.L1Loss(reduction="mean")
            upper_var_weights_tensor = None
            surface_var_weights_tensor = None
        self.register_buffer("upper_var_weights", upper_var_weights_tensor)
        self.register_buffer("surface_var_weights", surface_var_weights_tensor)

    def forward(
        self, input_upper: torch.Tensor, input_surface: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone_model(input_upper, input_surface)

    def configure_optimizers(self):
        # set optimizer
        optimizer = getattr(torch.optim, self.hparams.optim_config.name)(
            self.parameters(), **self.hparams.optim_config.args
        )

        # set learning rate schedule
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR = get_scheduler_with_warmup(
            optimizer,
            training_steps=int(self.trainer.estimated_stepping_batches),
            schedule_type=self.hparams.lr_schedule.name,
            **self.hparams.lr_schedule.args,
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "customized_lr",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def common_step(self, inp_data, target):
        """
        Calculates 1. model output, 2. total loss and 3. the MAE loss for each element.

        Args:
            inp_data (dict): A dictionary containing the input data with keys "upper_air" and "surface",
                whose structure like:
                    {
                        "upper_air": (B, Z, H, W, C),
                        "surface": (B, 1, H, W, C)
                    }
            target (dict): A dictionary containing the target data with keys "upper_air" and "surface",
                whose structure like:
                    {
                        "upper_air": (B, Z, H, W, C),
                        "surface": (B, 1, H, W, C)
                    }
        """
        # all data in the shape of (B, Z, H, W, C)
        oup_upper, oup_surface = self(inp_data["upper_air"], inp_data["surface"])
        if self.weighted_loss:
            raise NotImplementedError("WeightedMAE not implemented")
        else:
            loss_upper = self.criterion(oup_upper, target["upper_air"])
            loss_surface = self.criterion(oup_surface, target["surface"])
        total_loss = loss_upper + loss_surface * self.hparams.surface_alpha

        # MAE for each variable/level, shape of (Z, C)
        mae_upper = torch.abs(oup_upper - target["upper_air"]).mean(dim=(0, 2, 3))
        mae_surface = torch.abs(oup_surface - target["surface"]).mean(dim=(0, 2, 3))

        return total_loss, (oup_upper, oup_surface), (mae_upper, mae_surface)

    def training_step(self, batch, batch_idx):
        inp_data, target = batch
        loss, _, (mae_upper, mae_surface) = self.common_step(inp_data, target)
        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log_mae_for_each_element(
            "train", self.hparams.pressure_levels, self.hparams.upper_vars, mae_upper
        )
        self.log_mae_for_each_element(
            "train", ["Surface"], self.hparams.surface_vars, mae_surface
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inp_data, target = batch
        loss, _, (mae_upper, mae_surface) = self.common_step(inp_data, target)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log_mae_for_each_element(
            "val", self.hparams.pressure_levels, self.hparams.upper_vars, mae_upper
        )
        self.log_mae_for_each_element(
            "val", ["Surface"], self.hparams.surface_vars, mae_surface
        )
        return loss

    # Compute the 2-norm for each layer
    # If using mixed precision, the gradients are already unscaled here
    # def on_before_optimizer_step(self, optimizer):
    #     norms = grad_norm(self.backbone_model, norm_type=2)
    #     self.log(
    #         name="gradient_2norm", value=norms["grad_2.0_norm_total"], on_step=True
    #     )
    #     norms.pop("grad_2.0_norm_total")
    #     self.log_dict(norms, on_step=True)

    def log_mae_for_each_element(
        self, prefix: str, lv_names: list[str], var_names: list[str], mae: torch.Tensor
    ):
        for i, pl in enumerate(lv_names):
            for j, var in enumerate(var_names):
                self.log(
                    f"{prefix}_mae/{var}_{pl}", mae[i, j], on_step=True, on_epoch=False
                )
