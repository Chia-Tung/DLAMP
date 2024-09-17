import lightning as L
import onnxruntime as ort
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import trange

from ..diffusion_process import DDIMProcess, DDPMProcess
from ..loss_fn import EuclideanLoss
from ..model_utils import RunningAverage


def create_diffusion_module(diffusion_type: DDIMProcess | DDPMProcess):
    """
    In order to dynamically create the lightning module based on different
    diffusion types (DDPM or DDIM), we define this factory function.
    """

    class DiffusionLightningModule(L.LightningModule, diffusion_type):
        def __init__(
            self, *, test_dataloader, backbone_model_fn, regression_model_fn, **kwargs
        ):
            super().__init__()
            diffusion_type.__init__(
                self,
                n_steps=kwargs["timesteps"],
                min_beta=kwargs["beta_start"],
                max_beta=kwargs["beta_end"],
            )

            self.save_hyperparameters(
                ignore=["test_dataloader", "backbone_model_fn", "regression_model_fn"]
            )

            self._test_dataloader: DataLoader = test_dataloader
            self.backbone_model_fn = backbone_model_fn
            self.backbone_model: nn.Module = None
            self.regression_model_fn = regression_model_fn
            self.regress_model: ort.InferenceSession | nn.Module = None
            self.loss = EuclideanLoss()
            self.loss_record = RunningAverage()

        def forward(self, noisy_img, time_step, condtion) -> torch.Tensor:
            """
            Args:
                noisy_img (torch.Tensor): (B, C, H, W)
                time_step (torch.Tensor): (B,)
                condtion (torch.Tensor): (B, C, H, W)

            Returns:
                torch.Tensor: the predict noist with shape (B, C, H, W)
            """
            return self.backbone_model(noisy_img, time_step, condtion)

        def configure_optimizers(self):
            # set optimizer
            optimizer = getattr(torch.optim, self.hparams.optim_config.name)(
                self.parameters(), **self.hparams.optim_config.args
            )

            # set lr scheduler
            def lr_lambda(epoch):
                if epoch <= self.hparams.warmup_epochs:
                    lr_scale = 1
                else:
                    overflow = epoch - self.hparams.warmup_epochs
                    lr_scale = 0.97**overflow
                    if lr_scale < 1e-1:
                        lr_scale = 1e-1
                return lr_scale

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "customized_lr",
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        def configure_model(self):
            """
            Speed up model initialization. Trainer can create model directly on GPU.
            """
            if self.backbone_model is not None and self.regress_model is not None:
                return

            self.backbone_model = self.backbone_model_fn()
            self.regress_model = self.regression_model_fn(self.global_rank)

            # freeze regression model
            for param in self.regress_model.parameters():
                param.requires_grad = False

        def common_step(self, inp_data, target):
            """
            Args:
                inp_data (dict): A dictionary containing the input data, like:
                    {
                        'upper_air': torch.Tensor (B, Lv, H, W, C),
                        'surface': torch.Tensor (B, 1, H, W, C)
                    }
                target (dict): A dictionary containing the target data, like:
                    {
                        'upper_air': torch.Tensor (B, Lv, H, W, C),
                        'surface': torch.Tensor (B, 1, H, W, C)
                    }

            Returns:
                loss: the CRPS loss
            """
            first_guess = self.inference_regression(
                inp_data["upper_air"], inp_data["surface"], self.device
            )
            target = self.restruct_dimension(target["upper_air"], target["surface"])
            B = target.shape[0]

            # DDPM
            x_0 = target - first_guess  # (B, C, H, W)
            t = torch.randint(0, self.hparams.timesteps, (B,), dtype=torch.long)  # (B,)
            t = t.to(self.device)
            x_t, noise = self.q_xt_x0(x_0, t)
            pred_noise = self(x_t.float(), t, first_guess.float())
            loss = self.loss(pred_noise, noise)
            self.loss_record.add(loss.item() * B, B)
            return loss * self.hparams.loss_factor

        def training_step(self, batch, batch_idx):
            inp_data, target = batch
            loss = self.common_step(inp_data, target)
            self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
            self.log("orig_loss", self.loss_record.get(), on_step=True, sync_dist=True)
            return loss

        def validation_step(self, batch, batch_idx):
            inp_data, target = batch
            loss = self.common_step(inp_data, target)
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            return loss

        def on_train_epoch_end(self):
            self.loss_record.reset()

        def test_dataloader(self) -> DataLoader:
            """
            Load the test dataset from external `LightningDataModule`.

            The reason doing so is that the `test_dataloader` is not accessible during
            the `trainer.fit()` loop, but we need the `test_dataloader` to record the
            images in `LogPredictionSamplesCallback`.
            """
            return self._test_dataloader

        # ============== the following functions are not coherent w/ LightningModule ==============
        # ============== however they are critical for training DDPM models          ==============

        def restruct_dimension(self, x_upper, x_surface, is_numpy=False, device=None):
            """
            Args:
                x_upper (torch.Tensor): Tensor of shape (B, Lv, H, W, C1)
                x_surface (torch.Tensor): Tensor of shape (B, 1, H, W, C2)
                is_numpy (bool, optional): Whether the input is in numpy format
                device (torch.device, optional): Device of the output tensor

            Returns:
                torch.Tensor: Tensor of shape (B, Lv*C1+C2, H, W)
            """
            if is_numpy and device is not None:
                x_upper = torch.from_numpy(x_upper).to(device)
                x_surface = torch.from_numpy(x_surface).to(device)
            elif is_numpy and device is None:
                raise ValueError("If `is_numpy` is True, `device` must be provided.")

            x_upper = rearrange(x_upper, "b z h w c -> b (z c) h w")
            x_surface = rearrange(x_surface, "b 1 h w c -> b c h w")
            x = torch.cat([x_upper, x_surface], dim=1)
            return x

        def deconstruct(
            self, x: torch.Tensor, upper_ch: int, surface_ch: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Deconstructs a tensor `x` into two tensors `x_upper` and `x_surface`.

            Args:
                x (torch.Tensor): The input tensor of shape (B, Lv*C1+C2, H, W).
                upper_ch (int): The number of channels in the upper layer (C1).
                surface_ch (int): The number of channels in the surface layer (C2).

            Returns:
                tuple[torch.Tensor, torch.Tensor]: A tuple containing `x_upper` and `x_surface`.
                    - `x_upper` (torch.Tensor): The upper layer tensor of shape (B, Lv, H, W, C1).
                    - `x_surface` (torch.Tensor): The surface layer tensor of shape (B, 1, H, W, C2).
            """
            x_upper, x_surface = x[:, :-surface_ch], x[:, -surface_ch:]
            x_upper = rearrange(x_upper, "b (z c) h w -> b z h w c", c=upper_ch)
            x_surface = rearrange(x_surface, "b c h w -> b 1 h w c")
            return x_upper, x_surface

        def denoising(
            self, cond: torch.Tensor, device: torch.device
        ) -> dict[int, torch.Tensor]:
            """
            Reverse process to get the image from noise. Log 6 images in a list.
            """
            B, C, H, W = cond.shape
            x = torch.randn(B, C, H, W).to(device)  # Start with random noise
            ims = {self.hparams.timesteps: x}
            if DDPMProcess in self.__class__.__bases__:
                steps = trange(
                    self.hparams.timesteps - 1, -1, -1, desc="DDPM Denoising"
                )
                for step in steps:
                    t = torch.full((B,), step, dtype=torch.long).to(device)
                    with torch.no_grad():
                        pred_noise = self(x, t, cond)
                        x = self.sampling(x, pred_noise, t)
                    if step % (self.hparams.timesteps // 5) == 0:
                        ims[step] = x
            elif DDIMProcess in self.__class__.__bases__:
                ddim_steps = self.hparams.timesteps // 5
                skipped_steps = torch.linspace(
                    self.hparams.timesteps, 0, (ddim_steps + 1), dtype=torch.long
                )
                steps = trange(1, ddim_steps + 1, desc="DDIM Denoising")
                for step in steps:
                    curr_t = skipped_steps[step - 1] - 1  # t large
                    prev_t = skipped_steps[step] - 1  # t small
                    t = torch.full((B,), curr_t, dtype=torch.long).to(device)
                    with torch.no_grad():
                        pred_noise = self(x, t, cond)
                        x = self.sampling(
                            x, pred_noise, curr_t, prev_t, eta=0, simple_var=False
                        )
                    if step % (ddim_steps // 5) == 0:
                        key = int(prev_t + 1)
                        ims[key] = x
            return ims

        def inference_regression(
            self, input_upa: torch.Tensor, input_sfc: torch.Tensor, device: torch.device
        ) -> torch.Tensor:
            """
            Inference process for the regression model. This process can be either onnxruntime-gpu
            or original pytorch.

            Args:
                input_upa (torch.Tensor): (B, Lv, H, W, C1)
                input_sfc (torch.Tensor): (B, 1, H, W, C2)
                device (torch.device): Device of the output tensor.

            Returns:
                torch.Tensor: (B, Lv*C1+C2, H, W)
            """
            if self.regress_model is None:
                self.regress_model = self.regression_model_fn(device)

            if isinstance(self.regress_model, ort.InferenceSession):
                ort_inputs = {
                    self.regress_model.get_inputs()[0].name: input_upa.cpu().numpy(),
                    self.regress_model.get_inputs()[1].name: input_sfc.cpu().numpy(),
                }
                first_guess_upper, first_guess_surface = self.regress_model.run(
                    None, ort_inputs
                )
            elif isinstance(self.regress_model, torch.nn.Module):
                with torch.inference_mode():
                    first_guess_upper, first_guess_surface = self.regress_model(
                        input_upa, input_sfc
                    )
                first_guess_surface = torch.clone(first_guess_surface).detach_()
                first_guess_upper = torch.clone(first_guess_upper).detach_()
            else:
                raise NotImplementedError

            first_guess = self.restruct_dimension(
                first_guess_upper,
                first_guess_surface,
                is_numpy=isinstance(self.regress_model, ort.InferenceSession),
                device=device,
            )
            return first_guess

    return DiffusionLightningModule
