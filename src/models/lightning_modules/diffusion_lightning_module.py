import lightning as L
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import trange

from ..model_utils import RunningAverage


class DiffusionLightningModule(L.LightningModule):
    def __init__(
        self, *, test_dataloader, backbone_model_fn, regression_model_fn, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["test_dataloader", "backbone_model_fn", "regression_model_fn"]
        )

        self._test_dataloader: DataLoader = test_dataloader
        self.backbone_model_fn = backbone_model_fn
        self.backbone_model: nn.Module = None
        self.regression_model_fn = regression_model_fn
        self.regress_ort: ort.InferenceSession = None
        self.loss = nn.MSELoss()
        self.loss_record = RunningAverage()

        # DDPM
        self.beta = self.linear_beta_schedule(
            self.hparams.timesteps, self.hparams.beta_start, self.hparams.beta_end
        )

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

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
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
        if self.backbone_model is not None and self.regress_ort is not None:
            return

        self.backbone_model = self.backbone_model_fn()
        self.regress_ort = self.regression_model_fn(self.global_rank)

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
        ort_inputs = {
            self.regress_ort.get_inputs()[0].name: inp_data["upper_air"].cpu().numpy(),
            self.regress_ort.get_inputs()[1].name: inp_data["surface"].cpu().numpy(),
        }
        first_guess_upper, first_guess_surface = self.regress_ort.run(None, ort_inputs)
        first_guess = self.restruct_dimension(
            first_guess_upper,
            first_guess_surface,
            is_numpy=True,
            device=target["upper_air"].device,
        )
        target = self.restruct_dimension(target["upper_air"], target["surface"])
        B = target.shape[0]

        # only radar @ surface
        first_guess = first_guess[:, -1:, ...]
        target = target[:, -1:, ...]

        # device check
        if self.beta.device != target.device:
            self.beta = self.beta.to(target.device)

        # DDPM
        x_0 = target - first_guess  # (B, C, H, W)
        t = torch.randint(0, self.hparams.timesteps, (B,), dtype=torch.long).to(
            target.device
        )  # (B,)
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
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
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

    def gather(self, consts: torch.Tensor, t: torch.Tensor, x_dim: int):
        """
        Gather consts for t and reshape to feature map shape

        Args:
            consts (torch.Tensor): Tensor of shape (num_diffusion_steps,)
            t (torch.Tensor): Tensor of shape (batch_size,)
            x_dim (int): Number of dimensions of the input image

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 1, 1, 1)
        """
        c = consts.gather(-1, t)
        return c.reshape(-1, *((1,) * (x_dim - 1)))

    def denoising(
        self, cond: torch.Tensor, device: torch.device
    ) -> dict[int, torch.Tensor]:
        """
        Reverse process to get the image from noise. Log 6 images in a list.
        """
        B, C, H, W = cond.shape
        x = torch.randn(B, C, H, W).to(device)  # Start with random noise
        ims = {self.hparams.timesteps: x}
        for step in trange(self.hparams.timesteps - 1, -1, -1, desc="Denoising"):
            t = torch.full((B,), step, dtype=torch.long).to(device)
            with torch.no_grad():
                pred_noise = self(x, t, cond)
                x = self.p_xt(x, pred_noise, t)
            if step % (self.hparams.timesteps // 5) == 0:
                ims[step] = x
        return ims

    def q_xt_xtminus1(self, xtminus1: torch.Tensor, t: torch.Tensor):
        # √(1−βt)*xtm1
        alpha = 1.0 - self.beta
        alpha_t = self.gather(alpha, t, xtminus1.dim())
        mean = torch.sqrt(alpha_t) * xtminus1
        # βt I
        var = self.gather(self.beta, t, xtminus1.dim())
        # Noise shaped like xtm1
        eps = torch.randn_like(xtminus1)
        return mean + torch.sqrt(var) * eps, eps

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor):
        alpha = 1.0 - self.beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        mean = self.gather(sqrt_alpha_bar, t, x0.dim()) * x0
        var = self.gather(sqrt_one_minus_alpha_bar, t, x0.dim())
        eps = torch.randn_like(x0)
        return mean + var * eps, eps

    def p_xt(self, xt: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        if self.beta.device != xt.device:
            self.beta = self.beta.to(xt.device)
        alpha = 1.0 - self.beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        alpha_t = self.gather(alpha, t, xt.dim())
        alpha_bar_t = self.gather(alpha_bar, t, xt.dim())

        eps_coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean = (xt - eps_coef * noise) / torch.sqrt(alpha_t)

        posterior_var = self.beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        var = self.gather(posterior_var, t, xt.dim())
        eps = torch.randn_like(xt)
        if t == 0:
            return mean
        else:
            return mean + torch.sqrt(var) * eps

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def linear_beta_schedule(self, timesteps, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, timesteps)

    def quadratic_beta_schedule(self, timesteps, beta_start, beta_end):
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

    def sigmoid_beta_schedule(self, timesteps, beta_start, beta_end):
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
