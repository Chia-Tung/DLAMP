import lightning as L
import torch
import torch.nn as nn


class DiffusionLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        pass

    def common_step(self, *args, **kwargs):
        pass

    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def forward_process(self, x_0, t):
        # precalculations
        betas = self.linear_schedule()
        alphas = 1 - betas

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

        # 回傳與X_0相同size的noise tensor，也就是reparameterization的epsilon
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]

        return (
            sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise,
            noise,
        )

    def linear_schedule(self, timesteps=500, start=0.0001, end=0.02):
        """
        return a tensor of a linear schedule
        """
        return torch.linspace(start, end, timesteps)
