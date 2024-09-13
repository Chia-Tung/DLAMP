import torch
from torch import Tensor

from .ddpm_process import DDPMProcess


class DDIMProcess(DDPMProcess):

    def __init__(self, n_steps: int, min_beta: float = 0.0001, max_beta: float = 0.02):
        super().__init__(n_steps, min_beta, max_beta)

    def sampling(
        self,
        xt: Tensor,
        eps_model: Tensor,
        curr_t: Tensor,
        prev_t: Tensor,
        eta=1.0,
        simple_var=True,
    ) -> Tensor:
        """
        xₜ₋₁ = √(ᾱₜ₋₁/ᾱₜ) · xₜ +
            (√(1 - ᾱₜ₋₁ - (1 - ᾱₜ₋₁)/(1 - ᾱₜ)βₜ) - √(ᾱₜ₋₁(1 - ᾱₜ)/ᾱₜ)) · ϵ +
            √β̃ₜz
        where z ~ N(0, I)

        β̃ₜ = η · (1 - ᾱₜ₋₁) / (1 - ᾱₜ) * βₜ where βₜ = 1 - ᾱₜ / ᾱₜ₋₁
        """
        self.device_check(xt.device)

        alpha_bar_curr = self.alpha_bars[curr_t]
        alpha_bar_prev = self.alpha_bars[prev_t] if prev_t >= 0 else 1

        if simple_var:
            eta = 1
        beta_t = 1 - alpha_bar_curr / alpha_bar_prev
        var = eta * (1 - alpha_bar_prev) / (1 - alpha_bar_curr) * beta_t

        first_term = torch.sqrt(alpha_bar_prev / alpha_bar_curr) * xt
        second_term = (
            torch.sqrt(1 - alpha_bar_prev - var)
            - torch.sqrt(alpha_bar_prev * (1 - alpha_bar_curr) / alpha_bar_curr)
        ) * eps_model

        if simple_var:
            var = beta_t
        sigma_t = torch.sqrt(var)
        noise = torch.randn_like(xt) * sigma_t
        x_tminus1 = first_term + second_term + noise
        return x_tminus1
