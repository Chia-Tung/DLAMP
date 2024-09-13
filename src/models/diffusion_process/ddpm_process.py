import torch


class DDPMProcess:
    def __init__(
        self,
        n_steps: int,
        min_beta: float = 0.0001,
        max_beta: float = 0.02,
    ):
        betas = DDPMProcess.linear_beta_schedule(n_steps, min_beta, max_beta)
        self.betas = betas
        self.n_steps = n_steps
        self.prepare_constants()

    def q_xt_xtminus1(
        self, xtminus1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Equation: xₜ = √1 - βₜxₜ₋₁ + √βₜε
        """
        self.device_check(xtminus1.device)
        alpha = self.alphas[t].reshape(-1, 1, 1, 1)
        beta = self.betas[t].reshape(-1, 1, 1, 1)
        eps = torch.randn_like(xtminus1)
        xt = torch.sqrt(alpha) * xtminus1 + torch.sqrt(beta) * eps
        return xt, eps

    def q_xt_x0(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Equation: xₜ = √ᾱₜx₀ + √1 - ᾱₜε
        """
        self.device_check(x0.device)
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * eps
        return xt, eps

    def sampling(
        self,
        xt: torch.Tensor,
        eps_model: torch.Tensor,
        t: torch.Tensor,
        simple_var=True,
    ) -> torch.Tensor:
        """
        Equation: xₜ₋₁ = 1/√αₜ * [xₜ - (1 - αₜ) / √(1 - ᾱₜ) * εθ(xₜ, t)] + √β̃ₜz

        if simple_var == True:
            β̃ₜ = βₜ
        else:
            β̃ₜ = (1 - ᾱₜ₋₁) / (1 - ᾱₜ) * βₜ
        """
        self.device_check(xt.device)
        if t == 0:
            beta_t_hat = 0
        else:
            if simple_var:
                beta_t_hat = self.betas[t]
            else:
                beta_t_hat = (
                    (1 - self.alpha_bars[t - 1])
                    / (1 - self.alpha_bars[t])
                    * self.betas[t]
                )
        sigma_t = torch.sqrt(beta_t_hat) if beta_t_hat != 0 else 0
        noise = torch.randn_like(xt) * sigma_t

        mean = (
            xt - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps_model
        ) / torch.sqrt(self.alphas[t])
        x_tminus1 = mean + noise
        return x_tminus1

    def device_check(self, device: torch.device):
        if self.betas.device != device:
            self.betas = self.betas.to(device)
            self.prepare_constants()

    def prepare_constants(self):
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def cosine_beta_schedule(timesteps, min_beta=0.0001, max_beta=0.9999, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, min_beta, max_beta)

    @staticmethod
    def linear_beta_schedule(timesteps, min_beta, max_beta):
        return torch.linspace(min_beta, max_beta, timesteps)

    @staticmethod
    def quadratic_beta_schedule(timesteps, min_beta, max_beta):
        return torch.linspace(min_beta**0.5, max_beta**0.5, timesteps) ** 2

    @staticmethod
    def sigmoid_beta_schedule(timesteps, min_beta, max_beta):
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (max_beta - min_beta) + min_beta
