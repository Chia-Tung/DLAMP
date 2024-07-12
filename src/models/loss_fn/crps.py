import torch
import torch.nn as nn


class CRPS(nn.Module):
    def __init__(self, integral_number: int = 1000):
        super().__init__()
        self.number = integral_number

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return self._calculate_crps(prediction.flatten(), target.flatten())

    def _calculate_crps(self, prediction: torch.Tensor, target: torch.Tensor):
        min_val = torch.min(torch.min(prediction), torch.min(target))
        max_val = torch.max(torch.max(prediction), torch.max(target))

        x = torch.linspace(min_val, max_val, self.number, device=prediction.device)
        x = x.to(prediction.dtype)

        cdf_prediction = self._calculate_cdf(x, prediction)
        cdf_target = self._calculate_cdf(x, target)
        diff = torch.abs(cdf_prediction - cdf_target)

        return torch.trapz(diff**2, x)

    def _calculate_cdf(self, x: torch.Tensor, data: torch.Tensor):
        # use sigmoid to approximate the cdf, since genuine method:
        # return torch.mean((data.unsqueeze(1) <= x.unsqueeze(0)).float(), dim=0)
        # is not continuous.
        return torch.mean(
            torch.sigmoid((x.unsqueeze(0) - data.unsqueeze(1)) * 1000), dim=0
        )


class L1CRPS(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        sort_p, _ = torch.sort(torch.flatten(prediction))
        sort_t, _ = torch.sort(torch.flatten(target))
        dx = torch.abs(sort_p - sort_t)
        loss = torch.sum(dx) / torch.numel(target)

        return loss
