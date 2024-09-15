import torch
import torch.nn as nn


class EuclideanLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # pred and target shape: (Batch Size, Channels, Height, Width)
        loss = torch.sqrt(torch.sum((pred - target) ** 2, dim=(1, 2, 3)))
        # loss shape: (Batch Size,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
