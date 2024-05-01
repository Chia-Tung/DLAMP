import torch
import torch.nn as nn

__all__ = ["MultilayerPerceptron"]

class MultilayerPerceptron(nn.Module):
    def __init__(self, dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim*4)
        self.linear2 = nn.Linear(dim*4, dim)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x