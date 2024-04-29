import torch
import torch.nn as nn
from einops import rearrange, repeat

# TODO: 檢查input shape, window size __len__()
# TODO: input shape可被window size整除
# TODO: dim要被heads整除


class EarthAttention3D:
    def __init__(
        self,
        input_shape: tuple[int],
        dim: int,
        heads: int,
        dropout_rate: float,
        window_size: tuple[int],
    ):
        """
        3D window attention with the Earth-Specific bias,
        see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
        """
        self.linear1 = nn.Linear(dim, dim * 3, bias=True)
        self.linear2 = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        self.input_shape = input_shape
        self.window_size = window_size
        self.num_head = heads
        self.dim = dim
        self.scale = (dim // heads) ** 0.5

        self.img_Z, self.img_H, self.img_W = input_shape
        self.win_Z, self.win_H, self.win_W = window_size
        self.num_Z, self.num_H, self.num_W = map(
            lambda x, y: x // y, input_shape, window_size
        )

        # Record the number of different windows of the entire domain
        self.type_of_windows = self.num_Z * self.num_H

        # For each window, we will construct a set of parameters according to the paper
        self.earth_specific_bias = torch.empty(
            size=(
                (2 * self.win_W - 1) * self.win_H**2 * self.win_Z**2,
                self.type_of_windows,
                heads,
            ),
            dtype=torch.float32,
        )
        self.earth_specific_bias = nn.Parameter(self.earth_specific_bias)
        self.earth_specific_bias = nn.init.trunc_normal_(
            self.earth_specific_bias, std=0.02
        )

        self._construct_index()

    def _construct_index(self):
        """This function construct the position index to reuse symmetrical parameters of the position bias"""
        coords_zi = torch.arange(self.win_Z)
        coords_zj = -torch.arange(self.win_Z) * self.win_Z
        coords_hi = torch.arange(self.win_H)
        coords_hj = -torch.arange(self.win_H) * self.win_H
        coords_w = torch.arange(self.win_W)

        # Change the order of the index to calculate the index in total
        coords_1 = torch.stack(
            torch.meshgrid(coords_zi, coords_hi, coords_w, indexing="ij"), dim=0
        )
        coords_2 = torch.stack(
            torch.meshgrid(coords_zj, coords_hj, coords_w, indexing="ij"), dim=0
        )
        coords_flatten_1 = torch.flatten(coords_1, start_dim=1)
        coords_flatten_2 = torch.flatten(coords_2, start_dim=1)
        coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
        coords = rearrange(coords, "d1 d2 d3 -> d2 d3 d1")

        # Shift the index for each dimension to start from 0 and non-repetitive
        coords[:, :, 2] += self.win_W - 1
        coords[:, :, 1] *= 2 * self.win_W - 1
        coords[:, :, 0] *= (2 * self.win_W - 1) * self.win_H**2

        self.position_index = coords.sum(-1)
        self.position_index = self.position_index.flatten()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x (torch.Tensor): Tensor of shape (B*num_windows, win_Z*win_H*win_W, dim)
                where num_windows = num_Z*num_H*num_W.
            mask (torch.Tensor): Tensor of shape (num_windows, win_Z*win_H*win_W, win_Z*win_H*win_W).
        Returns:
            torch.Tensor: Tensor of shape (B*num_windows, win_Z*win_H*win_W, dim).
                where num_windows = num_Z*num_H*num_W.
        """
        original_shape = x.shape
        x = self.linear1(x)  # (B*num_windows, win_Z*win_H*win_W, dim*3)
        query, key, value = rearrange(
            x, "b zhw (qkv n_h d) -> qkv b n_h zhw d", qkv=3, n_h=self.num_head
        )  # [3](B*num_windows, num_head, win_Z*win_H*win_W, dim//num_head)
        attention = (query @ key.transpose(-2, -1)) / self.scale

        # EarthSpecificBias: shape of ((win_Z*win_H*win_W)^2, type_of_windows, num_head)
        EarthSpecificBias = self.earth_specific_bias[self.position_index]

        # EarthSpecificBias: shape of (type_of_windows, num_head, win_Z*win_H*win_W, win_Z*win_H*win_W)
        EarthSpecificBias = rearrange(
            EarthSpecificBias,
            "(zhw1 zhw2) t_w n_h-> t_w n_h zhw1 zhw2",
            zhw1=self.win_Z * self.win_H * self.win_W,
        )

        # Repeat the learnable bias to the same shape as the attention matrix
        # EarthSpecificBias: shape of (B*num_windows, num_head, win_Z*win_H*win_W, win_Z*win_H*win_W)
        EarthSpecificBias = repeat(
            EarthSpecificBias,
            "t_w n_h zhw1 zhw2 ->  (f t_w) n_h zhw1 zhw2",
            f=original_shape[0] // self.type_of_windows,
        )

        attention += EarthSpecificBias

        if mask is not None:
            mask = repeat(
                mask,
                "n_w zhw1 zhw2 -> (batch n_w) zhw1 zhw2",
                batch=original_shape[0] // (self.num_Z * self.num_H * self.num_W),
            )
            attention += mask[None, ::, None]

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        x = (attention @ value).transpose(1, 2).reshape(original_shape)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
