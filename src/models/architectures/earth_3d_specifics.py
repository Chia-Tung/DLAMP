import torch
import torch.nn as nn
from einops import rearrange, repeat

from ..model_utils import (
    crop_pad_3d,
    gen_3d_attn_mask,
    pad_3d,
    window_partition_3d,
    window_reverse_3d,
)
from .drop_path import DropPath

# Pangu Model
# TODO: 檢查input shape, window size __len__()
# TODO: dim要被heads整除

# Pangu Model, EarthSpecificBlock
# TODO: input shape可被window size整除, no need pad/crop
# TODO: error message:
# gen mask
# no padding for this version
# if roll: pad_mask + attn_mask
# if not roll: pad_mask


class EarthAttention3D(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int],
        dim: int,
        heads: int,
        dropout_rate: float,
        window_size: tuple[int],
    ) -> None:
        """
        3D window attention with the Earth-Specific bias,
        see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.

        Args:
            input_shape (tuple[int]): The shape of the input tensor whose dimensions are (Z, H, W). The input
                tensor represents a 3D image after patch embedding.
            dim (int): The dimension of the input tensor.
            heads (int): The number of attention heads.
            dropout_rate (float): The dropout rate.
            window_size (tuple[int]): The size of the window whose dimensions are (win_Z, win_H, win_W).

        Returns:
            None
        """
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 3, bias=True)
        self.linear2 = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        self.window_size = window_size
        self.num_head = heads
        self.dim = dim
        self.scale = (dim // heads) ** 0.5

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

    def _construct_index(self) -> None:
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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


class EarthSpecificBlock(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int],
        dim: int,
        heads: int,
        drop_path_ratio: float,
        dropout_rate: float,
        window_size: tuple[int],
        is_rolling: bool,
    ) -> None:
        """
        3D transformer block with Earth-Specific bias and window attention,
        see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
        The major difference is that we expand the dimensions to 3 and replace the relative position bias with Earth-Specific bias.

        Args:
            input_shape (tuple[int]): The shape of the input tensor whose dimensions are (Z, H, W). The input
                tensor represents a 3D image after patch embedding.
            dim (int): The dimension of the input tensor.
            heads (int): The number of attention heads.
            drop_path_ratio (float): The ratio of the drop path.
            dropout_rate (float): The dropout rate.
            window_size (tuple[int]): The size of the window whose dimensions are (win_Z, win_H, win_W).

        Returns:
            None
        """
        super().__init__()
        self.drop_path = DropPath(drop_prob=drop_path_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # self.linear = MLP(dim, 0)
        self.attention = EarthAttention3D(
            dim=dim,
            input_shape=input_shape,
            heads=heads,
            dropout_rate=dropout_rate,
            window_size=window_size,
        )

        self.img_Z, self.img_H, self.img_W = input_shape
        self.win_Z, self.win_H, self.win_W = window_size
        self.pad = pad_3d(input_shape, window_size)
        self.crop = crop_pad_3d(input_shape, window_size)
        self.mask = gen_3d_attn_mask(input_shape, window_size) if is_rolling else None
        self.register_buffer("attn_mask", self.mask)

    def forward(self, x, Z, H, W, roll):
        # Save the shortcut for skip-connection
        shortcut = x

        # Reshape input to three dimensions to calculate window attention
        reshape(x, target_shape=(x.shape[0], Z, H, W, x.shape[2]))

        # Zero-pad input if needed
        x = pad3D(x)

        # Store the shape of the input for restoration
        ori_shape = x.shape

        if roll:
            # Roll x for half of the window for 3 dimensions
            x = roll3D(
                x,
                shift=[
                    self.window_size[0] // 2,
                    self.window_size[1] // 2,
                    self.window_size[2] // 2,
                ],
            )
            # Generate mask of attention masks
            # If two pixels are not adjacent, then mask the attention between them
            # Your can set the matrix element to -1000 when it is not adjacent, then add it to the attention
            mask = gen_mask(x)
        else:
            # e.g., zero matrix when you add mask to attention
            mask = no_mask

        # Reorganize data to calculate window attention
        x_window = reshape(
            x,
            target_shape=(
                x.shape[0],
                Z // window_size[0],
                window_size[0],
                H // window_size[1],
                window_size[1],
                W // window_size[2],
                window_size[2],
                x.shape[-1],
            ),
        )
        x_window = TransposeDimensions(x_window, (0, 1, 3, 5, 2, 4, 6, 7))

        # Get data stacked in 3D cubes, which will further be used to calculated attention among each cube
        x_window = reshape(
            x_window,
            target_shape=(
                -1,
                window_size[0] * window_size[1] * window_size[2],
                x.shape[-1],
            ),
        )

        # Apply 3D window attention with Earth-Specific bias
        x_window = self.attention(x, mask)

        # Reorganize data to original shapes
        x = reshape(
            x_window,
            target_shape=(
                (
                    -1,
                    Z // window_size[0],
                    H // window_size[1],
                    W // window_size[2],
                    window_size[0],
                    window_size[1],
                    window_size[2],
                    x_window.shape[-1],
                )
            ),
        )
        x = TransposeDimensions(x, (0, 1, 4, 2, 5, 3, 6, 7))

        # Reshape the tensor back to its original shape
        x = reshape(x_window, target_shape=ori_shape)

        if roll:
            # Roll x back for half of the window
            x = roll3D(
                x,
                shift=[
                    -self.window_size[0] // 2,
                    -self.window_size[1] // 2,
                    -self.window_size[2] // 2,
                ],
            )

        # Crop the zero-padding
        x = Crop3D(x)

        # Reshape the tensor back to the input shape
        x = reshape(
            x,
            target_shape=(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4]),
        )

        # Main calculation stages
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.linear(x)))
        return x
