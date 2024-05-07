import torch
import torch.nn as nn
from einops import rearrange, repeat

from ..model_utils import (
    crop_pad_3d,
    is_divisible_elementwise,
    pad_3d,
    window_partition_3d,
    window_reverse_3d,
)
from .drop_path import DropPath
from .multilayer_perceptron import MultilayerPerceptron


class EarthSpecificLayer(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int],
        dim: int,
        heads: int,
        depth: int,
        drop_path_ratio_list: list[float],
        dropout_rate: float,
        window_size: tuple[int],
    ) -> None:
        """
        Basic layer of the network, contains either 2 or 6 blocks

        Args:
            input_shape (tuple[int]): The shape of the input tensor (inp_Z, inp_H, inp_W).
            dim (int): The dimension of the input tensor after patch embedding.
            heads (int): The number of heads in the multi-head attention layer.
            depth (int): The number of blocks in this layer.
            drop_path_ratio_list (list[float]): The drop path ratio for each block.
            dropout_rate (float): The dropout rate.
            window_size (tuple[int]): The window size with shape (win_Z, win_H, win_W).

        Raises:
            ValueError: If depth is not either 2 or 6.
            AssertionError: If the length of drop_path_ratio_list is not equal to depth.

        Returns:
            None
        """
        if depth not in [2, 6]:
            raise ValueError("depth should be either 2 or 6")
        assert (
            len(drop_path_ratio_list) == depth
        ), "length of drop_path_ratio_list should be equal to depth"
        super().__init__()

        self.depth = depth
        self.blocks = nn.ModuleList(
            (
                EarthSpecificBlock(
                    input_shape=input_shape,
                    dim=dim,
                    heads=heads,
                    drop_path_ratio=drop_path_ratio_list[i],
                    dropout_rate=dropout_rate,
                    window_size=window_size,
                    is_rolling=(i % 2 == 1),
                )
                for i in range(depth)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, inp_Z*inp_H*inp_W, dim).
        Returns:
            torch.Tensor: Tensor of shape (batch_size, inp_Z*inp_H*inp_W, dim).
        """
        for i in range(self.depth):
            x = self.blocks[i](x)
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
            input_shape (tuple[int]): The shape of the input tensor whose dimensions are (inp_Z, inp_H, inp_W). The input
                tensor represents a 3D image after patch embedding.
            dim (int): The dimension of the input tensor.
            heads (int): The number of attention heads.
            drop_path_ratio (float): The ratio of the drop path.
            dropout_rate (float): The dropout rate.
            window_size (tuple[int]): The size of the window whose dimensions are (win_Z, win_H, win_W).
            is_rolling (bool): Whether to use shifted window attention.
        Returns:
            None
        """
        assert is_divisible_elementwise(input_shape, window_size), (
            f"Input shape must be divisible by window_size {window_size}, but {input_shape} "
            "cannot be divided by it. \nIn the future, undivisible input shape is accepted only "
            "when `self.pad` and `self.crop` are implemented. One must take the padding mask "
            "into consideration when generating the attention mask. e.g. if not roll: pad_mask; "
            "if roll: pad_mask + attn_mask."
        )
        super().__init__()

        self.drop_path = DropPath(drop_prob=drop_path_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.MLP = MultilayerPerceptron(dim, 0)
        self.attention = EarthAttention3D(
            dim=dim,
            input_shape=input_shape,
            heads=heads,
            dropout_rate=dropout_rate,
            window_size=window_size,
        )

        self.is_rolling = is_rolling
        self.input_shape = input_shape
        self.window_size = window_size
        self.pad = pad_3d(input_shape, window_size)
        self.crop = crop_pad_3d(input_shape, window_size)
        if is_rolling:
            mask = self._gen_3d_attn_mask(input_shape, window_size)
            self.register_buffer("attn_mask", mask)

    def _gen_3d_attn_mask(
        self,
        input_shape: tuple[int, int, int],
        window_size: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Generates a 3D attention mask tensor based on the given image shape and window size.

        see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.

        Args:
            input_shape (tuple[int, int, int]): The shape of the image tensor (inp_Z, inp_H, inp_W).
            window_size (tuple[int, int, int]): The size of the sliding window (win_Z, win_H, win_W).

        Returns:
            torch.Tensor: The 3D attention mask tensor with shape (num_windows, win_Z*win_H*win_W, win_Z*win_H*win_W)
        """
        shift_size = tuple(i // 2 for i in window_size)
        img_mask = torch.zeros(1, input_shape[0], input_shape[1], input_shape[2], 1)
        z_slices = (
            slice(0, -window_size[0]),
            slice(-window_size[0], -shift_size[0]),
            slice(-shift_size[0], None),
        )
        h_slices = (
            slice(0, -window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        )
        w_slices = (
            slice(0, -window_size[2]),
            slice(-window_size[2], -shift_size[2]),
            slice(-shift_size[2], None),
        )
        cnt = 0
        for z in z_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, z, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition_3d(img_mask, window_size)
        mask_windows = mask_windows.reshape(
            -1, window_size[0] * window_size[1] * window_size[2]
        )
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, inp_Z*inp_H*inp_W, dim).
        Returns:
            torch.Tensor: Tensor of shape (B, inp_Z*inp_H*inp_W, dim).
        """
        shortcut = x
        inp_Z, inp_H, inp_W = self.input_shape
        x = rearrange(x, "b (z h w) c -> b z h w c", z=inp_Z, h=inp_H, w=inp_W)

        # backward shift
        if self.is_rolling:
            x = x.roll(shifts=[-i // 2 for i in self.window_size], dims=(1, 2, 3))

        # x: shape of (B * num_windows, win_Z*win_H*win_W, dim)
        x = window_partition_3d(x, self.window_size, combine_img_dim=True)

        # Apply 3D window attention with Earth-Specific bias
        x = self.attention(x, getattr(self, "attn_mask", None))

        # x: shape of (B, inp_Z, inp_H, inp_W, dim)
        x = window_reverse_3d(
            x, self.window_size, self.input_shape, from_combine_dim=True
        )

        # forward shift
        if self.is_rolling:
            x = x.roll(shifts=[i // 2 for i in self.window_size], dims=(1, 2, 3))

        x = rearrange(x, "b z h w c -> b (z h w) c")
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.MLP(x)))
        return x


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
            input_shape (tuple[int]): The shape of the input tensor whose dimensions are (inp_Z, inp_H, inp_W).
                The input tensor represents a 3D image after patch embedding.
            dim (int): The dimension of the input tensor.
            heads (int): The number of attention heads.
            dropout_rate (float): The dropout rate.
            window_size (tuple[int]): The size of the window whose dimensions are (win_Z, win_H, win_W).

        Returns:
            None
        """
        assert is_divisible_elementwise(
            [dim], [heads]
        ), f"dim {dim} must be divisible by heads {heads}"
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

        self.register_buffer("position_index", self._construct_index())

    def _construct_index(self) -> torch.Tensor:
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

        position_index = coords.sum(-1)
        position_index = position_index.flatten()
        return position_index

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
        orig_shape = x.shape
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
            f=orig_shape[0] // self.type_of_windows,
        )

        attention += EarthSpecificBias

        if mask is not None:
            mask = repeat(
                mask,
                "n_w zhw1 zhw2 -> (batch n_w) zhw1 zhw2",
                batch=orig_shape[0] // (self.type_of_windows * self.num_W),
            )
            attention += mask[::, None]

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        x = (attention @ value).transpose(1, 2).reshape(orig_shape)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
