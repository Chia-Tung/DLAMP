import math
from functools import reduce

import torch
from torch import nn

__all__ = ["UNet"]


class UNet(nn.Module):
    """
    DDPM U-Net adapted from `https://github.com/labmlai/annotated_deep_learning_
    paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py`
    """

    def __init__(
        self,
        input_channels: int = 21,
        hidden_dim: int = 128,
        ch_mults: tuple[int] | list[int] = (1, 2, 2, 1),
        is_attn: tuple[bool] = (False, False, False, True),
        attn_num_heads: int = 8,
    ) -> None:
        """
        Initialize the U-Net architecture with the specified parameters.

        Args:
            input_channels (int): Number of input channels.
            hidden_dim (int): Hidden dimension size.
            ch_mults (tuple[int] | list[int]): the list of channel numbers at each resolution.
                The number of channels is `ch_mults[i] * hidden_dim`.
                The final dimension is also the number of channels for time embedding.
            is_attn (tuple[bool]): List of booleans indicating whether to use attention at each resolution.
            attn_num_heads (int): Number of attention heads.

        Returns:
            None
        """
        assert len(ch_mults) == len(is_attn)
        super().__init__()

        # Number of resolutions
        self.n_resolutions = len(ch_mults)

        # Number of blocks for each resolution
        n_blocks = list(map(lambda x: 2 if x else 4, is_attn))

        # Project input into feature map
        self.proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding="same")

        # Time embedding layer. Time embedding has `hidden_dim * mul(ch_mults)` channels
        time_dim = hidden_dim * reduce(lambda x, y: x * y, ch_mults)
        self.time_emb = TimeEmbedding(time_dim)

        # First half of U-Net - decreasing resolution
        self.down = nn.ModuleDict()
        out_channels = in_channels = hidden_dim

        for i in range(self.n_resolutions):
            out_channels = in_channels * ch_mults[i]

            down_block = []
            for _ in range(n_blocks[i]):
                down_block.append(
                    ResAttnBlock(
                        in_channels, out_channels, time_dim, is_attn[i], attn_num_heads
                    )
                )
                in_channels = out_channels
            self.down[f"layer{i}_down_Res_Attn"] = nn.ModuleList(down_block)

            # Down sample at all resolutions except the last
            if i < self.n_resolutions - 1:
                self.down[f"layer{i}_downsample"] = Downsample(in_channels)

        # Second half of U-Net - increasing resolution
        self.up = nn.ModuleDict()
        for i in reversed(range(self.n_resolutions)):
            up_block = []
            for j in range(n_blocks[i] - 1):  # last block has to reduce the n_channels
                if i < self.n_resolutions - 1 and j == 0:
                    # The input has `in_channels + out_channels` because we concatenate the
                    # output of the same resolution from the first half of the U-Net
                    up_block.append(
                        ResAttnBlock(
                            in_channels + out_channels,
                            out_channels,
                            time_dim,
                            is_attn[i],
                            attn_num_heads,
                        )
                    )
                    continue
                up_block.append(
                    ResAttnBlock(
                        in_channels, out_channels, time_dim, is_attn[i], attn_num_heads
                    )
                )
            out_channels = in_channels // ch_mults[i]
            up_block.append(
                ResAttnBlock(
                    in_channels, out_channels, time_dim, is_attn[i], attn_num_heads
                )
            )
            in_channels = out_channels
            self.up[f"layer{i}_up_Res_Attn"] = nn.ModuleList(up_block)

            # Up sample at all resolutions except last
            if i > 0:
                self.up[f"layer{i}_upsample"] = Upsample(in_channels)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, hidden_dim)
        self.act = Swish()
        self.final = nn.Conv2d(
            in_channels, input_channels, kernel_size=(3, 3), padding="same"
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape `[batch_size, in_channels, height, width]`.
            t (torch.Tensor): Time tensor of shape `[batch_size]`.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, in_channels, height, width]`.
        """
        t = self.time_emb(t)
        x = self.proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = []
        # First half of U-Net
        for i in range(self.n_resolutions):
            for res_block in self.down[f"layer{i}_down_Res_Attn"]:
                x = res_block(x, t)

            if i < self.n_resolutions - 1:
                h.append(x)
                x = self.down[f"layer{i}_downsample"](x)

        # Second half of U-Net
        for i in reversed(range(self.n_resolutions)):
            for res_block in self.up[f"layer{i}_up_Res_Attn"]:
                x = res_block(x, t)

            if i > 0:
                x = self.up[f"layer{i}_upsample"](x)
                skip = h.pop()
                x = torch.cat([x, skip], dim=1)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))


class Downsample(nn.Module):
    """
    Scale down the feature map by 0.5
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape `[batch_size, in_channels, in_height, in_width]`

        Returns:
            Tensor of shape `[batch_size, in_channels, out_height // 2, out_width // 2]`
        """
        return self.conv(x)


class Upsample(nn.Module):
    """
    Scale up the feature map by 2
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape `[batch_size, in_channels, in_height, in_width]`

        Returns:
            Tensor of shape `[batch_size, in_channels, out_height * 2, out_width * 2]`
        """
        return self.conv(x)


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.linear1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.linear2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings
        [same as those from the transformer](../../transformers/positional_encoding.html)

        Args:
            t (torch.Tensor): Time tensor of shape `[batch_size,]`

        Returns:
            torch.Tensor: Positional embedding tensor of shape `[batch_size, n_channels]`
        """
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.linear1(emb))
        emb = self.linear2(emb)

        return emb


class ResAttnBlock(nn.Module):
    """
    This combines `ResidualBlock` and `AttentionBlock`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        attn_heads: int,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels, attn_heads)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape `[batch_size, in_channels, height, width]`.
            t (torch.Tensor): Time tensor of shape `[batch_size]`.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, out_channels, height, width]`.
        """
        x = self.res(x, t)
        x = self.attn(x)
        return x


class ResidualBlock(nn.Module):
    """
    A residual block has two convolution layers with group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        n_groups: int = 32,
    ):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of output channels
        * `time_channels` is the number channels in the time embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding="same"
        )

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding="same"
        )

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_linear = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape `[batch_size, in_channels, in_height, in_width]`
            t: Time tensor of shape `[batch_size, time_channels]`

        Returns:
            Tensor of shape `[batch_size, out_channels, in_height, in_width]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        h += self.time_linear(t)[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.act2(self.norm2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(
        self, n_channels: int, n_heads: int, d_k: int | None = None, n_groups: int = 32
    ):
        """
        Args:
            n_channels: Number of channels in the input
            n_heads: Number of heads in multi-head attention
            d_k: Number of dimensions in each head. If None, it is set to `n_channels // n_heads`.
            n_groups: Number of groups for [group normalization](../../normalization/group_norm/index.html)

        Returns:
            None
        """
        super().__init__()
        if d_k is None:
            d_k = n_channels // n_heads

        self.n_heads = n_heads
        self.d_k = d_k
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape `[batch_size, in_channels, in_height, in_width]`

        Returns:
            Tensor of shape `[batch_size, in_channels, in_height, in_width]`
        """
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res


class Swish(nn.Module):
    """
    The Swish actiavation function
    """

    def forward(self, x):
        return x * torch.sigmoid(x)
