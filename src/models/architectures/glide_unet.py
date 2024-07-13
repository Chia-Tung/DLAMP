import torch
from torch import Tensor, nn

from .unet import AttentionBlock, ResidualBlock, Swish, TimeEmbedding

__all__ = ["GlideUNet"]


class DownBlock(nn.Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        orig_channels: int,
    ):
        super().__init__()
        self.has_attn = has_attn
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
            self.cond_proj = nn.Conv2d(orig_channels, out_channels, kernel_size=(1, 1))
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        # return `cond` so the behavior is the same as Downsample
        x = self.res(x, t)
        if self.has_attn:
            cond_transform = self.cond_proj(cond)
            x += cond_transform
        x = self.attn(x)
        return x, cond


class UpBlock(nn.Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        orig_channels: int,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.has_attn = has_attn
        self.res = ResidualBlock(
            in_channels + out_channels, out_channels, time_channels
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
            self.cond_proj = nn.Conv2d(orig_channels, out_channels, kernel_size=(1, 1))
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        x = self.res(x, t)
        if self.has_attn:
            cond_transform = self.cond_proj(cond)
            x += cond_transform
        x = self.attn(x)
        return x, cond


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int, orig_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)
        self.cond_proj = nn.Conv2d(orig_channels, n_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        x = self.res1(x, t)
        cond_transform = self.cond_proj(cond)
        x += cond_transform
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels: int, orig_channels: int):
        super().__init__()
        self.conv_x = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))
        self.conv_cond = nn.Conv2d(orig_channels, orig_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        x = self.conv_x(x)
        cond = self.conv_cond(cond)
        return x, cond


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels, orig_channels):
        super().__init__()
        self.conv_x = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

        # If you are using attention(cond information) not only at the last layer,
        # you need to turn on `self.conv_cond` and cond = self.conv_cond(cond) to
        # upsample the cond.

        # Note: In Pytorch 2.3.1, there is a bug when using `nn.ConvTranspose2d` in
        # `DistributedDataParallel`. This bug makes gradients of `nn.ConvTranspose2d.weight`
        # and `nn.ConvTranspose2d.bias` to be None.

        # self.conv_cond = nn.ConvTranspose2d(orig_channels, orig_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        _ = cond
        # cond = self.conv_cond(cond)
        x = self.conv_x(x)
        return x, cond


class GlideUNet(nn.Module):
    def __init__(
        self,
        image_channels: int,
        hidden_dim: int,
        ch_mults: tuple[int, ...] | list[int] = (1, 2, 2, 4),
        is_attn: tuple[bool, ...] | list[int] = (False, False, True, True),
        n_blocks: int = 2,
    ):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `hidden_dim` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(
            image_channels, hidden_dim, kernel_size=(3, 3), padding=(1, 1)
        )

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        time_channels = hidden_dim * 4
        self.time_emb = TimeEmbedding(time_channels)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = hidden_dim
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels, out_channels, time_channels, is_attn[i], hidden_dim
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels, hidden_dim))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, time_channels, hidden_dim)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels, out_channels, time_channels, is_attn[i], hidden_dim
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels, out_channels, time_channels, is_attn[i], hidden_dim
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels, hidden_dim))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, hidden_dim)
        self.act = Swish()
        self.final = nn.Conv2d(
            in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1)
        )

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        * `cond` has shape `[batch_size, in_channels, height, width]`
        """

        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)
        cond = self.image_proj(cond)

        # Add condition at the beginning
        x += cond

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x, cond = m(x, t, cond)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t, cond)

        # Second half of U-Net
        for m in self.up:
            if not isinstance(m, Upsample):
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
            x, cond = m(x, t, cond)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))
