import unittest

import torch

from .glide_unet import GlideUNet, ResidualBlock


class GlideUNetTest(unittest.TestCase):
    def test_unet(self):
        batch_size = 16
        channels = 3
        hidden_dim = 128
        ch_mults = (1, 2, 2, 1)
        is_attn = (False, False, False, True)
        n_blocks = 4

        # input
        input_shape = (batch_size, channels, 224, 224)
        x = torch.randn(input_shape)
        x = x.cuda()

        # time step for diffusion
        t = torch.randint(0, 1000, (batch_size,))
        t = t.cuda()

        # condition input
        cond = torch.randn(input_shape)
        cond = cond.cuda()

        # model
        model = GlideUNet(
            image_channels=channels,
            hidden_dim=hidden_dim,
            ch_mults=ch_mults,
            is_attn=is_attn,
            n_blocks=n_blocks,
        )
        model = model.cuda()

        with torch.no_grad():
            y = model(x, t, cond)

        self.assertEqual(x.shape, y.shape)


if __name__ == "__main__":
    # CLI: python -m src.models.architectures.glide_unet_test
    unittest.main(verbosity=2)
