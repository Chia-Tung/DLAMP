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
        attn_num_heads = 8

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
            input_channels=channels,
            hidden_dim=hidden_dim,
            ch_mults=ch_mults,
            is_attn=is_attn,
            attn_num_heads=attn_num_heads,
        )
        model = model.cuda()

        with torch.no_grad():
            y = model(x, t, cond)

        self.assertEqual(x.shape, y.shape)


class ResidualBlockTest(unittest.TestCase):
    batch_size = 16
    H = 224
    W = 480
    test_case_0 = {
        "in_channels": 128,
        "out_channels": 256,
        "time_channels": 64,
    }
    test_case_1 = {
        "in_channels": 128,
        "out_channels": 64,
        "time_channels": 32,
    }

    def test_residual_block(self):
        for i in range(2):
            test_case = self.test_case_0 if i == 0 else self.test_case_1
            in_channels = test_case["in_channels"]
            out_channels = test_case["out_channels"]
            time_channels = test_case["time_channels"]

            # input
            x = torch.randn((self.batch_size, in_channels, self.H, self.W))
            x = x.cuda()
            t_emb = torch.randn((self.batch_size, time_channels))
            t_emb = t_emb.cuda()

            # model
            residual_block = ResidualBlock(in_channels, out_channels, time_channels)
            residual_block = residual_block.cuda()

            with torch.no_grad():
                y = residual_block(x, t_emb)

            self.assertEqual(
                y.shape, torch.Size([self.batch_size, out_channels, self.H, self.W])
            )


if __name__ == "__main__":
    # CLI: python -m src.models.architectures.glide_unet_test
    unittest.main(verbosity=2)
