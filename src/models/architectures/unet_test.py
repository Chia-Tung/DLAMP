import unittest

import torch

from .unet import AttentionBlock, Downsample, ResidualBlock, UNet, Upsample


class UnetTest(unittest.TestCase):
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

        # model
        model = UNet(
            input_channels=channels,
            hidden_dim=hidden_dim,
            ch_mults=ch_mults,
            is_attn=is_attn,
            attn_num_heads=attn_num_heads,
        )
        model = model.cuda()

        with torch.no_grad():
            y = model(x)

        self.assertEqual(x.shape, y.shape)


class DownsampleTest(unittest.TestCase):
    def test_downsample(self):
        channels = 3

        # input
        x = torch.randn((16, channels, 128, 128))
        x = x.cuda()

        # model
        downsample = Downsample(channels)
        downsample = downsample.cuda()

        with torch.no_grad():
            y = downsample(x)

        self.assertEqual(x.size(-2) / 2, y.size(-2))
        self.assertEqual(x.size(-1) / 2, y.size(-1))


class UpsampleTest(unittest.TestCase):
    def test_upsample(self):
        channels = 3

        # input
        x = torch.randn((16, channels, 128, 128))
        x = x.cuda()

        # model
        upsample = Upsample(channels)
        upsample = upsample.cuda()

        with torch.no_grad():
            y = upsample(x)

        self.assertEqual(x.size(-2) * 2, y.size(-2))
        self.assertEqual(x.size(-1) * 2, y.size(-1))


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

            # model
            residual_block = ResidualBlock(in_channels, out_channels, time_channels)
            residual_block = residual_block.cuda()

            with torch.no_grad():
                y = residual_block(x)

            self.assertEqual(
                y.shape, torch.Size([self.batch_size, out_channels, self.H, self.W])
            )


class AttentionBlockTest(unittest.TestCase):
    def test_attention_block(self):
        in_channels = 128
        attn_num_heads = 8

        # input
        x = torch.randn((16, in_channels, 32, 32))
        x = x.cuda()

        # model
        attention_block = AttentionBlock(in_channels, attn_num_heads)
        attention_block = attention_block.cuda()

        with torch.no_grad():
            y = attention_block(x)

        self.assertEqual(x.shape, y.shape)


if __name__ == "__main__":
    # CLI: python -m src.models.architectures.unet_test
    unittest.main(verbosity=2)
