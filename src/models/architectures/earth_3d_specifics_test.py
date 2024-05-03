import unittest
from math import prod

import numpy as np
import torch

from ..model_utils import window_partition_3d
from .earth_3d_specifics import EarthAttention3D, EarthSpecificBlock, EarthSpecificLayer


class EarthAttention3DTest(unittest.TestCase):
    input_shape = [8, 140, 180]
    dim = 192
    heads = 6
    dropout_rate = 0.15
    window_size = [2, 4, 4]

    def test_earth_specific_bias(self):
        earth_attn_3d = EarthAttention3D(
            input_shape=self.input_shape,
            dim=self.dim,
            heads=self.heads,
            dropout_rate=self.dropout_rate,
            window_size=self.window_size,
        )

        total_movement = (
            earth_attn_3d.win_Z**2  # absolute movement in Z-axis
            * earth_attn_3d.win_H**2  # absolute movement in H-axis
            * (2 * earth_attn_3d.win_W - 1)  # relative movement in W-axis
        )

        self.assertEqual(
            len(set([x.item() for x in earth_attn_3d.position_index])), total_movement
        )

    def test_earth_attn_output_shape(self):

        earth_attn_3d = EarthAttention3D(
            input_shape=self.input_shape,
            dim=self.dim,
            heads=self.heads,
            dropout_rate=self.dropout_rate,
            window_size=self.window_size,
        )

        # input_tensor: shape of (B, img_Z, img_H, img_W, C)
        input_tensor = torch.randn([1] + self.input_shape + [self.dim])
        input_window = window_partition_3d(
            input_tensor, self.window_size, combine_img_dim=True
        )
        orig_shape = input_window.shape
        output_window = earth_attn_3d(input_window)
        self.assertEqual(output_window.shape, orig_shape)


class EarthSpecificBlockTest(unittest.TestCase):
    input_shape = [8, 140, 180]
    dim = 192
    heads = 6
    drop_path_ratio = 0.1
    dropout_rate = 0.15
    window_size = [2, 4, 4]

    def test_earth_specific_block_output_shape(self):
        for is_rolling in [True, False]:
            earth_specific_block = EarthSpecificBlock(
                input_shape=self.input_shape,
                dim=self.dim,
                heads=self.heads,
                drop_path_ratio=self.drop_path_ratio,
                dropout_rate=self.dropout_rate,
                window_size=self.window_size,
                is_rolling=is_rolling,
            )

            # input_tensor: shape of (B, img_Z*img_H*img_W, C)
            input_tensor = torch.randn([1] + [prod(self.input_shape)] + [self.dim])
            output_tensor = earth_specific_block(input_tensor)
            self.assertEqual(output_tensor.shape, input_tensor.shape)

    def test_wrong_input_shape(self):
        with self.assertRaises(AssertionError):
            EarthSpecificBlock(
                input_shape=[8, 141, 181],
                dim=self.dim,
                heads=self.heads,
                drop_path_ratio=self.drop_path_ratio,
                dropout_rate=self.dropout_rate,
                window_size=self.window_size,
                is_rolling=False,
            )


class EarthSpecificLayerTest(unittest.TestCase):
    input_shape = [8, 140, 180]
    dim = 192
    heads = 6
    depth = 2
    drop_path_ratio_list = np.linspace(0, 0.2, depth)
    dropout_rate = 0.15
    window_size = [2, 4, 4]

    def test_earth_specific_layer_output_shape(self):
        earth_specific_layer = EarthSpecificLayer(
            input_shape=self.input_shape,
            dim=self.dim,
            heads=self.heads,
            depth=self.depth,
            drop_path_ratio_list=self.drop_path_ratio_list,
            dropout_rate=self.dropout_rate,
            window_size=self.window_size,
        )

        # input_tensor: shape of (B, img_Z*img_H*img_W, C)
        input_tensor = torch.randn([1] + [prod(self.input_shape)] + [self.dim])
        output_tensor = earth_specific_layer(input_tensor)
        self.assertEqual(output_tensor.shape, input_tensor.shape)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    loader.testMethodPrefix = "test_"  # prefix for test methods

    suite.addTests(loader.loadTestsFromTestCase(EarthAttention3DTest))
    suite.addTests(loader.loadTestsFromTestCase(EarthSpecificBlockTest))
    suite.addTests(loader.loadTestsFromTestCase(EarthSpecificLayerTest))

    ### add single test by loader
    # suite.addTest(loader.loadTestsFromName(f"{__name__}.EarthAttention3DTest.test_earth_specific_bias"))

    ### add single test by TestCase
    # suite.addTest(EarthAttention3DTest("test_attn_output_shape"))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
