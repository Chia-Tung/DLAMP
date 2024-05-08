import unittest
from copy import deepcopy

import torch
import yaml

from src.models import PanguModel


class PanguModelTest(unittest.TestCase):
    with open("config/model/dlamp_train.yaml", "r") as file:
        model_config = yaml.safe_load(file)

    TEST_CASE = deepcopy(model_config)
    TEST_CASE["image_shape"] = [224, 224]
    TEST_CASE["upper_levels"] = 6
    TEST_CASE["upper_channels"] = 4
    TEST_CASE["surface_channels"] = 1
    TEST_CASE.pop("model_name")

    def test_pangu_output_shape(self):
        device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
        model = PanguModel(**self.TEST_CASE).to(device)

        # input_tensor: shape of (B, img_Z, img_H, img_W, Ch)
        x_upper = torch.randn((16, 6, 224, 224, 4)).to(device)
        x_surface = torch.randn((16, 1, 224, 224, 1)).to(device)

        with torch.no_grad():
            y_upper, y_surface = model(x_upper, x_surface)

        self.assertEqual(x_upper.shape, y_upper.shape)
        self.assertEqual(x_surface.shape, y_surface.shape)


if __name__ == "__main__":
    # CLI: python -m src.models.architectures.pangu_model_test
    unittest.main(verbosity=2)
