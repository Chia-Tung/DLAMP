from datetime import datetime
from typing import Callable, Sequence

import numpy as np
import torch
from einops.layers.torch import Rearrange
from torchvision.transforms.v2 import CenterCrop, Compose, Resize, ToDtype, ToImage

from .data_compose import DataCompose
from .file_util import gen_data


class DataGenerator:
    def __init__(self, data_shape: list[int], image_shape: list[int]):
        self._data_shp = tuple(data_shape)
        self._img_shp = tuple(image_shape)
        self.preprocess = self._preprocess()

    def input_is_dict(fn: Callable) -> Callable:

        def wrapper(
            self,
            target_time: datetime,
            data_compose: DataCompose | dict[str, list[str]],
            to_numpy: bool = True,
        ) -> Sequence | list[Sequence]:
            """
            A wrapper that allows the function to accept either a DataCompose object or a dictionary as input.

            Parameters:
                target_time (datetime): The target time for data processing.
                data_compose (DataCompose | dict[str, list[str]]): The data compose object or a dictionary.
                    The dictionary should have the following structure:
                        {
                            "GeoHeight": ["Hpa200", "Hpa500", "Hpa700", "Hpa850", "Hpa925"],
                            "T": ["Hpa200", "Hpa500", "Hpa700", "Hpa850", "Hpa925"],
                            ...
                        }
                to_numpy (bool, optional): Flag to determine if the output should be in numpy format.
                    Defaults to True.
            """
            if isinstance(data_compose, DataCompose):
                return fn(self, target_time, data_compose, to_numpy)
            elif isinstance(data_compose, dict):
                data_list: list[DataCompose] = DataCompose.from_config(data_compose)
                output_data = []
                for element in data_list:
                    yield_data = fn(self, target_time, element, to_numpy)
                    output_data.append(yield_data)
                return output_data
            else:
                raise ValueError(
                    f"Unsupported data type: {type(data_compose)}, expected DataCompose or dict[str, str]."
                )

        return wrapper

    @input_is_dict
    def yield_data(
        self, target_time: datetime, data_compose: DataCompose, to_numpy: bool = True
    ) -> torch.Tensor | np.ndarray:
        """
        Generate a tensor or numpy array of processed data based on the target time and data compose.

        Parameters:
            target_time (datetime): The target time for generating the data.
            data_compose (DataCompose): The data compose object.
            to_numpy (bool, optional): Whether to return the data as a numpy array. Defaults to True.

        Returns:
            torch.Tensor or np.ndarray: The processed data in shape (H, W).
        """
        np_data = gen_data(target_time, data_compose, dtype=np.float32)  # (H, W)
        processed_data: torch.Tensor = self.preprocess(np_data[:, :, None])  # (H, W)
        return processed_data.numpy() if to_numpy else processed_data

    def _preprocess(self) -> torch.Tensor:
        """
        Perform preprocessing on the data by applying a series of transformations:
        1. Convert the data to torch.Tensor format in "channel first" order.
        2. Center crop the image based on the specified dimensions.
        3. Resize the image to the desired shape.
        4. Convert the image to torch float32 dtype.
        5. Rearrange the image dimensions from "c h w" to "h w".

        Returns:
            torch.Tensor: Preprocessed data
        """
        factors = [x // y for x, y in zip(self._data_shp, self._img_shp)]
        return Compose(
            [
                ToImage(),
                CenterCrop([n * x for n, x in zip(factors, self._img_shp)]),
                Resize(self._img_shp),
                ToDtype(torch.float32),
                Rearrange("c h w -> (c h) w"),
            ]
        )

    def _data_shape_check(self, target_dt: datetime, data: np.ndarray):
        """
        Check if the shape of the given data matches the original data shape.

        Args:
            target_dt (datetime): The target datetime for which the data is being checked.
            data (np.ndarray): The data array to be checked.

        Raises:
            AssertionError: If the shape of the data does not match the expected shape.

        Returns:
            None
        """
        assert (
            data.shape == self._data_shp
        ), f"{target_dt} data shape mismatch: {data.shape} != {self._data_shp}"
