from datetime import datetime
from typing import Callable, Sequence

import numpy as np
import torch
from einops.layers.torch import Rearrange
from torchvision.transforms.v2 import CenterCrop, Compose, Resize

from .data_compose import DataCompose
from .file_util import gen_data


class DataGenerator:
    def __init__(self, data_shape: list[int], image_shape: list[int]):
        self._data_shp = data_shape
        self._img_shp = image_shape
        self.preprocess = self._preprocess()

    def yield_data_hook(fn: Callable) -> Callable:
        def wrapper(
            self,
            target_time: datetime,
            data_compose: DataCompose | list[DataCompose],
            to_numpy: bool = True,
            **kwargs,
        ) -> Sequence | dict[str, Sequence]:
            """
            A wrapper function that handles data generation and preprocessing. The output can be either
            a single sequence data or a dictionary of sequences depending on the type of input data_compose.

            Parameters:
                target_time (datetime): The target timestamp for data generation.
                data_compose (DataCompose | list[DataCompose]): Single or multiple DataCompose objects
                    specifying the type and level of data to generate.
                to_numpy (bool, optional): Whether to return the processed data as numpy array or torch
                    Tensor. Defaults to True.

            Returns:
                Sequence | dict[str, Sequence]: Processed data either as a single sequence or
                dictionary of sequences if multiple DataCompose objects are provided.
            """
            data: np.ndarray | dict[str, np.ndarray] = gen_data(
                target_time, data_compose, dtype=np.float32, **kwargs
            )  # (H, W)

            if isinstance(data, np.ndarray):
                return fn(self, data, to_numpy)
            elif isinstance(data, dict):
                for key, np_data in data.items():
                    data[key] = fn(self, np_data, to_numpy)
                return data

        return wrapper

    @yield_data_hook
    def yield_data(
        self, np_data: np.ndarray, to_numpy: bool = True
    ) -> torch.Tensor | np.ndarray:
        """
        Generate a tensor or numpy array of processed data based on the provided numpy array.

        Parameters:
            np_data (np.ndarray): The numpy array containing the data to be processed.
            to_numpy (bool, optional): Whether to return the data as a numpy array. Defaults to True.

        Returns:
            torch.Tensor or np.ndarray: The processed data in shape (H, W).
        """
        if isinstance(self.preprocess, Compose):
            torch_data = torch.from_numpy(np_data[None]).type(
                torch.float32
            )  # (1, H, W)
            processed_data: torch.Tensor = self.preprocess(torch_data)  # (H, W)
            return processed_data.numpy() if to_numpy else processed_data
        else:
            return self.preprocess(np_data)

    def _preprocess(self) -> torch.Tensor:
        """
        Perform preprocessing on the data by applying a series of transformations:
        1. Center crop the image based on the specified dimensions.
        2. Resize the image to the desired shape.
        3. Rearrange the image dimensions from "c h w" to "h w".

        Returns:
            torch.Tensor: Preprocessed data
        """
        factors = [x // y for x, y in zip(self._data_shp, self._img_shp)]
        return Compose(
            [
                CenterCrop([n * x for n, x in zip(factors, self._img_shp)]),
                Resize(self._img_shp),
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
