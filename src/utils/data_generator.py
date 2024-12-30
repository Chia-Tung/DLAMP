from datetime import datetime
from typing import Callable, Sequence
from warnings import warn

import numpy as np
import torch
from einops.layers.torch import Rearrange
from scipy.interpolate import LinearNDInterpolator
from torchvision.transforms.v2 import CenterCrop, Compose, Resize

from .data_compose import DataCompose, DataType, Level
from .file_util import gen_data


class DataGenerator:
    def __init__(
        self,
        lat_range: list[float],
        lon_range: list[float],
        resolution: float,
        data_shape: list[int] | None = None,
        image_shape: list[int] | None = None,
    ):
        target_time = datetime(2022, 10, 1, 0)
        self.data_lat = gen_data(target_time, DataCompose(DataType.Lat, Level.NoRule))
        self.data_lon = gen_data(target_time, DataCompose(DataType.Lon, Level.NoRule))
        self.points = np.column_stack((self.data_lat.ravel(), self.data_lon.ravel()))
        self.target_lat, self.target_lon = self.setup_target_lat_lon(
            lat_range, lon_range, resolution
        )

        # shape info
        self._data_shp = data_shape if data_shape else self.data_lat.shape
        self._img_shp = image_shape if image_shape else self.target_lat.shape

        # preprocess
        # WARNING: interpolation is too slow currently. Should be fixed in the future
        # self.preprocess = self._interp
        self.preprocess = self._preprocess()

    def yield_data_hook(fn: Callable) -> Callable:
        def wrapper(
            self,
            target_time: datetime,
            data_compose: DataCompose | list[DataCompose],
            to_numpy: bool = True,
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
                target_time, data_compose, dtype=np.float32
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

    def _interp(self, data: np.ndarray) -> np.ndarray:
        """
        Interpolates the given 2D data to the target latitude and longitude using
        scipy.interpolate.LinearNDInterpolator.

        Args:
            data (np.ndarray): The 2D data to be interpolated.

        Returns:
            np.ndarray: The interpolated data.
        """
        warn("This function will be removed in the future.", DeprecationWarning)
        values = data.ravel()
        interp = LinearNDInterpolator(self.points, values)
        interpolated_data = interp(self.target_lat, self.target_lon)
        return interpolated_data

    def setup_target_lat_lon(
        self,
        lat_range: list[float],
        lon_range: list[float],
        resolution: float,
        epsilon: float = 1e-5,
    ):
        """
        Set up the target latitude and longitude arrays for interpolation.

        Args:
            lat_range (list[float]): The range of target latitudes.
            lon_range (list[float]): The range of target longitudes.
            resolution (float): The resolution of the target latitudes and longitudes.
            epsilon (float, optional): A small value to be added to the upper limit of
                the range to ensure that the upper limit is included. Defaults to 1e-5.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple of two 2D arrays: the first is the
                target latitude array, and the second is the target longitude array.
        """
        warn("This function will be removed in the future.", DeprecationWarning)
        lat = np.arange(lat_range[0], lat_range[1] + epsilon, resolution)
        lon = np.arange(lon_range[0], lon_range[1] + epsilon, resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        return lat_mesh, lon_mesh
