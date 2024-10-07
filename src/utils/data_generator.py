from datetime import datetime
from typing import Callable, Sequence

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
        lat = np.arange(lat_range[0], lat_range[1] + epsilon, resolution)
        lon = np.arange(lon_range[0], lon_range[1] + epsilon, resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        return lat_mesh, lon_mesh
