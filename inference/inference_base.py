import abc
import warnings
from datetime import datetime, timedelta

import numpy as np
from omegaconf import DictConfig

from src.datasets import CustomDataset
from src.managers import DataManager, DatetimeManager
from src.utils import DataCompose, DataType, Level


class InferenceBase(metaclass=abc.ABCMeta):
    def __init__(self, cfg: DictConfig, eval_cases: list[datetime] | None = None):
        # args
        self.cfg = cfg
        self.eval_cases = eval_cases

        # useful properties
        self.data_list = DataCompose.from_config(self.cfg.data.train_data)
        self.data_itv = timedelta(**self.cfg.data.time_interval)
        self.output_itv = timedelta(**self.cfg.inference.output_itv)
        self.showcase_length = self.cfg.plot.figure_columns
        self.pressure_lv: list[Level] = DataCompose.get_all_levels(
            self.data_list, only_upper=True
        )
        self.upper_vars: list[DataType] = DataCompose.get_all_vars(
            self.data_list, only_upper=True
        )
        self.surface_vars: list[DataType] = DataCompose.get_all_vars(
            self.data_list, only_surface=True
        )

        # data manager
        self.init_time_list = self.build_init_time_list()
        self.data_manager = DataManager(
            self.data_list,
            **self.cfg.data,
            **self.cfg.lightning,
            init_time_list=self.init_time_list,
        )
        self.data_manager.setup("predict")

        # custom setup
        self._setup()

    def build_init_time_list(self) -> list[datetime] | None:
        """
        Builds a list of initial time for evaluation.

        This function make sure all the initial time are valid during entire showcase length.
        And also make sure their forecast time exists.

        Returns:
            list[datetime | None]: A sorted list of initial times for evaluation.

        Raises:
            ValueError: If the sanity check fails for the initial time or forecast time.

        """
        if self.eval_cases is None:
            warnings.warn(
                "No custom eval cases are provided, using default EVAL_CASES defined in `src.const`.",
                UserWarning,
            )
            return None

        init_time_list = set()
        for eval_case in self.eval_cases:
            for num in range(self.showcase_length):
                fcst_time = eval_case + self.output_itv * num

                if not DatetimeManager.sanity_check(fcst_time, self.data_list):
                    raise ValueError(
                        f"Sanity check failed for fcst time: {fcst_time}, "
                        f"please choose another day instead {eval_case}."
                    )

            init_time_list.add(eval_case)
        return sorted(init_time_list)

    @property
    def init_time(self) -> list[datetime]:
        """
        Since `self.init_time_list` is not always available, this function returns default init times
        used by CustomDataset from `self.data_manager` or `self.init_time_list`
        """
        return (
            self.init_time_list
            if self.init_time_list is not None
            else self.data_manager._predict_dataset._init_time_list
        )

    @abc.abstractmethod
    def _setup(self):
        """
        Prepare all necessary objects for inference when calling `__init__`.
        """
        return NotImplemented

    @abc.abstractmethod
    def infer(self):
        """
        Inference process.
        """
        return NotImplemented

    def boundary_swapping(
        self, data: np.ndarray, dt: datetime, pct_grid_swap: float
    ) -> np.ndarray:
        """
        Swaps the boundary values of the predicted data with actual values
        from the dataset. The width of this boundary ring is determined by
        the `pct_grid_swap` parameter.

        Args:
            data (np.ndarray): Input data array with shape (batch, level,
                width, height, channel). Batch must be 1.
            dt (datetime): The datetime for which to get the actual values.
            pct_grid_swap (float): Percentage of grid width/height to swap
                at boundaries. Value should be between 0 and 1.

        Returns:
            np.ndarray: Data array with boundary values swapped, same shape as input
                (batch, level, width, height, channel).

        Raises:
            AssertionError: If batch size is not 1.
        """
        assert (
            0 < pct_grid_swap < 1
        ), f"Value should be between 0 and 1, but got {pct_grid_swap}"
        dataset: CustomDataset = self.data_manager._predict_dataset
        data_dict = dataset._get_variables_from_dt(dt)  # {"surface": (z, h, w, c)...}

        batch, level, width, height, channel = data.shape
        assert batch == 1, f"Only 1 eval case at a time, but got {batch}"
        edge_x = np.ceil(pct_grid_swap * width / 2).astype(np.int32)
        edge_y = np.ceil(pct_grid_swap * height / 2).astype(np.int32)

        # Create a boolean mask for the boundary ring
        mask = np.zeros((width, height), dtype=bool)
        mask[:edge_x, :] = mask[-edge_x:, :] = True
        mask[:, :edge_y] = mask[:, -edge_y:] = True

        if level == 1:
            data[0, 0, mask, :] = data_dict["surface"][0, mask, :]
        else:
            data[0, :, mask, :] = data_dict["upper_air"][:, mask, :].transpose(1, 0, 2)

        return data
