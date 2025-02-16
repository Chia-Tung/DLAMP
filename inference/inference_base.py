import abc
import warnings
from datetime import datetime, timedelta

import numpy as np
from omegaconf import DictConfig
from tqdm import trange

from src.datasets import CustomDataset
from src.managers import DataManager, DatetimeManager
from src.utils import DataCompose, DataGenerator, DataType, Level


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

    def _boundary_swapping(
        self, data: np.ndarray, dt: datetime, method: str, bdy_grid: int = 8
    ) -> np.ndarray:
        """
        Swaps the boundary values of the predicted data with actual values
        from the dataset.

        Args:
            data (np.ndarray): Input data array with shape (batch, level,
                width, height, channel). Batch must be 1.
            dt (datetime): The datetime for which to get the actual values.
            method (str): "linear" or "override".
                "linear" means linearly replacing the boundary values.
                "override" means replace all values on the boundary with the actual values.
            bdy_grid (int, optional): Number of pixels to swap. Defaults to 8.

        Returns:
            np.ndarray: Data array with boundary values swapped, same shape as input
                (batch, level, width, height, channel).

        Raises:
            AssertionError: If batch size is not 1.
        """
        batch, level, width, height, channel = data.shape
        assert batch == 1, f"Only 1 eval case at a time, but got {batch}"
        edge_x = edge_y = bdy_grid

        dataset: CustomDataset = self.data_manager._predict_dataset
        # {"surface": (z, h, w, c), "upper_air": (z, h, w, c)}
        data_dict = dataset._get_variables_from_dt(dt, is_input=True)
        gt_data = data_dict["surface"] if level == 1 else data_dict["upper_air"]

        if method == "override":
            # Create a boolean mask for the boundary ring
            mask = np.zeros((width, height), dtype=bool)
            mask[:edge_x, :] = mask[-edge_x:, :] = True
            mask[:, :edge_y] = mask[:, -edge_y:] = True

            if level == 1:
                data[0, 0, mask, :] = gt_data[0, mask, :]
            else:
                data[0, :, mask, :] = gt_data[:, mask, :].transpose(1, 0, 2)
        elif method == "linear":
            # Linear interpolation for boundary pixels
            for i in range(bdy_grid):
                alpha = i / bdy_grid  # 0 is outer, 1 is inner
                mixed_gt_data = alpha * data[0] + (1 - alpha) * gt_data

                mask = np.zeros((width, height), dtype=bool)
                mask[i : i + 1, :] = mask[-i - 1 : -i, :] = True
                mask[:, i : i + 1] = mask[:, -i - 1 : -i] = True
                if level == 1:
                    data[0, 0, mask, :] = mixed_gt_data[0, mask, :]
                else:
                    data[0, :, mask, :] = mixed_gt_data[:, mask, :].transpose(1, 0, 2)
        else:
            raise ValueError(f"Unknown method: {method}")
        return data

    def get_figure_materials(self, case_dt: datetime, data_compose: DataCompose):
        """Get ground truth and prediction data for plotting figures.

        Args:
            case_dt (datetime): The initial datetime to get data for
            data_compose (DataCompose): Configuration specifying the variable and level to retrieve

        Returns:
            tuple[np.ndarray, np.ndarray]: data with shape (showcase_length, H, W)
        """
        input_data = self.get_infer_results_from_dt(case_dt, "input", data_compose)
        output_data = self.get_infer_results_from_dt(case_dt, "output", data_compose)

        # (showcase_length, H, W)
        output_plot_data = np.concatenate((input_data, output_data), axis=0)

        # prepare ground truth data
        data_gnrt: DataGenerator = self.data_manager.data_gnrt
        gt_data = []
        for i in trange(self.showcase_length, desc=f"Get {data_compose} ground truth"):
            curr_time = case_dt + i * self.output_itv
            gt_data.append(data_gnrt.yield_data(curr_time, data_compose))  # (H, W)
        gt_data = np.stack(gt_data, axis=0)  # (showcase_length, H, W)

        return gt_data, output_plot_data

    def get_infer_results_from_dt(
        self, dt: datetime, phase: str, data_compose: DataCompose
    ) -> np.ndarray:
        """Get inference input/output data for a specific datetime and variable.

        Args:
            dt (datetime): The datetime to get data for.
            phase (str): Either "input" or "output" to specify which data to retrieve.
                The shape of input data is (time, level, height, width, channel).
                The shape of output data is (time, seq_len, level, height, width, channel).
            data_compose (DataCompose): Configuration specifying the variable and level.

        Returns:
            np.ndarray: The requested data array. The output shape are all the same:
                (seq_len, height, width), for input phase, seq_len is 1.

        Raises:
            AssertionError: If phase is not "input" or "output"
            ValueError: If the requested variable or level is not found
        """
        assert phase in ["input", "output"], f"invalid phase: {phase}"
        time_idx = self.init_time.index(dt)
        if data_compose.level.is_surface():
            data = getattr(self, f"{phase}_surface")
            var_idx = self.surface_vars.index(data_compose.var_name)
            return (
                data[time_idx, :, :, :, var_idx]
                if phase == "input"
                else data[time_idx, :, 0, :, :, var_idx]
            )
        else:
            data = getattr(self, f"{phase}_upper")
            level_idx = self.pressure_lv.index(data_compose.level)
            var_idx = self.upper_vars.index(data_compose.var_name)
            return (
                data[time_idx, level_idx : level_idx + 1, :, :, var_idx]
                if phase == "input"
                else data[time_idx, :, level_idx, :, :, var_idx]
            )
