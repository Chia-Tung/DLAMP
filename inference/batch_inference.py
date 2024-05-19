import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import torch
from omegaconf import DictConfig

from src.managers import DataManager, DatetimeManager
from src.models import get_builder
from src.utils import DataCompose, DataType, Level

from .infer_utils import prediction_postprocess

log = logging.getLogger(__name__)


class BatchInference:
    def __init__(self, cfg: DictConfig, eval_cases: list[datetime] | None = None):
        self.cfg = cfg
        self.eval_cases = eval_cases
        self._setup()

    def _setup(self):
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

        # assign model and trainer
        self.init_time_list = self.build_init_time_list()
        self.data_manager = DataManager(
            self.data_list,
            **self.cfg.data,
            **self.cfg.lightning,
            **self.cfg.inference,
            init_time_list=self.init_time_list,
        )
        self.model_builder = get_builder(self.cfg.model.model_name)(
            "predict", self.data_list, **self.cfg.model, **self.cfg.lightning
        )
        self.model = self.model_builder.build_model()
        self.trainer = self.model_builder.build_trainer(False)

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
            init_times = self.showcase_init_time_list(eval_case)
            for init_time in init_times:
                fcst_time = init_time + self.output_itv

                if not DatetimeManager.sanity_check(init_time, self.data_list):
                    raise ValueError(
                        f"Sanity check failed for init time: {init_time}, "
                        f"please choose another day instead {eval_case}."
                    )
                if not DatetimeManager.sanity_check(fcst_time, self.data_list):
                    raise ValueError(
                        f"Sanity check failed for fcst time: {fcst_time}, "
                        f"please choose another day instead {eval_case}."
                    )
                init_time_list.add(init_time)
        return sorted(init_time_list)

    def infer(self):
        """
        Do inference.

        Save the `input_upper`, `input_surface`, `target_upper`, `target_surface`,
        `output_upper`, and `output_surface` as attributes of the class. Each attribute
        is a `torch.Tensor` of shape (B, img_Z, img_H, img_W, Ch).

        Args:
            None

        Returns:
            None
        """
        predictions = self.trainer.predict(
            self.model, self.data_manager, ckpt_path=self.cfg.inference.best_ckpt
        )
        mapping = self.model.get_product_mapping()
        predictions = prediction_postprocess(predictions, mapping)
        for product_type, tensor in predictions.items():
            setattr(self, product_type, tensor)

        log.info(f"Batch inference finished at {datetime.now()}")

    def get_figure_materials(
        self, case_dt: datetime, data_compose: DataCompose
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the target and output data from a given case initial time.

        Args:
            case_dt (datetime): The datetime of the case.
            data_compose (DataCompose): The data composition object.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the target data and output data.
                - target_data (np.ndarray): The target data with shape (S, H, W).
                - output_data (np.ndarray): The output data with shape (S, H, W).
        """

        def innner_fn(phase: str) -> torch.Tensor:
            """
            Args:
                phase [str]: "input" | "target" | "output"
            """
            ret = []
            init_times = self.showcase_init_time_list(case_dt)
            for init_time in init_times:
                data = self.get_infer_outputs_from_dt(init_time, phase, data_compose)
                ret.append(data)
            ret = torch.stack(ret)  # (S, H, W)
            return ret

        target_data = innner_fn("target").numpy()
        output_data = innner_fn("output").numpy()
        return target_data, output_data

    def get_infer_outputs_from_dt(
        self, dt: datetime, phase: str, data_compose: DataCompose
    ) -> torch.Tensor:
        time_idx = self.init_time.index(dt)
        if data_compose.level.is_surface():
            var_idx = self.surface_vars.index(data_compose.var_name)
            return getattr(self, f"{phase}_surface")[time_idx, 0, :, :, var_idx]
        else:
            level_idx = self.pressure_lv.index(data_compose.level)
            var_idx = self.upper_vars.index(data_compose.var_name)
            return getattr(self, f"{phase}_upper")[time_idx, level_idx, :, :, var_idx]

    @property
    def init_time(self) -> list[datetime]:
        """
        Since `self.init_time_list` is not always available, this function returns default init times
        used by CustomDataset from `self.data_manager` or `self.init_time_list`
        """
        return (
            self.init_time_list
            if self.init_time_list is not None
            else self.trainer.predict_dataloaders.dataset._init_time_list
        )

    def showcase_init_time_list(self, eval_case: datetime) -> list[datetime]:
        return [eval_case + self.data_itv * i for i in range(self.showcase_length)]
