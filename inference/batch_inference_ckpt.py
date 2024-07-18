import logging
from datetime import datetime

import numpy as np
import torch
from omegaconf import DictConfig

from src.models.model_utils import get_builder
from src.utils import DataCompose

from .infer_utils import prediction_postprocess
from .inference_base import InferenceBase

log = logging.getLogger(__name__)


class BatchInferenceCkpt(InferenceBase):
    def __init__(self, cfg: DictConfig, eval_cases: list[datetime] | None = None):
        """
        This class can only inference by checkpoint.
        """
        super().__init__(cfg, eval_cases)

    def _setup(self):
        self.model_builder = get_builder(self.cfg.model.model_name)(
            "predict", self.data_list, **self.cfg.model, **self.cfg.lightning
        )
        self.model = self.model_builder.build_model(
            predict_iters=self.cfg.inference.output_itv.hours
        )
        self.trainer = self.model_builder.build_trainer(False)

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

    def showcase_init_time_list(self, eval_case: datetime) -> list[datetime]:
        return [eval_case + self.data_itv * i for i in range(self.showcase_length)]
