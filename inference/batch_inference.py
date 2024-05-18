import logging
import warnings
from datetime import datetime, timedelta

from omegaconf import DictConfig

from src.managers import DataManager, DatetimeManager
from src.models import get_builder
from src.utils import DataCompose

from .infer_utils import prediction_postprocess

log = logging.getLogger(__name__)


class BatchInference:
    def __init__(self, cfg: DictConfig, eval_cases: list[datetime] | None = None):
        self.cfg = cfg
        self.eval_cases = eval_cases
        self.data_list = DataCompose.from_config(cfg.data.train_data)
        self.init_time_list = self.build_init_time_list()
        self._setup()

    def _setup(self):
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

    def build_init_time_list(self) -> list[datetime]:
        """
        Builds a list of initial time for evaluation.

        This function make sure all the initial time are valid during entire showcase length.
        And also make sure their forecast time exists.

        Returns:
            list[datetime]: A sorted list of initial times for evaluation.

        Raises:
            ValueError: If the sanity check fails for the initial time or forecast time.

        """
        if self.eval_cases is None:
            warnings.warn(
                "No custom eval cases are provided, using default EVAL_CASES defined in `src.const`.",
                UserWarning,
            )
            return None

        data_itv = timedelta(**self.cfg.data.time_interval)
        output_itv = timedelta(**self.cfg.inference.output_itv)
        showcase_length = self.cfg.plot.figure_columns
        init_time_list = set()
        for eval_case in self.eval_cases:
            for i in range(showcase_length):
                init_time = eval_case + data_itv * i
                fcst_time = init_time + output_itv

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
