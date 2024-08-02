import abc
import warnings
from datetime import datetime, timedelta

from omegaconf import DictConfig

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
            **self.cfg.inference,
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

    def showcase_init_time_list(self, case_dt: datetime) -> list[datetime]:
        """
        If adding additional initial times to `init_time` list, override this function.
        """
        return [case_dt]

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
