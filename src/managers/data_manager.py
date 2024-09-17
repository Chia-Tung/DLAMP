import logging
from datetime import datetime

import lightning as L
from torch.utils.data import DataLoader

from ..datasets import CustomDataset
from ..utils import DataCompose, DataGenerator
from .datetime_manager import DatetimeManager

log = logging.getLogger(__name__)


class DataManager(L.LightningDataModule):
    def __init__(
        self,
        data_list: list[DataCompose],
        init_time_list: list[datetime] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data_list", "init_time_list", "train_data"])

        # internal property
        self.data_list = data_list
        self.init_time_list = init_time_list
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None
        self._predict_dataset = None

        # assistants
        self.dtm = DatetimeManager(
            kwargs["start_time"],
            kwargs["end_time"],
            kwargs["format"],
            kwargs["time_interval"],
        )
        self.data_gnrt = DataGenerator(kwargs["data_shape"], kwargs["image_shape"])

        # flags
        self._already_called: dict[str, bool] = {}
        for stage in ("fit", "validate", "test", "predict"):
            self._already_called[stage] = False

    def setup(self, stage: str):
        """
        Sets up the data for the specified stage.

        This function is called from every process across all the nodes and GPUs.

        Parameters:
            stage (str): The stage for which the data needs to be set up.
                Possible values are "fit", "validate", "test", or "predict".

        Returns:
            None

        Raises:
            NotImplementedError: If the stage is "predict".
            ValueError: If the stage is invalid.
        """
        if self._already_called[stage]:
            log.warning(f'Stage "{stage}" has already been called. Skipping...')
            return

        if not self.dtm.is_done:
            self.dtm.build_initial_time_list(self.data_list).random_split(
                **self.hparams.split_config
            ).build_eval_cases().swap_eval_cases_from_train_valid()
            self.dtm.is_done = True

        match stage:
            case "fit":
                self._train_dataset = self._setup("train")
                self._valid_dataset = self._setup("valid")
            case "validate":
                self._valid_dataset = self._setup("valid")
            case "test":
                self._test_dataset = self._setup("test")
            case "predict":
                self._predict_dataset = self._setup("predict")
            case _:
                log.error(f"Invalid stage: {stage}")
                raise ValueError(f"Invalid stage: {stage}")

        self._already_called[stage] = True
        self.info_log(f'Stage "{stage}" setup done')
        self.info_log(
            f"Total data collected: {len(self.dtm.time_list)}, Sampling Rate: {self.hparams.sampling_rate}"
        )
        self.info_log(
            f"Training Data Size: {len(self.dtm.train_time) // self.hparams.sampling_rate}, "
            f"Validating Data Size: {len(self.dtm.valid_time) // self.hparams.sampling_rate}, "
            f"Testing Data Size: {len(self.dtm.test_time) // self.hparams.sampling_rate}"
        )
        self.info_log(
            f"Data Shape: {self.hparams.data_shape}, Batch Size: {self.hparams.batch_size}"
        )

    def _setup(self, stage: str):
        """
        Sets up the `torch.utils.data.Dataset` for the specified stage.

        Parameters:
            stage (str): The stage for which the data needs to be set up.
                Possible values are "train", "valid", "test", or "predict".

        Returns:
            CustomDataset: The subclass of `torch.utils.data.Dataset`.
        """
        ordered_time = (
            sorted(self.init_time_list)
            if stage in ["test", "predict"] and self.init_time_list is not None
            else getattr(self.dtm, f"ordered_{stage}_time")
        )

        return CustomDataset(
            self.hparams.input_len,
            self.hparams.output_len,
            getattr(self.hparams, "output_itv", {"hours": 1}),
            self.data_gnrt,
            self.hparams.sampling_rate,
            ordered_time,
            self.data_list,
            is_train_or_valid=stage in ["train", "valid"],
        )

    def train_dataloader(self):
        """
        sampler and shuffle can not exist at the same time
        """
        return DataLoader(
            dataset=self._train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            drop_last=False,
        )

    def val_dataloader(self):
        """
        sampler and shuffle can not exist at the same time
        """
        return DataLoader(
            dataset=self._valid_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            drop_last=False,
        )

    def test_dataloader(self):
        """
        sampler and shuffle can not exist at the same time
        """
        return DataLoader(
            dataset=self._test_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            drop_last=False,
        )

    def predict_dataloader(self):
        """
        sampler and shuffle can not exist at the same time
        """
        return DataLoader(
            dataset=self._predict_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            drop_last=False,
        )

    def info_log(self, content: str):
        """
        Logs the given content with the class name prefixed.
        """
        log.info(f"[{self.__class__.__name__}] {content}")
