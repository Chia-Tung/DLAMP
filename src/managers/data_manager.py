import logging

import lightning as L
from torch.utils.data import DataLoader

from src.managers.datetime_manager import DatetimeManager
from src.utils import DataCompose


class DataManager(L.LightningDataModule):
    def __init__(self, data_list: list[DataCompose], **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["data_list"])

        # internal property
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

        # set up
        self.log = logging.getLogger(__name__)
        self.dtm = DatetimeManager(
            kwargs["start_time"], kwargs["end_time"], kwargs["time_interval"]
        )
        self.setup()

    def setup(self):
        """
        `setup` is called from every process across all the nodes.
        Setting state here is recommended.
        """
        self.log.debug(len(self.dtm.build_path_list()))
        pass

    # def _setup(self):
    #     # data loaders instantiate
    #     self._all_loaders = LoaderMapping.get_all_loaders(self._data_meta_info)

    #     # set initial time list
    #     for loader in self._all_loaders:
    #         self._datetime_maneger.import_time_from_loader(
    #             loader, self._ilen, self._olen, self._oint
    #         )

    #     # remove illegal datetime
    #     self._datetime_maneger.remove_illegal_time(self._start_date, self._end_date)

    #     # random split
    #     self._datetime_maneger.random_split(self._order_by_time, self._ratios)
    #     train_time, valid_time, test_time = (
    #         self._datetime_maneger.train_time,
    #         self._datetime_maneger.valid_time,
    #         self._datetime_maneger.test_time,
    #     )

    #     print(
    #         f"[{self.__class__.__name__}] "
    #         f"Total data collected: {len(train_time)+len(valid_time)+len(test_time)}, "
    #         f"Sampling Rate: {self._sampling_rate}\n"
    #         f"[{self.__class__.__name__}] "
    #         f"Training Data Size: {len(train_time) // self._sampling_rate}, "
    #         f"Validating Data Size: {len(valid_time) // self._sampling_rate}, "
    #         f"Testing Data Size: {len(test_time) // self._sampling_rate} \n"
    #         f"[{self.__class__.__name__}] "
    #         f"Image Shape: {self._target_shape}, Batch Size: {self._batch_size}"
    #     )

    #     self._train_dataset = AdoptedDataset(
    #         self._ilen,
    #         self._olen,
    #         self._oint,
    #         self._target_shape,
    #         self._target_lat,
    #         self._target_lon,
    #         initial_time_list=train_time,
    #         data_meta_info=self._data_meta_info,
    #         sampling_rate=self._sampling_rate,
    #         is_train=True,
    #     )

    #     self._valid_dataset = AdoptedDataset(
    #         self._ilen,
    #         self._olen,
    #         self._oint,
    #         self._target_shape,
    #         self._target_lat,
    #         self._target_lon,
    #         initial_time_list=valid_time,
    #         data_meta_info=self._data_meta_info,
    #         sampling_rate=self._sampling_rate,
    #         is_valid=True,
    #     )

    #     self._evalu_dataset = AdoptedDataset(
    #         self._ilen,
    #         self._olen,
    #         self._oint,
    #         self._target_shape,
    #         self._target_lat,
    #         self._target_lon,
    #         initial_time_list=test_time,
    #         data_meta_info=self._data_meta_info,
    #         sampling_rate=self._sampling_rate,
    #         is_test=True,
    #     )

    # def train_dataloader(self):
    #     return DataLoader(
    #         self._train_dataset,
    #         batch_size=self._batch_size,
    #         num_workers=self._workers,
    #         shuffle=True,
    #     )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self._valid_dataset,
    #         batch_size=self._batch_size,
    #         num_workers=self._workers,
    #         shuffle=False,
    #     )

    # def get_data_info(self):
    #     inp_data_map = self._train_dataset[0][0]
    #     return {
    #         "batch_size": self._batch_size,
    #         "shape": self._target_shape,
    #         "channel": {k: v.shape[1] for k, v in inp_data_map.items()},
    #         "ilen": self._ilen,
    #         "olen": self._olen,
    #     }
