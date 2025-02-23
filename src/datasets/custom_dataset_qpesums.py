from datetime import datetime

from src.utils import DataCompose
from src.utils.data_generator import DataGenerator
from src.utils.qperr_file_util import get_data

from .custom_dataset import CustomDataset


class CustomDatasetQPESUMS(CustomDataset):
    def __init__(
        self,
        inp_len: int,
        oup_len: int,
        oup_itv: dict[str, int],
        data_generator: DataGenerator,
        sampling_rate: int,
        init_time_list: list[datetime],
        data_list: list[DataCompose],
        add_time_features: bool,
        use_Kth_hour_pred: int | None,
        is_train_or_valid: bool,
    ):
        super().__init__(
            inp_len,
            oup_len,
            oup_itv,
            data_generator,
            sampling_rate,
            init_time_list,
            data_list,
            add_time_features,
            use_Kth_hour_pred,
            is_train_or_valid,
        )

    def __getitem__(self, index):
        if self._is_train_or_valid:
            index *= self._sr
        input_time = self._init_time_list[index]
        target_time = input_time + self._oitv

        input_rwrf, target_rwrf = super().__getitem__(index)
        target_qperr = self.get_qperr_from_dt(target_time)

        return input_rwrf, target_rwrf, target_qperr

    def get_qperr_from_dt(self, dt: datetime) -> float:
        qperr = get_data(dt)  # (561, 441)
        return qperr[1:, 1:]  # (560, 440)
