from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from torch.utils.data import Dataset

from .data_compose import DataCompose
from .file_util import gen_path, read_cwa_npfile


class CustomDataset(Dataset):
    def __init__(
        self,
        inp_len: int,
        oup_len: int,
        inp_itv: dict[str, int],
        oup_itv: dict[str, int],
        data_shape: list[int],
        sampling_rate: int,
        init_time_list: list[datetime],
        data_list: list[DataCompose],
        is_train: bool,
    ):
        super().__init__()
        self._ilen = inp_len
        self._olen = oup_len
        self._iitv = timedelta(**inp_itv)
        self._oitv = timedelta(**oup_itv)
        self._raw_shape = tuple(data_shape)
        self._sr = sampling_rate
        self._init_time_list = init_time_list
        self._data_list = data_list
        self._is_train = is_train

    def __len__(self):
        """
        Returns the length of the `_init_time_list` attribute, which represents
        the number of items in the dataset.
        """
        return (
            len(self._init_time_list)
            if self._is_train
            else len(self._init_time_list) // self._sr
        )

    def __getitem__(self, index):
        """
        Retrieves input and output data based on the given index.

        Parameters:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the input and output data.
        """
        if not self._is_train:
            index *= self._sr
        input_time = self._init_time_list[index]
        input = self._get_variables_from_dt(input_time)

        output_time = input_time + self._oitv
        output = self._get_variables_from_dt(output_time)

        return input, output

    def _get_variables_from_dt(self, dt: datetime):
        """
        Retrieves data from a given datetime object.

        Parameters:
            dt (datetime): The datetime object to retrieve variables from.

        Returns:
            dict: A dictionary containing the variables retrieved from the datetime object.
                The dictionary has the following structure:
                {
                    'upper_air': numpy.ndarray,
                    'surface': numpy.ndarray
                }
                Each key in the dictionary corresponds to levels of variables, and the values
                are numpy arrays containing the variables stacked along the specified axis.
        """
        pre_output = defaultdict(list)
        for data_compose in self._data_list:
            sub_dir_path = gen_path(dt, data_compose)
            data = read_cwa_npfile(sub_dir_path, data_compose.is_radar)
            pre_output[data_compose.level.value].append(data)

        # sort by level, ascending order: 200, 300, 500, 700, 850, 925, H00
        pre_output = dict(sorted(pre_output.items()))

        # stack by level
        output = defaultdict(list)
        for k, v in pre_output.items():
            tmp = np.stack(v, axis=-1)  # (h, w, c)

            if k in ["200", "300", "500", "700", "850", "925", "H00"]:
                output["upper_air"].append(tmp)
            else:
                output["surface"].append(tmp)

        # integration
        for k, v in output.items():
            output[k] = np.stack(v, axis=0)  # {'upper_air': (lv, h, w, c), ...}
            self._shape_check(dt, output[k])

        return output

    def _shape_check(self, target_dt: datetime, data: np.ndarray):
        """
        Check if the shape of the given data matches the expected shape.

        Args:
            target_dt (datetime): The target datetime for which the data is being checked.
            data (np.ndarray): The data array to be checked.

        Raises:
            AssertionError: If the shape of the data does not match the expected shape.

        Returns:
            None
        """
        assert (
            data.shape[-3:-1] == self._raw_shape
        ), f"{target_dt} data shape mismatch: {data.shape} != {self._raw_shape}"
