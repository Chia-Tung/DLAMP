from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from torch.utils.data import Dataset

from ..utils import DataCompose, DataGenerator


class CustomDataset(Dataset):
    def __init__(
        self,
        inp_len: int,
        oup_len: int,
        oup_itv: dict[str, int],
        data_generator: DataGenerator,
        sampling_rate: int,
        init_time_list: list[datetime],
        data_list: list[DataCompose],
        is_train: bool,
    ):
        super().__init__()
        self._ilen = inp_len
        self._olen = oup_len
        self._oitv = timedelta(**oup_itv)
        self._data_gnrt = data_generator
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

    def _get_variables_from_dt(self, dt: datetime) -> dict[str, np.ndarray]:
        """
        Retrieves data from a given datetime object.

        Parameters:
            dt (datetime): The datetime object to retrieve variables from.

        Returns:
            dict: A dictionary containing the variables retrieved from the datetime object.
                The dictionary has the following structure:
                {
                    'upper_air': numpy.ndarray (z, h, w, c),
                    'surface': numpy.ndarray (z, h, w, c)
                }
                Each key in the dictionary corresponds to levels of variables, and the values
                are numpy arrays containing the variables stacked along the specified axis.
        """
        pre_output = defaultdict(list)
        # via traversing data_list, the levels/vars are in the the same order as the
        # order in `config/data/data_config.yaml`
        for data_compose in self._data_list:
            data = self._data_gnrt.yield_data(dt, data_compose, to_numpy=True)
            pre_output[data_compose.level].append(data)

        # concatenate by variable, group by level
        output = defaultdict(list)
        for level, value in pre_output.items():
            value = np.stack(value, axis=-1)  # (h, w, c)

            if level.is_surface():
                output["surface"].append(value)
            else:
                output["upper_air"].append(value)

        # concatenate by level
        # Warning: LightningModule doesn't support defaultdict as input/output
        final = {}
        for key, value in output.items():
            final[key] = np.stack(value, axis=0)  # {'upper_air': (lv, h, w, c), ...}

        return final

    def get_internal_index_from_dt(self, dt: datetime) -> int:
        """
        Given a datetime, this function returns the index.

        If `_is_train` is True, the function returns the index directly.
        Otherwise, it returns the index divided by `_sr`.

        Parameters:
            dt (datetime): The datetime object to find the index for.

        Returns:
            int: The index of the datetime object in the `_init_time_list` attribute.
        """
        idx = self._init_time_list.index(dt)
        return idx if self._is_train else idx // self._sr
