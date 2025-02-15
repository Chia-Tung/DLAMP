from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..standardization import standardization
from ..utils import DataCompose, DataGenerator, Level, TimeUtil


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
        add_time_features: bool,
        use_Kth_hour_pred: int | None,
        is_train_or_valid: bool,
    ):
        super().__init__()
        self._ilen = inp_len
        self._olen = oup_len
        self._oitv = timedelta(**oup_itv)
        self._data_gnrt = data_generator
        self._sr = sampling_rate
        self._init_time_list = init_time_list
        self._data_list = data_list
        self.add_time_features = add_time_features
        self.use_Kth_hour_pred = use_Kth_hour_pred
        self._is_train_or_valid = is_train_or_valid

    def __len__(self):
        """
        Returns the length of the `_init_time_list` attribute, which represents
        the number of items in the dataset.
        """
        return (
            len(self._init_time_list) // self._sr
            if self._is_train_or_valid
            else len(self._init_time_list)
        )

    def __getitem__(self, index):
        """
        Retrieves input and output data based on the given index.

        Parameters:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the input and output data.
        """
        if self._is_train_or_valid:
            index *= self._sr
        input_time = self._init_time_list[index]
        input = self._get_variables_from_dt(input_time, is_input=True)

        output_time = input_time + self._oitv
        output = self._get_variables_from_dt(output_time, is_input=False)

        return input, output

    def _get_variables_from_dt(
        self, dt: datetime, is_input: bool
    ) -> dict[str, np.ndarray]:
        """
        Retrieves data from a given datetime object.

        Parameters:
            dt (datetime): The datetime object to retrieve variables from.
            is_input: If True, it's possible to prepare the datetime features.

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
        data_dict = self._data_gnrt.yield_data(
            dt, self._data_list, use_Kth_hour_pred=self.use_Kth_hour_pred
        )
        for var_level_str, data in data_dict.items():
            data = standardization(var_level_str, data)
            _, level = DataCompose.retrive_var_level_from_string(var_level_str)
            if level.is_surface():
                pre_output[Level.Surface].append(data)
            else:
                pre_output[level].append(data)

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
            stack_data = np.stack(value, axis=0)  # (lv, h, w, c)

            if is_input and key == "surface" and self.add_time_features:
                # add DoY and ToD (1, h, w, c+4)
                time_features = TimeUtil.create_time_features(dt, stack_data.shape[1:3])
                stack_data = np.concatenate([stack_data, time_features[None]], axis=-1)

            final[key] = stack_data  # {'upper_air': (lv, h, w, c), ...}

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
        return idx // self._sr if self._is_train_or_valid else idx

    def average_pooling(
        self, data: np.ndarray, kernel_size: int = 9, stride: int = 1
    ) -> np.ndarray:
        """
        Applies average pooling to the input data.

        Args:
            data (np.ndarray): The input data to be pooled with shape (lv, h, w, c).
            kernel_size (int, optional): The kernel size for average pooling. Defaults to 9.
            stride (int, optional): The stride for average pooling. Defaults to 1.

        Returns:
            np.ndarray: The pooled data.
        """
        # Convert to PyTorch tensor
        tensor_data = torch.from_numpy(data).float()

        # Reshape to (lv*c, 1, h, w) for avg_pool2d
        lv, h, w, c = tensor_data.shape
        tensor_data = (
            tensor_data.permute(0, 3, 1, 2).contiguous().reshape(lv * c, 1, h, w)
        )

        # Apply average pooling
        pooled_data = F.avg_pool2d(
            tensor_data,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            count_include_pad=False,
        )

        # Reshape back to (lv, h, w, c)
        pooled_data = (
            torch.reshape(pooled_data, (lv, c, h, w)).permute(0, 2, 3, 1).contiguous()
        )

        return pooled_data.numpy()
