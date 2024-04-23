from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.const import EVAL_CASES
from src.utils import DataCompose, TimeUtil, gen_path


class DatetimeManager:
    def __init__(
        self,
        start_time: str,
        end_time: str,
        interval: dict[str, int],
        format: str = "%Y_%m_%d_%H_%M",
    ):
        self.start_time = datetime.strptime(start_time, format)
        self.end_time = datetime.strptime(end_time, format)
        self.interval = timedelta(**interval)
        self.log = logging.getLogger(__name__)

        # internal property
        self.path_dict: dict[datetime, Path] = dict()
        self.train_time: list[datetime] = list()
        self.valid_time: list[datetime] = list()
        self.test_time: list[datetime] = list()

    def build_path_list(self) -> DatetimeManager:
        """
        Builds a list of parent directories from the start time to the end time.

        Args:
            None

        Returns:
            list[Path]: A list of parent directories that exist between the start and end time.
        """
        current_time = self.start_time
        while current_time <= self.end_time:
            current_parent_dir = gen_path(current_time)
            if current_parent_dir.exists():
                self.path_dict[current_time] = current_parent_dir
            current_time += self.interval
        return self

    def random_split(
        self, order_by_time: bool, ratios: list[float | int]
    ) -> DatetimeManager:
        """
        Splits the data into training, validation, and testing sets randomly or sequentially.

        Two split strategies:
        1. sequentially sample (order by time)
        2. random shuffle (not order by time)

        Args:
            order_by_time (bool): If True, the data is split sequentially based on time. If False, the data is shuffled randomly.
            ratios (list[float | int]): A list of three values representing the ratios of the data to be allocated for training, validation, and testing respectively.

        Returns:
            DatetimeManager: The updated DatetimeManager object with the split data.
        """
        assert (
            len(ratios) == 3
        ), f"ratios should be [train_r, valid_r, test_r], but {ratios}"
        # summation = 1
        ratios = np.array(ratios) / np.array(ratios).sum()

        time_list_array = np.array(list(self.path_dict.keys()))
        if order_by_time:
            ratios = np.round(ratios * 10).astype(int)
            chunk_size = ratios.sum()

            for i in range(chunk_size):
                tmp = time_list_array[i::chunk_size]
                if i < ratios[0]:
                    self.train_time.extend(list(tmp))
                elif i >= chunk_size - ratios[-1]:
                    self.test_time.extend(list(tmp))
                else:
                    self.valid_time.extend(list(tmp))

            # ==================================================
            # DON'T sort the time list, or the `AdoptedDataset`
            # will sample those data in the front forever (bias).
            #
            # self.train_time.sort()
            # self.valie_time.sort()
            # self.test_time.sort()
            # ==================================================
        else:
            random.seed(1000)
            random.shuffle(time_list_array)
            ratios *= len(time_list_array)
            for category, category_idx in {"train": 0, "valid": 1, "test": 2}.items():
                start_idx = np.sum(ratios[:category_idx], dtype=int)
                end_idx = np.sum(ratios[: category_idx + 1], dtype=int)
                self.__setattr__(
                    f"{category}_time", list(time_list_array[start_idx:end_idx])
                )
        return self

    def remove_evaluation_cases(self) -> DatetimeManager:
        for dt in EVAL_CASES:
            pass
        return self

    def sanity_check(self, data_list: list[DataCompose]) -> DatetimeManager:
        """
        Check the sanity of the path dictionary by verifying the existence of sub-directory paths for each key.

        Parameters:
            data_list (list[DataCompose]): A list of DataCompose objects representing the data.

        Returns:
            DatetimeManager: The updated DatetimeManager object with the removed keys from the path dictionary.
        """
        cnt = 0
        for key, _ in self.path_dict.items():
            sub_dir_paths = [gen_path(key, data) for data in data_list]
            if not all([path.exists() for path in sub_dir_paths]):
                self.path_dict.pop(key)
                cnt += 1
        self.log.info(f"Removed {cnt} keys during sanity check.")
        return self
