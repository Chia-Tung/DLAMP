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
        self.train_time: set[datetime] = set()
        self.valid_time: set[datetime] = set()
        self.test_time: set[datetime] = set()
        self.blacklist: set[datetime] = set()

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

        time_list_array = list(self.path_dict.keys())
        if order_by_time:
            ratios = np.round(ratios * 10).astype(int)
            chunk_size = ratios.sum()
            time_list_array = np.array(time_list_array)
            for i in range(chunk_size):
                tmp = time_list_array[i::chunk_size]
                if i < ratios[0]:
                    self.train_time.update(tmp)
                elif i >= chunk_size - ratios[-1]:
                    self.test_time.update(tmp)
                else:
                    self.valid_time.update(tmp)

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
                    f"{category}_time", set(time_list_array[start_idx:end_idx])
                )

        self.log.debug(f"train_time size (original): {len(self.train_time)}")
        self.log.debug(f"valid_time size (original): {len(self.valid_time)}")
        self.log.debug(f"test_time size (original): {len(self.test_time)}")
        return self

    def build_blacklist(self) -> DatetimeManager:
        """
        Remove evaluation cases from the training set.

        Evaluation cases are defined in `src.const.EVAL_CASES`.
        These cases are blacklisted from the training set.

        Returns:
            DatetimeManager: The updated DatetimeManager object with the evaluation cases removed.
        """

        def get_datetime_list(dt_list, fn) -> list[datetime]:
            """
            Get a list of datetimes by applying a function to each element of a list of datetimes.

            Parameters:
                dt_list (list[datetime]): A list of datetimes.
                fn (Callable[[int, int, int], datetime]): A function that takes year, month, and
                    day as input and returns a datetime.

            Returns:
                list[datetime]: A list of datetimes.
            """
            ret = []
            for dt in dt_list:
                ret.extend(fn(dt.year, dt.month, dt.day, interval=self.interval))
            return ret

        for key, value in EVAL_CASES.items():
            if key == "one_day":
                self.blacklist |= set(get_datetime_list(value, TimeUtil.entire_period))
            elif key == "three_days":
                self.blacklist |= set(get_datetime_list(value, TimeUtil.three_days))

        self.log.debug(f"Blacklist size: {len(self.blacklist)}")
        return self

    def sanity_check(self, data_list: list[DataCompose]) -> DatetimeManager:
        """
        Check the sanity of the path dictionary by verifying the existence of sub-directory paths for each key.

        Parameters:
            data_list (list[DataCompose]): A list of DataCompose objects representing the data.

        Returns:
            DatetimeManager: The updated DatetimeManager object with the removed keys from the path dictionary.
        """
        keys_to_remove = []
        for key, _ in self.path_dict.items():
            sub_dir_paths = [gen_path(key, data) for data in data_list]
            if not all([path.exists() for path in sub_dir_paths]):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.path_dict[key]

        self.log.info(f"Removed {len(keys_to_remove)} keys during sanity check.")
        return self

    def swap_eval_cases_from_train_valid(self) -> DatetimeManager:
        """
        Swaps evaluation cases from the train and valid sets.

        Returns:
            DatetimeManager: The updated DatetimeManager object after swapping the evaluation cases.
        """

        def fn(name: str) -> None:
            dataset = getattr(self, f"{name}_time")
            clashes = dataset & self.blacklist

            for dt in clashes:
                dataset.remove(dt)
                self.test_time.add(dt)
                while True:
                    swap_dt = random.choice(list(self.test_time))
                    if swap_dt not in self.blacklist:
                        self.test_time.remove(swap_dt)
                        dataset.add(swap_dt)
                        break

            self.log.info(f"Swapped {len(clashes)} eval cases from {name} to test.")

        fn("train")
        fn("valid")

        return self
