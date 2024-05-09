from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

from ..const import EVAL_CASES
from ..utils import DataCompose, TimeUtil, gen_path

log = logging.getLogger(__name__)


class DatetimeManager:
    BC = "[Bottleneck Check]"

    def __init__(
        self,
        start_time: str,
        end_time: str,
        format: str,
        interval: dict[str, int],
    ):
        self.start_time = datetime.strptime(start_time, format)
        self.end_time = datetime.strptime(end_time, format)
        self.interval = timedelta(**interval)

        # internal property
        self.time_list: list[datetime] = list()
        self.train_time: set[datetime] = set()
        self.valid_time: set[datetime] = set()
        self.test_time: set[datetime] = set()
        self.blacklist: set[datetime] = set()
        self._done = False

    def build_path_list(self) -> DatetimeManager:
        """
        Builds a list of parent directories between the start time and end time,
        with intervals specified by the `interval` attribute. Both current parent
        directory and next parent directory must exist.

        Returns:
            DatetimeManager: The updated DatetimeManager object with the built path list.
        """
        s = time.time()
        current_time = self.start_time
        current_parent_dir = gen_path(current_time)
        while current_time < self.end_time:
            next_parent_dir = gen_path(current_time + self.interval)
            if current_parent_dir.exists() and next_parent_dir.exists():
                self.time_list.append(current_time)
            current_time += self.interval
            current_parent_dir = next_parent_dir
        log.debug(f"{self.BC} Built path list in {time.time() - s:.5f} seconds.")
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
        s = time.time()
        assert (
            len(ratios) == 3
        ), f"ratios should be [train_r, valid_r, test_r], but {ratios}"
        # summation = 1
        ratios = np.array(ratios) / np.array(ratios).sum()

        time_list_array = self.time_list
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

        log.debug(f"{self.BC} Split data in {time.time() - s:.5f} seconds.")
        log.debug(f"train_time size (original): {len(self.train_time)}")
        log.debug(f"valid_time size (original): {len(self.valid_time)}")
        log.debug(f"test_time size (original): {len(self.test_time)}")
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

        s = time.time()
        for key, value in EVAL_CASES.items():
            if key == "one_day":
                self.blacklist |= set(get_datetime_list(value, TimeUtil.entire_period))
            elif key == "three_days":
                self.blacklist |= set(get_datetime_list(value, TimeUtil.three_days))

        log.debug(f"{self.BC} Built blacklist in {time.time() - s:.5f} seconds.")
        log.debug(f"Blacklist size: {len(self.blacklist)}")
        return self

    def sanity_check(self, data_list: list[DataCompose]) -> DatetimeManager:
        """
        Check the sanity of the path dictionary by verifying the existence of sub-directory paths for each key.

        Parameters:
            data_list (list[DataCompose]): A list of DataCompose objects representing the data.

        Returns:
            DatetimeManager: The updated DatetimeManager object with the removed keys from the path dictionary.
        """
        s = time.time()
        dt_to_remove = set()
        for dt in tqdm(self.time_list, desc="Data sanity check"):
            sub_dir_generator = (gen_path(dt, data) for data in data_list)
            while True:
                try:
                    sub_dir = next(sub_dir_generator)
                    if not sub_dir.exists():
                        dt_to_remove.add(dt)
                        break
                except StopIteration:
                    break

        self.time_list = list(set(self.time_list) - dt_to_remove)

        log.debug(f"{self.BC} Sanity check in {time.time() - s:.5f} seconds.")
        log.info(f"Removed {len(dt_to_remove)} keys during sanity check.")
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

            log.info(f"Swapped {len(clashes)} eval cases from {name} to test.")

        s = time.time()
        fn("train")
        fn("valid")
        log.debug(f"{self.BC} Swapped eval cases in {time.time() - s:.5f} seconds.")
        return self

    @property
    def ordered_train_time(self) -> list[datetime]:
        return sorted(self.train_time)

    @property
    def ordered_valid_time(self) -> list[datetime]:
        return sorted(self.valid_time)

    @property
    def ordered_test_time(self) -> list[datetime]:
        return sorted(self.test_time)

    @property
    def is_done(self) -> bool:
        return self._done

    @is_done.setter
    def is_done(self, new_value: bool) -> None:
        self._done = new_value
