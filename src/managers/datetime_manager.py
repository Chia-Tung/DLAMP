from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..const import BLACKLIST_PATH, EVAL_CASES
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
        self.format = format

        # internal property
        self.time_list: list[datetime] = list()
        self.train_time: set[datetime] = set()
        self.valid_time: set[datetime] = set()
        self.test_time: set[datetime] = set()
        self.blacklist: set[datetime] = set()
        self._done = False

    def build_initial_time_list(self, data_list: list[DataCompose]) -> DatetimeManager:
        if not Path(BLACKLIST_PATH).exists():
            self._build_init_time_list(data_list, save_output=True)
        else:
            self._quick_build_init_time_list()
        return self

    def _build_init_time_list(
        self, data_list: list[DataCompose], save_output: bool
    ) -> None:
        """
        Builds initial time list based on the start time and end time. Eliminates datetime which
        does not meet the sanity check for both current time and next time.

        Args:
            data_list (list[DataCompose]): The list of DataCompose objects.
            save_output (bool): Whether to save the blacklist of initial time to a file.

        Returns:
            None
        """
        s = time.time()
        remove = []
        skip_current: bool = False
        num_time = int((self.end_time - self.start_time) / self.interval) + 1
        pbar = tqdm(total=num_time, desc="Building initial time list...")

        current_time = self.start_time
        pbar.update(1)
        while current_time < self.end_time:
            next_time = current_time + self.interval

            if not self._sanity_check(next_time, data_list):
                remove.extend([current_time, next_time])
                current_time += 2 * self.interval
                skip_current = False
                pbar.update(2)
                continue

            # skip checking current time if `skip_current = True`
            if not skip_current and not self._sanity_check(current_time, data_list):
                remove.append(current_time)
                current_time += self.interval
                skip_current = True
                pbar.update(1)
                continue

            self.time_list.append(current_time)
            current_time += self.interval
            skip_current = True
            pbar.update(1)
        pbar.close()

        if save_output:
            with open(BLACKLIST_PATH, "w") as f:
                for dt in remove:
                    f.write(dt.strftime(self.format) + "\n")

        log.info(f"Removed {len(remove)} datetimes during data sanity check.")
        log.debug(f"{self.BC} Built initial time list in {time.time() - s:.5f} sec.")

    def _quick_build_init_time_list(self) -> None:
        """
        Builds the initial time list based on the start and end times. Datetimes that are in the 
        blacklist are excluded from the time list. 
        """
        s = time.time()

        blacklist = []
        with open(BLACKLIST_PATH, "r") as f:
            for line in f:
                dt = datetime.strptime(line.strip(), self.format)
                blacklist.append(dt)

        num_time = int((self.end_time - self.start_time) / self.interval) + 1
        with tqdm(total=num_time) as pbar:
            pbar.set_description("Building initial time list...")
            current_time = self.start_time
            pbar.update(1)
            while current_time < self.end_time:
                if current_time not in blacklist:
                    self.time_list.append(current_time)
                current_time += self.interval
                pbar.update(1)

        log.info(f"Removed {len(blacklist)} datetimes during data sanity check.")
        log.debug(f"{self.BC} Built initial time list in {time.time() - s:.5f} sec.")

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

        log.debug(f"{self.BC} Split data in {time.time() - s:.5f} sec.")
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

        log.debug(f"{self.BC} Built blacklist in {time.time() - s:.5f} sec.")
        log.debug(f"Blacklist size: {len(self.blacklist)}")
        return self

    def _sanity_check(self, dt: datetime, data_list: list[DataCompose]) -> bool:
        """
        Check the sanity of the parent directory (../rwrf/rwf_201706/2017060100000000) regarding
        dt by verifying the existence of all target data files under this parent directory.

        Parameters:
            dt (datetime): The target parent directory to check
            data_list (list[DataCompose]): A list of DataCompose objects representing the data.

        Returns:
            bool: True if all target data files exist, False otherwise.
        """
        data_filename_generator = (gen_path(dt, data) for data in data_list)
        while True:
            try:
                data_filename = next(data_filename_generator)
                if not data_filename.exists():
                    return False
            except StopIteration:
                return True

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
        log.debug(f"{self.BC} Swapped eval cases in {time.time() - s:.5f} sec.")
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
