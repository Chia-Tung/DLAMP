from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..const import BLACKLIST_PATH, DATA_SOURCE, EVAL_CASES
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
        self.eval_cases: set[datetime] = set()
        self._done = False

    def build_initial_time_list(
        self, data_list: list[DataCompose], use_Kth_hour_pred: int | None
    ) -> DatetimeManager:
        if not Path(BLACKLIST_PATH).exists():
            self._build_init_time_list(data_list, use_Kth_hour_pred, save_output=True)
        else:
            self._quick_build_init_time_list()
        return self

    def _build_init_time_list(
        self,
        data_list: list[DataCompose],
        use_Kth_hour_pred: int | None,
        save_output: bool,
    ) -> None:
        """
        Builds initial time list based on the start time and end time. Eliminates datetime which
        does not meet the sanity check for both current time and next time.

        Args:
            data_list (list[DataCompose]): The list of DataCompose objects.
            use_Kth_hour_pred (int | None): Use Kth hour prediciton to generate the file path
                if not None. Else, use the oringal inital time.
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

            if not self.sanity_check(next_time, data_list, use_Kth_hour_pred):
                remove.extend([current_time, next_time])
                current_time += 2 * self.interval
                skip_current = False
                pbar.update(2)
                continue

            # skip checking current time if `skip_current = True`
            if not skip_current and not self.sanity_check(
                current_time, data_list, use_Kth_hour_pred
            ):
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
        self, ratios: list[float | int], split_method: str = "random"
    ) -> DatetimeManager:
        """
        Split the time list into train, validation and test sets based on specified ratios and method.

        Args:
            ratios (list[float | int]): List of 3 numbers specifying the ratio split between
                train, validation and test sets. Will be normalized to sum to 1.
            split_method (str, optional): Method to use for splitting. Options are:
                - "random": Randomly shuffle and split based on ratios
                - "sequential": Split sequentially in chunks based on ratios
                - "half_month": Split by half-month periods, e.g. 1/1-1/15, 1/16-1/30.
                Defaults to "random".

        Returns:
            DatetimeManager: Returns self for method chaining.
        """
        s = time.time()
        assert (
            len(ratios) == 3
        ), f"ratios should be [train_r, valid_r, test_r], but {ratios}"

        ratios = np.array(ratios) / np.array(ratios).sum()

        time_list = self.time_list.copy()
        if split_method == "random":
            random.seed(1000)
            random.shuffle(time_list)
            ratios *= len(time_list)
            for category, category_idx in {"train": 0, "valid": 1, "test": 2}.items():
                start_idx = np.sum(ratios[:category_idx], dtype=int)
                end_idx = np.sum(ratios[: category_idx + 1], dtype=int)
                self.__setattr__(f"{category}_time", set(time_list[start_idx:end_idx]))
        elif split_method == "sequential":
            ratios = np.round(ratios * 10).astype(int)
            chunk_size = ratios.sum()
            time_list_array = np.array(time_list)
            for i in range(chunk_size):
                tmp = time_list_array[i::chunk_size]
                if i < ratios[0]:
                    self.train_time.update(tmp)
                elif i >= chunk_size - ratios[-1]:
                    self.test_time.update(tmp)
                else:
                    self.valid_time.update(tmp)
        elif split_method == "half_month":
            # Group datetimes by half-month periods
            half_month_groups = defaultdict(list)
            for dt in time_list:
                half = "1st_half" if dt.day <= 15 else "2nd_half"
                group_key = f"{dt.strftime('%b')}_{half}"  # e.g. "Jan_1st_half"
                half_month_groups[group_key].append(dt)

            # Randomly shuffle groups
            groups = list(half_month_groups.values())
            random.seed(1000)
            random.shuffle(groups)
            num_groups = len(groups)
            train_end = int(num_groups * ratios[0])
            valid_end = int(num_groups * (ratios[0] + ratios[1]))

            # Assign groups to splits
            for i, group in enumerate(groups):
                if i < train_end:
                    self.train_time.update(group)
                elif i < valid_end:
                    self.valid_time.update(group)
                else:
                    self.test_time.update(group)

        log.debug(f"{self.BC} Split data in {time.time() - s:.5f} sec.")
        log.debug(f"train_time size (original): {len(self.train_time)}")
        log.debug(f"valid_time size (original): {len(self.valid_time)}")
        log.debug(f"test_time size (original): {len(self.test_time)}")
        return self

    def build_eval_cases(self) -> DatetimeManager:
        """
        Builds the evaluation cases from `src.const.EVAL_CASES`. Note that all evaluation cases
        must have existed in the initial time list. These cases are blacklisted from the training set.

        Returns:
            DatetimeManager: The updated DatetimeManager object with the evaluation cases removed.
        """
        days_map = {
            "one_day": 1,
            "three_days": 3,
            "five_days": 5,
            "seven_days": 7,
        }

        s = time.time()
        for key, value in EVAL_CASES.items():
            n_days = days_map.get(key)

            if n_days is None:
                raise RuntimeError(f"Invalid days: {key}")

            for dt in value:
                self.eval_cases |= set(
                    TimeUtil.N_days_time_list(
                        dt.year, dt.month, dt.day, self.interval, n_days
                    )
                )

        self.eval_cases &= set(self.time_list)
        log.debug(f"{self.BC} Built eval case list in {time.time() - s:.5f} sec.")
        log.debug(f"eval case list size: {len(self.eval_cases)}")
        return self

    @staticmethod
    def sanity_check(
        dt: datetime,
        data_list: list[DataCompose],
        use_Kth_hour_pred: int | None = None,
        data_source: str = DATA_SOURCE,
    ) -> bool:
        """
        Parameters:
            dt (datetime): The target parent directory to check.
            data_list (list[DataCompose]): A list of DataCompose objects representing the data.
            use_Kth_hour_pred (int | None): Use Kth hour prediciton to generate the file path
                if not None. Else, use the oringal inital time.
            data_source (str): The way checking the validity depends on different data sources.
                e.g.
                    "NEO171_RWRF" -> data stored on neo171 server, check files one by one
                    "CWA_RWRF" -> data stored on CWA HPC, check dataset by ncdump

        Returns:
            bool: True if all target data files exist, False otherwise.
        """
        match data_source:
            case "NEO171_RWRF":
                data_filename_generator = (gen_path(dt, data) for data in data_list)
                while True:
                    try:
                        data_filename = next(data_filename_generator)
                        if not data_filename.exists():
                            return False
                    except StopIteration:
                        return True
            case "CWA_RWRF":
                # since CWA prepared the data for us, we believe all variables are consistent
                # in every netCDF file. Thus, we only check the file existence here.
                data_filename = gen_path(dt, use_Kth_hour_pred=use_Kth_hour_pred)
                return True if data_filename.exists() else False
            case _:
                log.error(f"Invalid data_source: {data_source}")
                raise ValueError(f"Invalid data_source: {data_source}")

    def swap_eval_cases_from_train_valid(self) -> DatetimeManager:
        """
        Swaps evaluation cases from the train and valid sets.

        Returns:
            DatetimeManager: The updated DatetimeManager object after swapping the evaluation cases.
        """
        no_swap: bool = len(self.test_time) == 0

        def fn(name: str) -> None:
            dataset = getattr(self, f"{name}_time")
            clashes = dataset & self.eval_cases

            for dt in clashes:
                dataset.remove(dt)
                self.test_time.add(dt)

                if no_swap:
                    continue
                while True:
                    swap_dt = random.choice(list(self.test_time))
                    if swap_dt not in self.eval_cases:
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
    def ordered_predict_time(self) -> list[datetime]:
        return sorted(self.eval_cases)

    @property
    def is_done(self) -> bool:
        return self._done

    @is_done.setter
    def is_done(self, new_value: bool) -> None:
        self._done = new_value
