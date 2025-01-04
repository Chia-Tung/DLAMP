from datetime import datetime, timedelta

import numpy as np


class TimeUtil:
    @staticmethod
    def entire_period(
        year: int,
        month: int,
        day: int,
        hour: int | None = None,
        interval: dict[str, int] | timedelta = {"minutes": 1},
    ) -> list[datetime]:
        """
        Generate a list of datetime objects representing the entire period for a given date and time interval.

        Parameters:
            year (int): The year of the target date.
            month (int): The month of the target date.
            day (int): The day of the target date.
            hour (int | None, optional): The hour of the target date. Defaults to None.
            interval (dict[str, int] | timedelta, optional): The time interval used to iterate through the target dates.
                The keys of the dictionary must be one of the following: "days", "seconds", "microseconds",
                "milliseconds", "minutes", "hours", "weeks". Defaults to {"minutes": 1}.

        Returns:
            list[datetime]: A list of datetime objects representing the entire period for the given date and time interval.
        """
        if isinstance(interval, dict):
            interval = timedelta(**interval)
        time_list = []
        if hour:
            dt = datetime(year, month, day, hour)
            while dt.hour == hour:
                time_list.append(dt)
                dt += interval
        else:
            dt = datetime(year, month, day)
            while dt.day == day:
                time_list.append(dt)
                dt += interval
        return time_list

    @staticmethod
    def N_days_time_list(
        year: int,
        month: int,
        day: int,
        interval: dict[str, int] | timedelta,
        n_days: int,
    ) -> list[datetime]:
        """
        Generate a list of datetime objects representing the three days before and after a given date.

        Parameters:
            year (int): The year of the target date.
            month (int): The month of the target date.
            day (int): The day of the target date.
            interval (dict[str, int] | timedelta): The time interval used to iterate through the target dates.
                The keys of the dictionary must be one of the following: "days", "seconds", "microseconds",
                "milliseconds", "minutes", "hours", "weeks". Defaults to {"minutes": 1}.

        Returns:
            list[datetime]: A list of datetime objects representing the three days before and after the given date.
        """
        assert n_days >= 1, f"n_days must be a positive integer but get {n_days}"
        half_range = n_days // 2
        start = -half_range if n_days > 1 else 0
        end = half_range + 1 if n_days % 2 == 1 else half_range

        target_t = [
            datetime(year, month, day) + i * timedelta(days=1)
            for i in range(start, end)
        ]

        time_list = []
        for calendar in target_t:
            time_list.extend(
                TimeUtil.entire_period(
                    calendar.year, calendar.month, calendar.day, interval=interval
                )
            )
        return time_list

    @staticmethod
    def create_time_features(
        curr_dt: datetime, array_shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Computes the sine and cosine transformations of the Day of Year (DoY) and Time
        of Day (ToD) from the provided datetime object and returns arrays of the specified
        shape filled with these values.

        Parameters:
        - curr_dt (datetime): The datetime object for which to compute DoY and ToD.
        - array_shape (tuple): The desired 2D shape of the output arrays.

        Returns:
        - time_features (np.ndarray): Array of shape (**array_shape, 4) filled with the
            computed DoY and ToD values.
        """
        assert len(array_shape) == 2, "array_shape must be a tuple of length 2"

        # Calculate Day of Year (DoY)
        doy = curr_dt.timetuple().tm_yday

        # Calculate Time of Day (ToD) in hours
        tod = curr_dt.hour + curr_dt.minute / 60 + curr_dt.second / 3600

        # Compute sine and cosine transformations for DoY
        doy_sin = np.sin(2 * np.pi * doy / 365.0)
        doy_cos = np.cos(2 * np.pi * doy / 365.0)

        # Compute sine and cosine transformations for ToD
        tod_sin = np.sin(2 * np.pi * tod / 24.0)
        tod_cos = np.cos(2 * np.pi * tod / 24.0)

        # Create arrays filled with the computed values
        doy_sin_array = np.full(array_shape, doy_sin, dtype=np.float32)
        doy_cos_array = np.full(array_shape, doy_cos, dtype=np.float32)
        tod_sin_array = np.full(array_shape, tod_sin, dtype=np.float32)
        tod_cos_array = np.full(array_shape, tod_cos, dtype=np.float32)

        # stack the arrays (H, W, 4)
        time_features = np.stack(
            [doy_sin_array, doy_cos_array, tod_sin_array, tod_cos_array], axis=-1
        )

        return time_features
