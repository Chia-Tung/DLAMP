from datetime import datetime, timedelta


class TimeUtil:
    @staticmethod
    def entire_period(
        year: int,
        month: int,
        day: int,
        hour: int | None = None,
        interval: dict[str, int] = {"minutes": 1},
    ) -> list[datetime]:
        """
        Generate a list of datetime objects representing the entire period for a given date and time interval.

        Parameters:
            year (int): The year of the target date.
            month (int): The month of the target date.
            day (int): The day of the target date.
            hour (int | None, optional): The hour of the target date. Defaults to None.
            interval (dict[str, int], optional): The time interval used to iterate through the target dates.
                The keys of the dictionary must be one of the following: "days", "seconds", "microseconds",
                "milliseconds", "minutes", "hours", "weeks". Defaults to {"minutes": 1}.

        Returns:
            list[datetime]: A list of datetime objects representing the entire period for the given date and time interval.
        """
        time_interval = timedelta(**interval)
        time_list = []
        if hour:
            dt = datetime(year, month, day, hour)
            while dt.hour == hour:
                time_list.append(dt)
                dt += time_interval
        else:
            dt = datetime(year, month, day)
            while dt.day == day:
                time_list.append(dt)
                dt += time_interval
        return time_list

    @staticmethod
    def three_days(
        year: int, month: int, day: int, interval: dict[str, int]
    ) -> list[datetime]:
        """
        Return a list of datetime objects representing three days with the specified time interval.

        Parameters
        ----------
        year : int
            The year of the target date.
        month : int
            The month of the target date.
        day : int
            The day of the target date.
        interval : dict[str, int]
            The time interval used to iterate through the target days.
            The keys of the dictionary must be one of the following:
            "days", "seconds", "microseconds", "milliseconds",
            "minutes", "hours", "weeks".

        Returns
        -------
        list[datetime]
            A list of datetime objects representing the three days with the specified time interval.
        """
        target_t = [
            datetime(year, month, day) + i * timedelta(days=1) for i in range(-1, 2)
        ]
        time_list = []
        for calendar in target_t:
            time_list.extend(
                TimeUtil.entire_period(
                    calendar.year, calendar.month, calendar.day, interval=interval
                )
            )
        return time_list
