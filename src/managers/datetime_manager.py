from datetime import datetime, timedelta
from pathlib import Path

from src.utils import gen_path


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

    def build_path_list(self) -> list[Path]:
        """
        Builds a list of parent directories from the start time to the end time.

        Args:
            None

        Returns:
            list[Path]: A list of parent directories that exist between the start and end time.
        """
        path_list: list[Path] = []
        current_time = self.start_time
        while current_time <= self.end_time:
            current_parent_dir = gen_path(current_time)
            if current_parent_dir.exists():
                path_list.append(current_parent_dir)
            current_time += self.interval
        return path_list
