from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np

from src.const import DATA_PATH

from . import DataCompose


def gen_data(target_time: datetime, data_compose: DataCompose):
    file_dir = gen_path(target_time, data_compose)
    return read_cwa_npfile(file_dir, data_compose.is_radar)


def read_cwa_npfile(file_path: Path, is_radar: bool = False) -> np.ndarray:
    """
    The x and y grids point of RWRF model data are 450 and 450, respectively.

    One thing needs to notice, the data type may save as little-endian with
    double precision or big-endian with double precision. User may design a
    judgment rule for checking if there are weird values after read data.
    """
    data = np.fromfile(file_path, dtype=">d", count=-1, sep="").reshape(450, 450)

    if is_radar and (np.all(data < 0.1) and np.all(np.log(data) > -10000)):
        return np.fromfile(file_path, dtype="<d", count=-1, sep="").reshape(450, 450)

    if (not is_radar) and np.all(np.log(data) < -500):
        return np.fromfile(file_path, dtype="<d", count=-1, sep="").reshape(450, 450)

    return data


def gen_path_hook(func) -> Path:
    @wraps(func)
    def wrap(target_time: datetime, data_compose: None | DataCompose = None):
        """
        Example:
            target_time = datetime(2021, 6, 4, 5)
            data_config = {"var": "Radar", "lv": "NoRule"}
        """
        if data_compose is None:
            return func(target_time)
        return func(target_time) / data_compose.sub_dir_name

    return wrap


@gen_path_hook
def gen_path(target_time: datetime) -> Path:
    return (
        Path(DATA_PATH)
        / f"rwf_{target_time.strftime('%Y%m')}"
        / f"{target_time.strftime('%Y%m%d%H%M')}0000"
    )
