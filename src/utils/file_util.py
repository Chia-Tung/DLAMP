from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np

from src.const import DATA_PATH

from . import DataType


def gen_data(target_time: datetime, data_config: dict):
    assert all(
        x in data_config.keys() for x in ["var", "lv"]
    ), "data_config must contain 'var' and 'lv'"

    file_dir = gen_path(target_time, data_config)
    return read_cwa_npfile(file_dir, data_config["var"] == "Radar")


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
    def wrap(target_time: datetime, data_config: None | dict[str, str] = None):
        """
        Example:
            target_time = datetime(2021, 6, 4, 5)
            data_config = {"var": "Radar", "lv": "NoRule"}
        """
        if data_config is None:
            return func(target_time)
        return func(target_time) / DataType.gen_dir_name(**data_config)

    return wrap


@gen_path_hook
def gen_path(target_time: datetime) -> Path:
    return (
        Path(DATA_PATH)
        / f"rwf_{target_time.strftime('%Y%m')}"
        / f"{target_time.strftime('%Y%m%d%H%M')}0000"
    )
