import warnings
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np

from ..const import DATA_PATH
from .data_compose import DataCompose


def gen_data(
    target_time: datetime, data_compose: DataCompose, dtype: np.dtype | None = None
) -> np.ndarray:
    """
    Generate numpy array data for a given target time and data composition.

    Args:
        target_time (datetime): The target time for which the data is generated.
        data_compose (DataCompose): The data composition object.
        dtype (np.dtype | None, optional): The data type of the generated data. Defaults to None.

    Returns:
        The generated data.

    """
    file_dir = gen_path(target_time, data_compose)
    return read_cwa_npfile(file_dir, data_compose.is_radar, dtype)


def read_cwa_npfile(
    file_path: Path, is_radar: bool, dtype: np.dtype | None = None
) -> np.ndarray:
    """
    The x and y grids point of RWRF model data are 450 and 450, respectively.

    One thing needs to notice, the data type may save as little-endian with
    double precision or big-endian with double precision. User may design a
    judgment rule for checking if there are weird values after read data.
    """
    data = np.fromfile(file_path, dtype=">d", count=-1, sep="").reshape(450, 450)

    # since log(0) = -inf, log(neg) = nan, np.all() will always return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        log_data = np.log(data)
    log_data = np.ma.array(log_data, mask=np.isnan(log_data) + np.isneginf(log_data))

    if is_radar and (np.all(data < 0.1) and np.all(log_data > -10000)):
        data = np.fromfile(file_path, dtype="<d", count=-1, sep="").reshape(450, 450)

    if (not is_radar) and np.all(log_data < -500):
        data = np.fromfile(file_path, dtype="<d", count=-1, sep="").reshape(450, 450)

    if dtype is not None:
        data = data.astype(dtype)

    return data


def gen_path_hook(func) -> Path:
    """
    A decorator function that wraps the given `func` and returns a modified function
    providing the full sub-directory path based on the given data composition.

    Parameters:
        func (Callable): The function to be wrapped.

    Returns:
        Callable: The modified function.
    """

    @wraps(func)
    def wrap(target_time: datetime, data_compose: None | DataCompose = None):
        if data_compose is None:
            return func(target_time)
        return func(target_time) / data_compose.sub_dir_name

    return wrap


@gen_path_hook
def gen_path(target_time: datetime) -> Path:
    """
    A function that generates a path based on the given target time.

    Parameters:
        target_time (datetime): The target time for generating the path.

    Returns:
        Path: The generated path.
    """
    return (
        Path(DATA_PATH)
        / f"rwf_{target_time.strftime('%Y%m')}"
        / f"{target_time.strftime('%Y%m%d%H%M')}0000"
    )


def convert_hydra_dir_to_timestamp(hydra_dir: Path | str) -> str:
    """
    Convert a directory path to a timestamp string. Or just return itself if it's in `str` type.

    Args:
        hydra_dir (Path | str): The path to the hydra output directory.

    Returns:
        str: The timestamp string in the format "%y%m%d_%H%M%S".

    Raises:
        ValueError: If the hydra directory path cannot be parsed into a datetime object.
    """
    try:
        dt = datetime.strptime(
            f"{hydra_dir.parent.name} {hydra_dir.name}", "%Y-%m-%d %H:%M:%S"
        )
    except:
        if isinstance(hydra_dir, str):
            warnings.warn(
                f'given hydra dir "{hydra_dir}" can\'t be parsed into datetime, '
                f"return itself ({hydra_dir}) as timestamp",
                UserWarning,
            )
            return hydra_dir
        else:
            raise ValueError(
                f"given hydra dir {hydra_dir} can't be parsed into datetime"
            )

    return dt.strftime("%y%m%d_%H%M%S")
