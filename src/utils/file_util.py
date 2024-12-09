import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from ..const import DATA_PATH, DATA_SOURCE
from .data_compose import DataCompose, DataType


def gen_data(
    target_time: datetime,
    data_compose: DataCompose | list[DataCompose],
    dtype: np.dtype | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    """
    Generate numpy array data for a given target time and data composition.

    Args:
        target_time (datetime): The target time for which the data is generated.
        data_compose (DataCompose | list[DataCompose]): The data composition object or a list
            of data composition objects.
        dtype (np.dtype | None, optional): The data type of the generated data. Defaults to None.

    Returns:
        The generated data.

    """
    match DATA_SOURCE:
        case "NEO171_RWRF":
            if isinstance(data_compose, DataCompose):
                file_name = gen_path(target_time, data_compose)
                return read_cwa_npfile(file_name, data_compose.is_radar, dtype)
            elif isinstance(data_compose, list):
                ret = {}
                for ele in data_compose:
                    file_name = gen_path(target_time, ele)
                    ret[str(ele)] = read_cwa_npfile(file_name, ele.is_radar, dtype)
                return ret
        case "CWA_RWRF":
            file_name = gen_path(target_time)
            return read_cwa_ncfile(file_name, data_compose, dtype)
        case _:
            raise ValueError(f"Unknown data source: {DATA_SOURCE}")


def read_cwa_ncfile(
    file_path: Path,
    data_compose: DataCompose | list[DataCompose],
    dtype: np.dtype | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    """
    Read data from a NetCDF file based on the provided data composition.

    Args:
        file_path (Path): Path to the NetCDF file.
        data_compose (DataCompose | list[DataCompose]): A single DataCompose object or a list of
            DataCompose objects specifying what data to extract.
        dtype (np.dtype | None, optional): The numpy dtype to cast the data to. If None, keeps original dtype.
            Defaults to None.

    Returns:
        np.ndarray | dict[str, np.ndarray]: If data_compose is a single DataCompose object, returns a numpy array
            containing the requested data. If data_compose is a list, returns a dictionary mapping DataCompose
            string representations to their corresponding numpy arrays.
    """
    dataset = xr.open_dataset(str(file_path))
    pres_lvs = dataset[DataType.P.nc_key].values

    def fn(dc: DataCompose):
        data = dataset[dc.combined_key].values.squeeze()  # (Z, H, W)
        data = data.astype(dtype) if dtype is not None else data
        if not dc.level.is_surface():
            (idx,) = np.where(pres_lvs == float(dc.level.nc_key))
            data = data[idx[0]]
        return data  # (H, W)

    if isinstance(data_compose, DataCompose):
        return fn(data_compose)
    elif isinstance(data_compose, list):
        ret = {}
        for ele in data_compose:
            ret[str(ele)] = fn(ele)
        return ret


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

    if is_radar:
        data[data < 0] = 0  # set negative values to 0

    return data


def gen_path(
    target_time: datetime,
    data_compose: None | DataCompose = None,
    data_source: str = DATA_SOURCE,
) -> Path:
    """
    A function that generates a path based on the given target time and data compose.

    Parameters:
        target_time (datetime): The target time for generating the path.
        data_compose (None | DataCompose): The data composition object.
        data_source (str): The way generating the path depends on different data sources.
            e.g.
                "NEO171_RWRF" -> data stored on neo171 server
                "CWA_RWRF" -> data stored on CWA HPC

    Returns:
        Path: The generated path.
    """
    match data_source:
        case "NEO171_RWRF":
            assert data_compose is not None, "DataCompose is required for NEO171_RWRF"
            return (
                Path(DATA_PATH)
                / f"rwf_{target_time.strftime('%Y%m')}"
                / f"{target_time.strftime('%Y%m%d%H%M')}0000"
                / data_compose.basename
            )
        case "CWA_RWRF":
            return (
                Path(DATA_PATH)
                / f"RWRF_{target_time.strftime('%Y-%m')}"
                / f"{target_time.strftime('%Y-%m-%d_%H')}"
                / f"wrfout_d01_{target_time.strftime('%Y-%m-%d_%H')}_interp"
            )
        case _:
            raise ValueError(f"Invalid data source: {data_source}")


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
