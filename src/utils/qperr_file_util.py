from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from ..const import QPERR_DATA_PATH


def get_path(target_time: datetime):
    return (
        Path(QPERR_DATA_PATH)
        / target_time.strftime("%Y%m")
        / f"{target_time.strftime('%Y%m%d_%H%M')}.nc"
    )


def get_data(target_time: datetime):
    qperr_filename = get_path(target_time)
    dataset = xr.open_dataset(str(qperr_filename))
    data = dataset["qperr"].values
    data = np.nan_to_num(data, nan=0.0)
    return data
