import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from .const import BLACKLIST_PATH, DATA_CONFIG_PATH, STANDARDIZATION_PATH
from .utils import DataCompose, DataGenerator, DataType, gen_path

# load already calculated mean and standard deviation
if Path(STANDARDIZATION_PATH).exists():
    with open(STANDARDIZATION_PATH, "r") as f:
        stat_dict: dict = json.load(f)
else:
    stat_dict = {}

MEAN_THRESHOLD = 1e-4


def calc_standardization(
    start_time: datetime = datetime(2021, 1, 1),
    end_time: datetime = datetime(2022, 12, 31),
    sample_size: int = 100,
    num_criteria: int = 1000,
) -> None:
    """
    calculates the mean and standard deviation from a dataset within a specified time range.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # load config
    with open(DATA_CONFIG_PATH, "r") as stream:
        data_config = yaml.safe_load(stream)

    # load blacklist
    with open(BLACKLIST_PATH, "r") as f:
        blacklist = [
            datetime.strptime(line.strip(), "%Y-%m-%d %H:%M")
            for line in f
            if line.strip()  # Skip empty lines
        ]

    # data generator
    data_list = DataCompose.from_config(data_config["train_data"])
    data_gnrt = DataGenerator(data_config["data_shape"], data_config["image_shape"])
    use_Kth_hour_pred = (
        data_config["use_Kth_hour_pred"] if "use_Kth_hour_pred" in data_config else None
    )

    def _progress_one_step(dt, month_cnt):
        dt += timedelta(hours=8)
        if (dt - start_time) / timedelta(days=30) > month_cnt:
            month_cnt += 1
            logging.info(f"now is processing {dt}")
        return dt, month_cnt

    for data_compose in tqdm(data_list):
        dt = start_time
        month_cnt = 0
        container = []
        logging.info(f"start executing {data_compose}")

        if str(data_compose) in stat_dict:
            logging.info(
                f"skip {data_compose} because it already exists in {STANDARDIZATION_PATH}"
            )
            continue

        while dt < end_time:
            if (
                gen_path(dt, data_compose, use_Kth_hour_pred).exists()
                and dt not in blacklist
            ):
                data: np.ndarray = data_gnrt.yield_data(
                    dt, data_compose, use_Kth_hour_pred=use_Kth_hour_pred
                )

                # random pick values from data
                indices = np.arange(data.size)
                chosen_indices = np.random.choice(indices, sample_size, replace=False)
                rows, cols = np.unravel_index(chosen_indices, data.shape)
                random_values = data[rows, cols]

                if (
                    data_compose.var_name == DataType.SWDOWN
                    and np.mean(random_values) == 0
                ):
                    dt, month_cnt = _progress_one_step(dt, month_cnt)
                    continue

                container.append(random_values)

            dt, month_cnt = _progress_one_step(dt, month_cnt)

        # clip values to 10-90 percentile range
        all_data = np.stack(container).flatten()
        lower_bound = np.percentile(all_data, 10)
        upper_bound = np.percentile(all_data, 90)
        filtered_data = all_data[(all_data > lower_bound) & (all_data < upper_bound)]

        if len(filtered_data) < num_criteria:
            logging.info(
                f"skip {data_compose} because data sample {len(filtered_data)} is not enough"
            )
            continue

        stat_dict[str(data_compose)] = {
            "mean": float(np.mean(filtered_data)),
            "std": float(np.std(filtered_data)),
        }

        # write into json file
        with open(STANDARDIZATION_PATH, "w") as f:
            json.dump(stat_dict, f, indent=4)


def standardization(dc_name: str, array: np.ndarray) -> np.ndarray:
    if dc_name in stat_dict:
        stat = stat_dict[dc_name]
        if stat["mean"] < MEAN_THRESHOLD:
            return array
        else:
            return (array - stat["mean"]) / stat["std"]
    else:
        return np.zeros_like(array)  # force to be zero if no standardization


def destandardization(array: np.ndarray) -> np.ndarray:
    """
    destandardize the data based on the mean and standard deviation calculated from the dataset

    Args:
        array (np.ndarray): The data array to be destandardized with shape (lv, H, W, C) or
        (B, lv, H, W, C).

    Returns:
        np.ndarray: The destandardized data with shape (lv, H, W, C) or (B, lv, H, W, C).
    """
    # load config
    with open(DATA_CONFIG_PATH, "r") as stream:
        data_config = yaml.safe_load(stream)
    data_list = DataCompose.from_config(data_config["train_data"])

    # main
    num_array_dim = len(array.shape)
    if num_array_dim not in [4, 5]:
        raise ValueError(f"Expected 4D or 5D array, got {num_array_dim}D")

    is_surface = array.shape[1 if num_array_dim == 5 else 0] == 1
    return _destandardize(array, data_list, is_surface)


def _destandardize(
    array: np.ndarray, data_list: list[DataCompose], is_sfc: bool
) -> np.ndarray:
    """Handle destandardization for surface or upper-level variables.

    Args:
        array: Input array with shape (lv, H, W, C) or (B, lv, H, W, C)
        data_list: List of DataCompose objects
        is_sfc: Boolean indicating if processing surface level data

    Returns:
        Destandardized array with same shape as input
    """
    new_array = np.zeros_like(array)

    # Filter data compositions and get indices based on level type
    if is_sfc:
        filtered_dc = [dc for dc in data_list if dc.level.is_surface()]
        variables = DataCompose.get_all_vars(data_list, only_surface=True)
    else:
        filtered_dc = [dc for dc in data_list if not dc.level.is_surface()]
        levels = DataCompose.get_all_levels(data_list, only_upper=True)
        variables = DataCompose.get_all_vars(data_list, only_upper=True)

    # Process each data composition
    for dc in filtered_dc:
        if str(dc) not in stat_dict:
            continue

        stat = stat_dict[str(dc)]
        lv_idx = 0 if is_sfc else levels.index(dc.level)
        var_idx = variables.index(dc.var_name)

        if len(array.shape) == 5:
            new_array[:, lv_idx, :, :, var_idx] = destandardize_array(
                array[:, lv_idx, :, :, var_idx], stat
            )
        else:
            new_array[lv_idx, :, :, var_idx] = destandardize_array(
                array[lv_idx, :, :, var_idx], stat
            )

    return new_array


def destandardize_array(array: np.ndarray, stat: dict[str, float]) -> np.ndarray:
    """Apply destandardization to a single array using statistics from stat_dict."""
    return (
        array if stat["mean"] < MEAN_THRESHOLD else array * stat["std"] + stat["mean"]
    )


if __name__ == "__main__":
    calc_standardization()
