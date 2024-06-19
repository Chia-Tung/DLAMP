import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from .const import STANDARDIZATION_PATH
from .utils import DataCompose, Level, gen_data, gen_path


def calc_standardization(
    start_time: datetime = datetime(2020, 1, 1),
    end_time: datetime = datetime(2021, 1, 1),
    random_num: int = 100,
) -> None:
    """
    calculates the mean and standard deviation from a dataset within a specified time range.
    """
    # load config
    with open("config/data/rwrf.yaml", "r") as stream:
        data_config = yaml.safe_load(stream)

    # load already calculated mean and standard deviation
    if Path(STANDARDIZATION_PATH).exists():
        with open(STANDARDIZATION_PATH, "r") as f:
            stat_dict_already: dict = json.load(f)
    else:
        stat_dict_already = {}

    # calculation
    data_list = DataCompose.from_config(data_config["train_data"])
    for data_compose in tqdm(data_list):
        dt = start_time
        container = []
        print(f"start executing {data_compose}")

        if str(data_compose) in stat_dict_already:
            print(
                f"skip {data_compose} because it already exists in {STANDARDIZATION_PATH}"
            )
            continue

        while dt < end_time:
            if gen_path(dt, data_compose).exists():
                data = gen_data(dt, data_compose)  # (450, 450)
                # random pick values from data
                indices = np.arange(data.size)
                chosen_indices = np.random.choice(indices, random_num, replace=False)
                # Convert the flat indices to 2D indices
                rows, cols = np.unravel_index(chosen_indices, data.shape)
                random_values = data[rows, cols]
                container.append(random_values)

            dt += timedelta(hours=1)
            if (dt - start_time) % timedelta(days=30) == timedelta(days=0):
                print(f"now is processing {dt}")

        all_data = np.stack(container)
        stat_dict_already[str(data_compose)] = {
            "mean": np.mean(all_data),
            "std": np.std(all_data),
        }

    # write into json file
    with open(STANDARDIZATION_PATH, "w") as f:
        json.dump(stat_dict_already, f)


def destandardization(array: np.ndarray) -> np.ndarray:
    """
    destandardize the data based on the mean and standard deviation calculated from the dataset

    Args:
        array (np.ndarray): The data array to be destandardized with shape (lv, H, W, C).

    Returns:
        np.ndarray: The destandardized data with shape (lv, H, W, C).
    """

    # load already calculated mean and standard deviation
    assert Path(STANDARDIZATION_PATH).exists(), "Calculate the mean and std first."
    with open(STANDARDIZATION_PATH, "r") as f:
        stat_dict_already: dict = json.load(f)

    # load config
    with open("config/data/rwrf.yaml", "r") as stream:
        data_config = yaml.safe_load(stream)
    data_list = DataCompose.from_config(data_config["train_data"])
    pressure_levels = DataCompose.get_all_levels(data_list, only_upper=True)
    upper_vars = DataCompose.get_all_vars(data_list, only_upper=True)
    surface_vars = DataCompose.get_all_vars(data_list, only_surface=True)

    # calculate
    if array.shape[0] != 1:  # upper
        for i, lv in enumerate(pressure_levels):
            for j, var in enumerate(upper_vars):
                tmp_compose: DataCompose = DataCompose(var, lv)
                array[i, :, :, j] = (
                    array[i, :, :, j] * stat_dict_already[str(tmp_compose)]["std"]
                    + stat_dict_already[str(tmp_compose)]["mean"]
                )
        return array
    else:  # surface
        for j, var in enumerate(surface_vars):
            tmp_compose: DataCompose = DataCompose(var, Level.Surface)
            array[0, :, :, j] = (
                array[0, :, :, j] * stat_dict_already[str(tmp_compose)]["std"]
                + stat_dict_already[str(tmp_compose)]["mean"]
            )
        return array


if __name__ == "__main__":
    calc_standardization()
