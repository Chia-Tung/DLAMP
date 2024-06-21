import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from .const import STANDARDIZATION_PATH
from .utils import DataCompose, DataType, Level, gen_data, gen_path


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
        array (np.ndarray): The data array to be destandardized with shape (lv, H, W, C) or
        (B, lv, H, W, C).

    Returns:
        np.ndarray: The destandardized data with shape (lv, H, W, C) or (B, lv, H, W, C).
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

    # define inner function
    def fn(
        array: np.ndarray,
        levels: list[Level],
        vars: list[DataType],
        batch: bool = False,
    ):
        new_array = np.zeros_like(array)
        for i, lv in enumerate(levels):
            for j, var in enumerate(vars):
                tmp_compose = DataCompose(var, lv)
                if batch:
                    new_array[:, i, :, :, j] = (
                        array[:, i, :, :, j]
                        * stat_dict_already[str(tmp_compose)]["std"]
                        + stat_dict_already[str(tmp_compose)]["mean"]
                    )
                else:
                    new_array[i, :, :, j] = (
                        array[i, :, :, j] * stat_dict_already[str(tmp_compose)]["std"]
                        + stat_dict_already[str(tmp_compose)]["mean"]
                    )
        return new_array

    # calculate
    num_array_dim = len(array.shape)
    match num_array_dim:
        case 4:  # w/o batch
            if array.shape[0] != 1:
                destandardize_levels = pressure_levels
                destandardize_vars = upper_vars
            else:
                destandardize_levels = [Level.Surface]
                destandardize_vars = surface_vars
            return fn(array, destandardize_levels, destandardize_vars)
        case 5:  # w/ batch
            if array.shape[1] != 1:
                destandardize_levels = pressure_levels
                destandardize_vars = upper_vars
            else:
                destandardize_levels = [Level.Surface]
                destandardize_vars = surface_vars
            return fn(array, destandardize_levels, destandardize_vars, batch=True)
        case _:
            raise ValueError("The shape of the input array is not supported.")


if __name__ == "__main__":
    calc_standardization()
