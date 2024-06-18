from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import trange

from .const import LAND_SEA_MASK_PATH, TOPOGRAPHY_MASK_PATH
from .utils import DataGenerator


def gen_const_masks():
    """
    Prepare land_sea mask and terrain mask for training. The coordinates are the
    same as those recorded in config.yaml.
    """

    # load config
    with open("config/data/rwrf.yaml", "r") as stream:
        data_config = yaml.safe_load(stream)
        data_shape = data_config["data_shape"]

    with open("config/lightning/pangu_rwrf.yaml", "r") as stream:
        lightning_config = yaml.safe_load(stream)
        image_shape = lightning_config["image_shape"]

    data_gnrt = DataGenerator(data_shape, image_shape)
    target_lat, target_lon = data_gnrt.yield_data(
        datetime(2022, 10, 1, 0), {"Lat": ["NoRule"], "Lon": ["NoRule"]}
    )

    # real terrain data
    # OBJECTID_1	OBJECTID	townname	countyname	BASIN_NAME	N_1	E_1	高程	坡度	坡向	Shape_Leng	ORIG_FID	geometry
    # 0	1	1	None	None	None	25.3000	120.0	0.0	0.0	0.0	0.05	0	POINT (120.00000 25.30000)
    # 1	2	2	None	None	None	25.2875	120.0	0.0	0.0	0.0	0.05	1	POINT (120.00000 25.28750)
    # 2	3	3	None	None	None	25.2750	120.0	0.0	0.0	0.0	0.05	2	POINT (120.00000 25.27500)
    # 3	4	4	None	None	None	25.2625	120.0	0.0	0.0	0.0	0.05	3	POINT (120.00000 25.26250)
    # 4	5	5	None	None	None	25.2500	120.0	0.0	0.0	0.0	0.05	4	POINT (120.00000 25.25000)
    filename = "./assets/terrain_shp/GIS_terrain.shp"
    terrain_data: gpd.GeoDataFrame = gpd.read_file(filename)
    terrain_lat: np.ndarray = np.sort(terrain_data["N_1"].unique())  # (273,)
    terrain_lon: np.ndarray = np.sort(terrain_data["E_1"].unique())  # (161,)
    assert len(terrain_lat) * len(terrain_lon) == len(terrain_data)

    # mapping
    terrain_mask = np.zeros_like(target_lat, dtype=np.float32)
    for i in trange(target_lat.shape[0]):
        for j in range(target_lat.shape[1]):
            lat = target_lat[i, j]
            lon = target_lon[i, j]

            if (
                lat < terrain_lat[0]
                or lat > terrain_lat[-1]
                or lon < terrain_lon[0]
                or lon > terrain_lon[-1]
            ):
                continue

            closest_lat = find_closest_value(terrain_lat, lat)
            closest_lon = find_closest_value(terrain_lon, lon)
            combined_filter = terrain_data[
                (terrain_data["E_1"] == closest_lon)
                & (terrain_data["N_1"] == closest_lat)
            ]
            terrain_mask[i, j] = combined_filter["高程"].values

    # save npy
    np.save(TOPOGRAPHY_MASK_PATH, terrain_mask)
    np.save(LAND_SEA_MASK_PATH, np.where(terrain_mask > 0.5, 1, 0))
    print("done")


def find_closest_value(input_array: np.ndarray, target: float) -> float:
    assert len(input_array.shape) == 1, "Input array must be 1D"
    new_array = input_array - target
    min_index = np.argmin(np.abs(new_array))  # Only the first occurrence is returned.
    return input_array[min_index]


def plot(terrain_mask, lat, lon):
    c = plt.pcolor(lon, lat, terrain_mask)
    ax = c.axes
    ax.axis("equal")
    plt.colorbar()


if __name__ == "__main__":
    gen_const_masks()
