from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.const import DBZ_COLOR, DBZ_LV, DBZ_NORM, FIGURE_PATH
from src.utils import DataCompose, DataType, Level, gen_data

from .tw_background import TwBackground


class VizRadar(TwBackground):
    def __init__(self):
        super().__init__()
        self.title_suffix = "_radar_reflectivity"

    def plot(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        title: str = "",
        grid_on: bool = False,
    ):
        # since lat/lon may not be monotonically increasing in a same pace
        if len(lat.shape) == 2 and len(lon.shape) == 2:
            lat = np.linspace(lat[0, 0], lat[-1, 0], lat.shape[0])
            lon = np.linspace(lon[0, 0], lon[0, -1], lon.shape[1])

        fig, ax = super().plot_bg(grid_on)

        # plot data
        ax.pcolormesh(
            lon,
            lat,
            data,
            edgecolors="none",
            shading="auto",
            norm=DBZ_NORM,
            cmap=DBZ_COLOR,
            zorder=0,
        )
        if title:
            ax.set_title(title + self.title_suffix)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # colorbar
        cbar = fig.colorbar(
            cm.ScalarMappable(norm=DBZ_NORM, cmap=DBZ_COLOR),
            cax=cax,
            ticks=np.arange(DBZ_LV[0], DBZ_LV[-1] + 1, 5),
        )
        cbar.ax.set_title("dBZ")

        return fig, ax

    def plot_mxn(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        all_init_times: list[datetime],
    ):
        """
        Args:
            lon (np.ndarray): The longitude data with shape (H, W).
            lat (np.ndarray): The latitude data with shape (H, W).
            ground_truth (np.ndarray): The ground truth data with shape (S, H, W).
            prediction (np.ndarray): The predicted data with shape (S, H, W).
            all_init_times (list[datetime]): A list of all initial times in length S.
        """
        assert len(ground_truth.shape) == 3
        assert ground_truth.shape[-2:] == lat.shape
        assert len(all_init_times) == ground_truth.shape[0]

        rows = 2  # gt/pred
        columns = len(all_init_times)
        fig, ax = plt.subplots(rows, columns, figsize=(7, 14), dpi=200, facecolor="w")

        # how to plot multiple subplots

        return fig, ax


if __name__ == "__main__":
    target_time = datetime(2022, 10, 16, 0)
    data_radar = gen_data(target_time, DataCompose(DataType.Radar, Level.NoRule))
    data_lat = gen_data(target_time, DataCompose(DataType.Lat, Level.Surface))
    data_lon = gen_data(target_time, DataCompose(DataType.Lon, Level.Surface))

    viz = VizRadar()
    fig, ax = viz.plot(data_lon, data_lat, data_radar)
    fig.savefig(
        f"{FIGURE_PATH}/{target_time.strftime('%Y%m%d_%H%M')}_mos.png",
        transparent=False,
    )
    plt.close()
