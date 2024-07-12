from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.const import DBZ_COLOR, DBZ_LV, DBZ_NORM, FIGURE_PATH
from src.utils import DataCompose, DataType, Level, gen_data

from .tw_background import TwBackground


class VizRadar(TwBackground):
    def __init__(self):
        super().__init__()
        self.title_suffix = "_radar_reflectivity"

    def plot_mxn(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        all_init_times: list[datetime] = [],
        grid_on: bool = False,
    ) -> tuple[Figure, Axes]:
        """
        Args:
            lon (np.ndarray): The longitude data with shape (H, W).
            lat (np.ndarray): The latitude data with shape (H, W).
            ground_truth (np.ndarray): The ground truth data with shape (S, H, W).
            prediction (np.ndarray): The predicted data with shape (S, H, W).
            all_init_times (list[datetime]): A list of all initial times in length S.
            grid_on (bool, optional): Whether to show grid. Defaults to False.
        """
        assert len(ground_truth.shape) == 3
        assert ground_truth.shape[-2:] == lat.shape

        # since lat/lon may not be monotonically increasing in a same pace
        if len(lat.shape) == 2 and len(lon.shape) == 2:
            lat = np.linspace(lat[0, 0], lat[-1, 0], lat.shape[0])
            lon = np.linspace(lon[0, 0], lon[0, -1], lon.shape[1])

        rows = 2  # gt/pred
        columns = ground_truth.shape[0]
        plt.close()
        fig, ax = plt.subplots(rows, columns, figsize=(20, 7), dpi=200, facecolor="w")

        # ground truth
        for j in range(columns):
            tmp_ax = ax[0, j]
            time_title = (
                all_init_times[j].strftime("%Y%m%d_%H%M") if all_init_times else ""
            )
            fig, tmp_ax = self.plot_bg(fig, tmp_ax, grid_on)
            fig, tmp_ax = self._plot_radar(
                fig, tmp_ax, lon, lat, ground_truth[j], time_title
            )

        # prdiction
        for j in range(columns):
            tmp_ax = ax[1, j]
            time_title = (
                all_init_times[j].strftime("%Y%m%d_%H%M") if all_init_times else ""
            )
            fig, tmp_ax = self.plot_bg(fig, tmp_ax, grid_on)
            fig, tmp_ax = self._plot_radar(
                fig, tmp_ax, lon, lat, prediction[j], time_title
            )

        return fig, ax

    def plot_1x1(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        title: str = "",
        grid_on: bool = False,
    ) -> tuple[Figure, Axes]:
        # since lat/lon may not be monotonically increasing in a same pace
        if len(lat.shape) == 2 and len(lon.shape) == 2:
            lat = np.linspace(lat[0, 0], lat[-1, 0], lat.shape[0])
            lon = np.linspace(lon[0, 0], lon[0, -1], lon.shape[1])

        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200, facecolor="w")
        fig, ax = super().plot_bg(fig, ax, grid_on)
        fig, ax = self._plot_radar(fig, ax, lon, lat, data, title)

        return fig, ax

    def _plot_radar(
        self,
        fig: Figure,
        ax: Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        title: str = "",
    ) -> tuple[Figure, Axes]:
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

    def plot_1xn(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: list[np.ndarray],
        grid_on: bool = False,
        titles: list[str] = [],
    ):
        if len(lat.shape) == 2 and len(lon.shape) == 2:
            lat = np.linspace(lat[0, 0], lat[-1, 0], lat.shape[0])
            lon = np.linspace(lon[0, 0], lon[0, -1], lon.shape[1])

        cols = len(data)
        plt.close()
        fig, ax = plt.subplots(1, cols, figsize=(20, 7), dpi=200, facecolor="w")
        for j in range(cols):
            tmp_ax = ax[j]
            fig, tmp_ax = self.plot_bg(fig, tmp_ax, grid_on)
            fig, tmp_ax = self._plot_radar(fig, tmp_ax, lon, lat, data[j], titles[j])

        return fig, ax


if __name__ == "__main__":
    target_time = datetime(2022, 10, 16, 0)
    data_radar = gen_data(target_time, DataCompose(DataType.Radar, Level.NoRule))
    data_lat = gen_data(target_time, DataCompose(DataType.Lat, Level.Surface))
    data_lon = gen_data(target_time, DataCompose(DataType.Lon, Level.Surface))

    viz = VizRadar()
    fig, ax = viz.plot_1x1(data_lon, data_lat, data_radar)
    fig.savefig(
        f"{FIGURE_PATH}/{target_time.strftime('%Y%m%d_%H%M')}_mos.png",
        transparent=False,
    )
    plt.close()
