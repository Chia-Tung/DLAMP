from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.const import FIGURE_PATH
from src.utils import DataCompose, DataType, Level, gen_data

from .tw_background import TwBackground


class VizVor(TwBackground):
    def __init__(self, pressure_level: int):
        super().__init__()
        self.press_lv = pressure_level
        self.title_suffix = f"Vorticity@{self.press_lv}\n"

    def plot_mxn(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        ground_truth_u: np.ndarray,
        ground_truth_v: np.ndarray,
        prediction_u: np.ndarray,
        prediction_v: np.ndarray,
        grid_resolution_in_meter: list[int],
        all_init_times: list[datetime] = [],
    ) -> tuple[Figure, Axes]:
        assert len(ground_truth_u.shape) == 3
        assert ground_truth_u.shape[-2:] == lat.shape

        # since lat/lon may not be monotonically increasing in a same pace
        if len(lat.shape) == 2 and len(lon.shape) == 2:
            lat = np.linspace(lat[0, 0], lat[-1, 0], lat.shape[0])
            lon = np.linspace(lon[0, 0], lon[0, -1], lon.shape[1])

        rows = 2  # gt/pred
        columns = ground_truth_u.shape[0]
        fig, ax = plt.subplots(rows, columns, figsize=(20, 7), dpi=200, facecolor="w")

        # ground truth
        for j in range(columns):
            tmp_ax = ax[0, j]
            time_title = (
                all_init_times[j].strftime("%Y%m%d_%H%M") if all_init_times else ""
            )
            fig, tmp_ax = self.plot_bg(fig, tmp_ax)
            fig, tmp_ax = self._plot_vor(
                fig,
                tmp_ax,
                lon,
                lat,
                ground_truth_u[j],
                ground_truth_v[j],
                grid_resolution_in_meter,
                time_title,
            )

        # prdiction
        for j in range(columns):
            tmp_ax = ax[1, j]
            time_title = (
                all_init_times[j].strftime("%Y%m%d_%H%M") if all_init_times else ""
            )
            fig, tmp_ax = self.plot_bg(fig, tmp_ax)
            fig, tmp_ax = self._plot_vor(
                fig,
                tmp_ax,
                lon,
                lat,
                prediction_u[j],
                prediction_v[j],
                grid_resolution_in_meter,
                time_title,
            )

        return fig, ax

    def plot_1x1(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        grid_resolution_in_meter: list[int],
        title: str = "",
    ) -> tuple[Figure, Axes]:
        # since lat/lon may not be monotonically increasing in a same pace
        if len(lat.shape) == 2 and len(lon.shape) == 2:
            lat = np.linspace(lat[0, 0], lat[-1, 0], lat.shape[0])
            lon = np.linspace(lon[0, 0], lon[0, -1], lon.shape[1])

        fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200, facecolor="w")
        fig, ax = super().plot_bg(fig, ax)
        fig, ax = self._plot_vor(
            fig, ax, lon, lat, u_wind, v_wind, grid_resolution_in_meter, title
        )

        return fig, ax

    def _plot_vor(
        self,
        fig: Figure,
        ax: Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        grid_resolution_in_meter: list[int],
        title: str = "",
    ) -> tuple[Figure, Axes]:
        # calculate vorticity
        vor_u = (u_wind[1:, :] - u_wind[0:-1, :]) / grid_resolution_in_meter[1]
        vor_u = np.concatenate([vor_u[0:1, :], vor_u], axis=0)
        vor_v = (v_wind[:, 1:] - v_wind[:, 0:-1]) / grid_resolution_in_meter[0]
        vor_v = np.concatenate([vor_v[:, 0:1], vor_v], axis=1)
        vor = vor_v - vor_u

        # plot data
        conf = ax.contourf(
            lon,
            lat,
            10**5 * vor,
            np.arange(-100, 101, 2),
            cmap="bwr",
            zorder=0,
            extend="both",
        )
        if title:
            ax.set_title(self.title_suffix + title)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # colorbar
        cbar = fig.colorbar(conf, cax=cax)
        cbar.ax.set_title("($10^{-5} s^{-1}$)")
        cbar.set_ticks(np.arange(-100, 101, 20))

        return fig, ax

    def plot_1xn(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        u_wind_list: list[np.ndarray],
        v_wind_list: list[np.ndarray],
        grid_resolution_in_meter: list[int],
        titles: list[str] = [],
        grid_on: bool = False,
    ):
        if len(lat.shape) == 2 and len(lon.shape) == 2:
            lat = np.linspace(lat[0, 0], lat[-1, 0], lat.shape[0])
            lon = np.linspace(lon[0, 0], lon[0, -1], lon.shape[1])

        cols = len(u_wind_list)

        plt.close()
        fig, ax = plt.subplots(1, cols, figsize=(20, 7), dpi=200, facecolor="w")
        for j in range(cols):
            tmp_ax = ax[j]
            title = titles[j] if titles else ""
            fig, tmp_ax = self.plot_bg(fig, tmp_ax, grid_on)
            fig, tmp_ax = self._plot_vor(
                fig,
                tmp_ax,
                lon,
                lat,
                u_wind_list[j],
                v_wind_list[j],
                grid_resolution_in_meter,
                title,
            )

        return fig, ax


if __name__ == "__main__":
    target_time = datetime(2022, 10, 16, 0)
    u850 = gen_data(target_time, DataCompose(DataType.U, Level.Hpa850))
    v850 = gen_data(target_time, DataCompose(DataType.V, Level.Hpa850))
    data_lat = gen_data(target_time, DataCompose(DataType.Lat, Level.Surface))
    data_lon = gen_data(target_time, DataCompose(DataType.Lon, Level.Surface))
    grid_point_resolution = [2e3, 2e3]  # unit: m (for lon/lat)

    viz = VizVor()
    fig, ax = viz.plot_1x1(
        data_lon,
        data_lat,
        u850,
        v850,
        grid_point_resolution,
    )
    fig.savefig(
        f"{FIGURE_PATH}/{target_time.strftime('%Y%m%d_%H%M')}_vorticity.png",
        transparent=False,
    )
    plt.close()
