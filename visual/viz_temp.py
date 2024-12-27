from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.const import FIGURE_PATH, TEMP_COLOR, TEMP_LV
from src.utils import DataCompose, DataType, Level, gen_data

from .tw_background import TwBackground


class VizTemp(TwBackground):
    def __init__(self, pressure_level: int | None = None):
        super().__init__()
        self.press_lv = pressure_level
        self.title_suffix = f"Temperature@{self.press_lv}" if pressure_level else ""

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
            fig, tmp_ax = self._plot_temp(
                fig, tmp_ax, lon, lat, ground_truth[j], time_title
            )

        # prediction
        for j in range(columns):
            tmp_ax = ax[1, j]
            time_title = (
                all_init_times[j].strftime("%Y%m%d_%H%M") if all_init_times else ""
            )
            fig, tmp_ax = self.plot_bg(fig, tmp_ax, grid_on)
            fig, tmp_ax = self._plot_temp(
                fig, tmp_ax, lon, lat, prediction[j], time_title
            )

        return fig, ax

    def plot_1x1(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        temp: np.ndarray,
        title: str = "",
    ) -> tuple[Figure, Axes]:
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200, facecolor="w")
        fig, ax = super().plot_bg(fig, ax)
        fig, ax = self._plot_temp(fig, ax, lon, lat, temp, title)

        return fig, ax

    def _plot_temp(
        self,
        fig: Figure,
        ax: Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        temp: np.ndarray,
        title: str = "",
    ) -> tuple[Figure, Axes]:
        # from Kelvin to Celsius
        temp_c = temp - 273.15

        # plot data
        conf = ax.contourf(
            lon,
            lat,
            temp_c,
            levels=TEMP_LV,
            colors=TEMP_COLOR,
            zorder=0,
            extend="max",
        )
        if title:
            ax.set_title(f"{title} {self.title_suffix}")

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # colorbar
        cbar = fig.colorbar(conf, cax=cax)
        cbar.ax.set_title("$^{o}$C")

        return fig, ax

    def plot_1xn(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        titles: list[str] = [],
        grid_on: bool = False,
    ):
        cols = data.shape[0]

        plt.close()
        fig, ax = plt.subplots(1, cols, figsize=(18, 2.5), dpi=200, facecolor="w")
        for j in range(cols):
            tmp_ax = ax[j]
            title = titles[j] if titles else ""
            fig, tmp_ax = self.plot_bg(fig, tmp_ax, grid_on)
            fig, tmp_ax = self._plot_temp(fig, tmp_ax, lon, lat, data[j], title)

        return fig, ax


if __name__ == "__main__":
    target_time = datetime(2022, 10, 16, 0)
    t850 = gen_data(target_time, DataCompose(DataType.T, Level.Hpa850))
    data_lat = gen_data(target_time, DataCompose(DataType.Lat, Level.Surface))
    data_lon = gen_data(target_time, DataCompose(DataType.Lon, Level.Surface))

    viz = VizTemp("Hpa850")
    fig, ax = viz.plot_1x1(data_lon, data_lat, t850)
    fig.savefig(
        f"{FIGURE_PATH}/{target_time.strftime('%Y%m%d_%H%M')}_temperature.png",
        transparent=False,
    )
    plt.close()
