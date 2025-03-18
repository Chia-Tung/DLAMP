from datetime import datetime

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .tw_background import TwBackground


class VizSwdown(TwBackground):
    def __init__(self):
        super().__init__()

    def plot_1x1(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        title: str = "",
        grid_on: bool = False,
    ) -> tuple[Figure, Axes]:
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200, facecolor="w")
        fig, ax = super().plot_bg(fig, ax, grid_on)
        fig, ax = self._plot_swdown(fig, ax, lon, lat, data, title)

        return fig, ax

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
            fig, tmp_ax = self._plot_pressure(
                fig, tmp_ax, lon, lat, ground_truth[j], time_title
            )

        # prediction
        for j in range(columns):
            tmp_ax = ax[1, j]
            time_title = (
                all_init_times[j].strftime("%Y%m%d_%H%M") if all_init_times else ""
            )
            fig, tmp_ax = self.plot_bg(fig, tmp_ax, grid_on)
            fig, tmp_ax = self._plot_swdown(
                fig, tmp_ax, lon, lat, prediction[j], time_title
            )

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
            fig, tmp_ax = self._plot_swdown(fig, tmp_ax, lon, lat, data[j], title)

        return fig, ax

    def _plot_swdown(
        self,
        fig: Figure,
        ax: Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        title: str = "",
    ) -> tuple[Figure, Axes]:
        data = np.round(data / 100)  # pa to hpa
        conf = ax.contourf(
            lon,
            lat,
            data,
            cmap="viridis",
            levels=np.arange(-40, 400, 40),  # SWDOWN
            # levels=np.arange(0, 300, 30),  # OLR
            zorder=0,
            extend="both",
        )

        # inline lables
        clabels = ax.clabel(
            conf, inline=True, colors="k", fontsize=10, use_clabeltext=False
        )
        for label in clabels:
            label.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="white"),
                    path_effects.Normal(),
                ]
            )

        if title:
            ax.set_title(title)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # colorbar
        cbar = fig.colorbar(conf, cax=cax)
        cbar.ax.set_title("$\\frac{W}{m^2}$")

        return fig, ax
