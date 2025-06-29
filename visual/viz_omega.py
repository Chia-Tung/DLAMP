import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .tw_background import TwBackground


class VizOmega(TwBackground):
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
        fig, ax = self._plot_omega(fig, ax, lon, lat, data, title)

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
            fig, tmp_ax = self._plot_omega(fig, tmp_ax, lon, lat, data[j], title)

        return fig, ax

    def _plot_omega(
        self,
        fig: Figure,
        ax: Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        title: str = "",
    ) -> tuple[Figure, Axes]:
        # data = -data  # omega to w
        conf = ax.contourf(
            lon,
            lat,
            data,
            cmap="bwr",
            levels=np.arange(-0.5, 0.5, 0.05),
            zorder=0,
            extend="both",
        )

        if title:
            ax.set_title(title)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # colorbar
        cbar = fig.colorbar(conf, cax=cax)
        cbar.ax.set_title("$\\frac{m}{s}$")

        return fig, ax
