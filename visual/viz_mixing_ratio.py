import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .tw_background import TwBackground


class VizMixingRatio(TwBackground):
    def __init__(self, pressure_level: int | None = None):
        super().__init__()
        self.press_lv = pressure_level
        self.title_suffix = f"Q@{self.press_lv}" if pressure_level else ""

    def plot_1xn(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: list[np.ndarray],
        titles: list[str] = [],
        grid_on: bool = False,
    ):
        cols = len(data)

        plt.close()
        fig, ax = plt.subplots(1, cols, figsize=(15, 2.5), dpi=200, facecolor="w")
        for j in range(cols):
            tmp_ax = ax[j]
            title = titles[j] if titles else ""
            fig, tmp_ax = self.plot_bg(fig, tmp_ax, grid_on)
            fig, tmp_ax = self._plot_q(fig, tmp_ax, lon, lat, data[j], title)

        return fig, ax

    def plot_1x1(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        title: str = "",
    ) -> tuple[Figure, Axes]:
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200, facecolor="w")
        fig, ax = super().plot_bg(fig, ax)
        fig, ax = self._plot_q(fig, ax, lon, lat, data, title)

        return fig, ax

    def _plot_q(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        title: str = "",
    ) -> tuple[Figure, Axes]:
        # data *= 1000  # kg/kg -> g/kg
        conf = ax.contourf(
            lon,
            lat,
            data,
            levels=np.arange(0, 2.1, 0.1),
            cmap="BuPu",
            # cmap="BrBG",
            zorder=0,
        )

        if title:
            ax.set_title(f"{title} {self.title_suffix}")

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # colorbar
        cbar = fig.colorbar(conf, cax=cax)
        cbar.ax.set_title("g/kg")

        return fig, ax
