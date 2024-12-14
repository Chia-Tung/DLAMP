import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .tw_background import TwBackground


class VizPressure(TwBackground):
    def __init__(self, pressure_level: int | None = None):
        super().__init__()
        self.press_lv = pressure_level
        self.title_suffix = f"GPH@{self.press_lv}Hpa" if pressure_level else ""

    def plot_1x1(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        title: str = "",
        grid_on: bool = False,
    ) -> tuple[Figure, Axes]:
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200, facecolor="w")
        fig, ax = super().plot_bg(fig, ax, grid_on)
        fig, ax = self._plot_pressure(fig, ax, lon, lat, data, title)

        return fig, ax

    def _plot_pressure(
        self,
        fig: Figure,
        ax: Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        title: str = "",
    ) -> tuple[Figure, Axes]:
        conf = ax.contourf(
            lon,
            lat,
            data,
            cmap="viridis",
            zorder=0,
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
            ax.set_title(f"{title} {self.title_suffix}")

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # colorbar
        cbar = fig.colorbar(conf, cax=cax)
        cbar.ax.set_title("hpa")

        return fig, ax
