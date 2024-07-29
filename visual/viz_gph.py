import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .tw_background import TwBackground


class VizGph(TwBackground):
    def __init__(self, pressure_level: int):
        """
        This class plot geopotential height (Z) for a given pressure level.
        If you have geopotential ($phi$), Z = phi / g0 where g0 = 9.80665 m/s^2.
        """
        super().__init__()
        self.press_lv = pressure_level
        self.title_suffix = f"GPH@{self.press_lv}Hpa"

    def plot_1xn(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        data: list[np.ndarray],
        titles: list[str] = [],
        grid_on: bool = False,
    ):
        if len(lat.shape) == 2 and len(lon.shape) == 2:
            lat = np.linspace(lat[0, 0], lat[-1, 0], lat.shape[0])
            lon = np.linspace(lon[0, 0], lon[0, -1], lon.shape[1])

        cols = len(data)

        plt.close()
        fig, ax = plt.subplots(1, cols, figsize=(20, 7), dpi=200, facecolor="w")
        for j in range(cols):
            tmp_ax = ax[j]
            title = titles[j] if titles else ""
            fig, tmp_ax = self.plot_bg(fig, tmp_ax, grid_on)
            fig, tmp_ax = self._plot_gph(fig, tmp_ax, lon, lat, data[j], title)

        return fig, ax

    def _plot_gph(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        data: np.ndarray,
        title: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        conf = ax.contourf(
            lon,
            lat,
            data,
            levels=np.linspace(5550, 5900, 15),
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
        cbar.ax.set_title("m")

        return fig, ax
