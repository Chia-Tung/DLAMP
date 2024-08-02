import geopandas as gpd
import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.const import COUNTY_SHP_PATH


class TwBackground:
    def __init__(self):
        self.canvas_settings()
        self.county_data = gpd.read_file(COUNTY_SHP_PATH)

    def canvas_settings(self) -> None:
        font = {"family": "sans-serif", "weight": "bold", "size": 14}
        axes = {
            "titlesize": 16,
            "titleweight": "bold",
            "labelsize": 14,
            "labelweight": "bold",
        }
        mpl.rc("font", **font)  # pass in the font dict as kwargs
        mpl.rc("axes", **axes)

    def plot_bg(
        self, fig: Figure, ax: Axes, grid_on: bool = False
    ) -> tuple[Figure, Axes]:
        """
        Plots the county data on a figure
        """
        fig.tight_layout()
        # fig.patch.set_visible(False)
        # ax.axis("off")
        ax = self.county_data.plot(
            ax=ax, color="none", edgecolor="k", linewidth=1, zorder=1
        )

        # canvas setting
        # ax.set_xlim(118, 123.5) # QPESUMS
        # ax.set_ylim(20, 27) # QPESUMS
        ax.set_xlim(116, 125.7)  # RWRF
        ax.set_ylim(19.4, 28)  # RWRF
        # ax.set_xlim(116.5, 125)  # interp
        # ax.set_ylim(19.75, 27.75)  # interp

        # default grid zorder is 2.5
        if grid_on:
            ax.grid(True, linestyle="--", color="k", alpha=0.8)

        return fig, ax
