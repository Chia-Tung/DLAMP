import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.const import COUNTY_SHP_PATH


class TwBackground:
    def __init__(self):
        self.canvas_settings()
        self.county_data = gpd.read_file(COUNTY_SHP_PATH)

    def canvas_settings(self):
        font = {"family": "sans-serif", "weight": "bold", "size": 14}
        axes = {
            "titlesize": 16,
            "titleweight": "bold",
            "labelsize": 14,
            "labelweight": "bold",
        }
        mpl.rc("font", **font)  # pass in the font dict as kwargs
        mpl.rc("axes", **axes)

    def plot_bg(self, grid_on: bool = False):
        """
        Plots the county data on a figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200, facecolor="w")
        ax = self.county_data.plot(
            ax=ax, color="none", edgecolor="k", linewidth=1, zorder=1
        )

        # canvas setting
        # ax.set_xlim(118, 123.5) # QPESUMS
        # ax.set_ylim(20, 27) # QPESUMS
        ax.set_xlim(116, 125.7)
        ax.set_ylim(19.4, 28)

        # default grid zorder is 2.5
        if grid_on:
            ax.grid(True, linestyle="--", color='k', alpha=0.8)

        return fig, ax
