import geopandas as gpd
import matplotlib.pyplot as plt

from src.const import COUNTY_SHP_PATH


class TwBackground:
    def __init__(self):
        self.county_data = gpd.read_file(COUNTY_SHP_PATH)

    def plot_bg(self, rows=1, columns=1):
        """
        Plots the county data on a figure

        Parameters:
            rows (int): The number of rows in the plot grid. Defaults to 1.
            columns (int): The number of columns in the plot grid. Defaults to 1.

        Returns:
            matplotlib.axes._subplots.AxesSubplot: The matplotlib subplot containing the plot.
        """
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200, facecolor="w")
        ax = self.county_data.plot(
            ax=ax, color="none", edgecolor="k", linewidth=1, zorder=1
        )

        # canvas setting
        ax.set_xlim(118, 123.5)
        ax.set_ylim(20, 27)
        return fig, ax
