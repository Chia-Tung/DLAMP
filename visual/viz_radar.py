from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from src.const import DBZ_COLOR, DBZ_LV, DBZ_NORM, FIGURE_PATH
from src.utils import gen_data

from .tw_background import TwBackground


class VizRadar(TwBackground):
    def __init__(self):
        super().__init__()

    def plot(self, lon: np.ndarray, lat: np.ndarray, data: np.ndarray):
        fig, ax = super().plot_bg()

        # plot data
        ax.pcolormesh(
            lon,
            lat,
            data,
            edgecolors="none",
            shading="auto",
            norm=DBZ_NORM,
            cmap=DBZ_COLOR,
            zorder=0,
        )

        # colorbar
        cbar = fig.colorbar(
            cm.ScalarMappable(norm=DBZ_NORM, cmap=DBZ_COLOR),
            ax=ax,
            ticks=np.arange(DBZ_LV[0], DBZ_LV[-1] + 1, 5),
        )
        cbar.ax.set_title("dBZ")

        return fig, ax


if __name__ == "__main__":
    # `export PYTHONPATH=$PYTHONPATH:/wk171/handsomedong/DLAMP` in CLI
    target_time = datetime(2022, 10, 1, 0)
    data_radar = gen_data(target_time, {"var": "Radar", "lv": "NoRule"})
    data_lat = gen_data(target_time, {"var": "Lat", "lv": "Surface"})
    data_lon = gen_data(target_time, {"var": "Lon", "lv": "Surface"})

    viz = VizRadar()
    fig, ax = viz.plot(data_lon, data_lat, data_radar)
    fig.savefig(
        f"{FIGURE_PATH}/{target_time.strftime('%Y%m%d_%H%M')}_mos.png",
        transparent=False,
    )
    plt.close()
