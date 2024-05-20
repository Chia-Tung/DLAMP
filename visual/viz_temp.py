from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from src.const import FIGURE_PATH, TEMP_COLOR, TEMP_LV
from src.utils import DataCompose, DataType, Level, gen_data

from .tw_background import TwBackground


class VizTemp(TwBackground):
    def __init__(self):
        super().__init__()

    def plot(self, lon: np.ndarray, lat: np.ndarray, temp: np.ndarray):
        fig, ax = super().plot_bg()

        # temperature
        temp_c = temp - 273.15

        # plot data
        conf = ax.contourf(
            lon,
            lat,
            temp_c,
            levels=TEMP_LV,
            colors=TEMP_COLOR,
            zorder=0,
        )

        # colorbar
        cbar = fig.colorbar(conf, ax=ax)
        cbar.ax.set_title("temperature ($^{o}$C)")

        return fig, ax


if __name__ == "__main__":
    target_time = datetime(2022, 10, 16, 0)
    t850 = gen_data(target_time, DataCompose(DataType.T, Level.Hpa850))
    data_lat = gen_data(target_time, DataCompose(DataType.Lat, Level.Surface))
    data_lon = gen_data(target_time, DataCompose(DataType.Lon, Level.Surface))

    viz = VizTemp()
    fig, ax = viz.plot(data_lon, data_lat, t850)
    fig.savefig(
        f"{FIGURE_PATH}/{target_time.strftime('%Y%m%d_%H%M')}_temperature.png",
        transparent=False,
    )
    plt.close()
