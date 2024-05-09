from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from src.const import FIGURE_PATH, WSP_COLOR, WSP_LV
from src.utils import gen_data

from .tw_background import TwBackground


class VizWind(TwBackground):
    def __init__(self):
        super().__init__()

    def plot(
        self, lon: np.ndarray, lat: np.ndarray, u_wind: np.ndarray, v_wind: np.ndarray
    ):
        fig, ax = super().plot_bg()

        # wind speed
        scalar = np.hypot(u_wind, v_wind)

        # plot data
        conf = ax.contourf(
            lon,
            lat,
            scalar,
            levels=WSP_LV,
            colors=WSP_COLOR,
            zorder=0,
        )
        # since lat/lon is not monotonically increasing in a same pace
        lat_1d = np.linspace(lat[0, 0], lat[-1, 0], lat.shape[0])
        lon_1d = np.linspace(lon[0, 0], lon[0, -1], lon.shape[1])
        ax.streamplot(lon_1d, lat_1d, u_wind, v_wind, zorder=0, color="C0")

        # colorbar
        cbar = fig.colorbar(conf, ax=ax)
        cbar.ax.set_title("speed")

        return fig, ax


if __name__ == "__main__":
    target_time = datetime(2022, 10, 1, 0)
    u850 = gen_data(target_time, {"var": "U", "lv": "Hpa850"})
    v850 = gen_data(target_time, {"var": "V", "lv": "Hpa850"})
    data_lat = gen_data(target_time, {"var": "Lat", "lv": "Surface"})
    data_lon = gen_data(target_time, {"var": "Lon", "lv": "Surface"})

    viz = VizWind()
    fig, ax = viz.plot(data_lon, data_lat, u850, v850)
    fig.savefig(
        f"{FIGURE_PATH}/{target_time.strftime('%Y%m%d_%H%M')}_wind.png",
        transparent=False,
    )
    plt.close()
