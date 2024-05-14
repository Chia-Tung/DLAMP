from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from src.const import FIGURE_PATH, WSP_COLOR, WSP_LV
from src.utils import gen_data
from src.utils.data_compose import DataCompose
from src.utils.data_type import DataType, Level

from .tw_background import TwBackground


class VizWind(TwBackground):
    def __init__(self):
        super().__init__()

    def plot(
        self, lon: np.ndarray, lat: np.ndarray, u_wind: np.ndarray, v_wind: np.ndarray
    ):
        # since lat/lon may not be monotonically increasing in a same pace
        if len(lat.shape) == 2 and len(lon.shape) == 2:
            lat = np.linspace(lat[0, 0], lat[-1, 0], lat.shape[0])
            lon = np.linspace(lon[0, 0], lon[0, -1], lon.shape[1])

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
        ax.streamplot(lon_1d, lat_1d, u_wind, v_wind, zorder=0, color='k')

        # colorbar
        cbar = fig.colorbar(conf, ax=ax)
        cbar.ax.set_title("speed")

        return fig, ax


if __name__ == "__main__":
    # `export PYTHONPATH=$PYTHONPATH:/wk171/handsomedong/DLAMP` in CLI
    target_time = datetime(2022, 9, 3, 0)
    u850 = gen_data(target_time, DataCompose(DataType.U, Level.Hpa850))
    v850 = gen_data(target_time, DataCompose(DataType.V, Level.Hpa850))
    data_lat = gen_data(target_time, DataCompose(DataType.Lat, Level.Surface))
    data_lon = gen_data(target_time, DataCompose(DataType.Lon, Level.Surface))

    viz = VizWind()
    fig, ax = viz.plot(data_lon, data_lat, u850, v850)
    fig.savefig(
        f"{FIGURE_PATH}/{target_time.strftime('%Y%m%d_%H%M')}_wind.png",
        transparent=False,
    )
    plt.close()
