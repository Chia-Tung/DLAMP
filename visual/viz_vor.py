from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from src.const import FIGURE_PATH
from src.utils import DataCompose, DataType, Level, gen_data

from .tw_background import TwBackground


class VizVor(TwBackground):
    def __init__(self):
        super().__init__()

    def plot(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        grid_point_resolution: np.ndarray,
    ):
        fig, ax = super().plot_bg()

        # calculate vorticity
        vor_u = (u_wind[1:, :] - u_wind[0:-1, :]) / grid_point_resolution[1]
        vor_u = np.concatenate([vor_u[0:1, :], vor_u], axis=0)
        vor_v = (v_wind[:, 1:] - v_wind[:, 0:-1]) / grid_point_resolution[0]
        vor_v = np.concatenate([vor_v[:, 0:1], vor_v], axis=1)
        vor = vor_v - vor_u

        # plot data
        conf = ax.contourf(
            lon,
            lat,
            10**5 * vor,
            np.arange(-100, 101, 2),
            cmap="bwr",
            zorder=0,
            extend="both",
        )

        # colorbar
        cbar = fig.colorbar(conf, ax=ax)
        cbar.ax.set_title("vorticity ($10^{-5} s^{-1}$)")
        cbar.set_ticks(np.arange(-100, 101, 20))

        return fig, ax


if __name__ == "__main__":
    target_time = datetime(2022, 10, 16, 0)
    u850 = gen_data(target_time, DataCompose(DataType.U, Level.Hpa850))
    v850 = gen_data(target_time, DataCompose(DataType.V, Level.Hpa850))
    data_lat = gen_data(target_time, DataCompose(DataType.Lat, Level.Surface))
    data_lon = gen_data(target_time, DataCompose(DataType.Lon, Level.Surface))
    grid_point_resolution = [2000, 2000]  # unit: m (for lon/lat)

    viz = VizVor()
    fig, ax = viz.plot(
        data_lon,
        data_lat,
        u850,
        v850,
        grid_point_resolution,
    )
    fig.savefig(
        f"{FIGURE_PATH}/{target_time.strftime('%Y%m%d_%H%M')}_vorticity.png",
        transparent=False,
    )
    plt.close()
