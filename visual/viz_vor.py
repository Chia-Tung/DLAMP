from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from src.const import FIGURE_PATH
from src.utils import gen_data
from src.utils.data_compose import DataCompose
from src.utils.data_type import DataType, Level

from .tw_background import TwBackground

# def cal_dist(lat1, lon1, lat2, lon2):
#     r = 6371
#     phi1 = np.radians(lat1)
#     phi2 = np.radians(lat2)
#     delta_phi = np.radians(lat2 - lat1)
#     delta_lambda = np.radians(lon2 - lon1)
#     a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
#     res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 -a)))
#     return np.round(res, 2)

class VizVor(TwBackground):
    def __init__(self):
        super().__init__()

    def plot(
        self, lon: np.ndarray, lat: np.ndarray, u_wind: np.ndarray, v_wind: np.ndarray
    ):
        fig, ax = super().plot_bg()

        # create dx and dy to calculate vorticity
        # dy = np.empty((lat.shape[0]-1, lon.shape[1]))
        # dx = np.empty((lat.shape[0], lon.shape[1]-1))
        # for i in range(lat.shape[0]-1):
        #     for j in range(lon.shape[1]):
        #         dy[i,j] = cal_dist(lat[i,j], lon[i,j], lat[i+1,j], lon[i,j])
        # for i in range(lat.shape[0]):
        #     for j in range(lon.shape[1]-1):
        #         dx[i,j] = cal_dist(lat[i,j], lon[i,j], lat[i,j], lon[i,j+1])

        # calculate vorticity
        vor_u = (u_wind[1:,:]-u_wind[0:-1,:])/(2*1000)
        vor_u = np.concatenate((np.reshape(vor_u[0,:],(1,lon.shape[1])),vor_u),axis=0)
        vor_v = (v_wind[:,1:]-v_wind[:,0:-1])/(2*1000)
        vor_v = np.concatenate((np.reshape(vor_v[:,0],(lat.shape[0],1)),vor_v),axis=1)
        vor = vor_u*(-1) + vor_v

        # plot data
        conf = ax.contourf(
            lon,
            lat,
            10**5*vor,
            np.arange(-60,61,2),
            cmap="bwr",
            zorder=0,
            extend='both'
        )

        # colorbar
        cbar = fig.colorbar(conf, ax=ax)
        cbar.ax.set_title("vorticity ($10^{-5} s^{-1}$)")

        return fig, ax


if __name__ == "__main__":
    # `export PYTHONPATH=$PYTHONPATH:/wk171/handsomedong/DLAMP` in CLI
    target_time = datetime(2022, 9, 3, 0)
    u850 = gen_data(target_time, DataCompose(DataType.U, Level.Hpa850))
    v850 = gen_data(target_time, DataCompose(DataType.V, Level.Hpa850))
    data_lat = gen_data(target_time, DataCompose(DataType.Lat, Level.Surface))
    data_lon = gen_data(target_time, DataCompose(DataType.Lon, Level.Surface))

    viz = VizVor()
    fig, ax = viz.plot(data_lon, data_lat, u850, v850)
    fig.savefig(
        f"{FIGURE_PATH}/{target_time.strftime('%Y%m%d_%H%M')}_vorticity.png",
        transparent=False,
    )
    plt.close()
