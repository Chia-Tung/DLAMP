from datetime import datetime

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from inference import BatchInference
from src.utils import DataCompose
from visual import *


@hydra.main(version_base=None, config_path="config", config_name="predict")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, True)
    eval_cases = [datetime(2021, 8, 7), datetime(2022, 9, 12)]

    # Inference
    infer_machine = BatchInference(cfg, eval_cases)
    infer_machine.infer()

    # Prepare lat/lon
    data_gnrt = infer_machine.data_manager.data_gnrt
    lat, lon = data_gnrt.yield_data(
        datetime(2022, 10, 1, 0), {"Lat": ["NoRule"], "Lon": ["NoRule"]}
    )

    # Plot radar
    (data_compose,) = DataCompose.from_config({"Radar": ["Surface"]})
    radar_painter = VizRadar()
    for eval_case in tqdm(eval_cases, desc="Plot radar figures"):
        all_init_times = infer_machine.showcase_init_time_list(eval_case)
        gt, pred = infer_machine.get_figure_materials(eval_case, data_compose)
        fig, ax = radar_painter.plot_mxn(lon, lat, gt, pred, grid_on=True)
        fig.savefig(
            f"./gallery/{data_compose}_{eval_case.strftime('%Y%m%d_%H%M')}.png",
            transparent=False,
        )
        plt.close()

    # plot wind 850
    u_compose, v_compose = DataCompose.from_config({"U": ["Hpa850"], "V": ["Hpa850"]})
    wind_painter = VizWind(u_compose.level.name)
    for eval_case in tqdm(eval_cases, desc="Plot wind figures"):
        all_init_times = infer_machine.showcase_init_time_list(eval_case)
        gt_u, pred_u = infer_machine.get_figure_materials(eval_case, u_compose)
        gt_v, pred_v = infer_machine.get_figure_materials(eval_case, v_compose)
        fig, ax = wind_painter.plot_mxn(lon, lat, gt_u, gt_v, pred_u, pred_v)
        fig.savefig(
            f"./gallery/{u_compose}_{eval_case.strftime('%Y%m%d_%H%M')}.png",
            transparent=False,
        )
        plt.close()


if __name__ == "__main__":
    main()
