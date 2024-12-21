import importlib
from datetime import datetime

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils import DataCompose, DataGenerator
from visual import *


@hydra.main(version_base=None, config_path="config", config_name="predict")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, True)
    eval_cases = [datetime(2023, 7, 27, 0)]
    eval_cases.sort()

    # Inference
    if cfg.inference.infer_type == "ckpt":
        infer_machine = getattr(
            importlib.import_module("inference"), "BatchInferenceCkpt"
        )
    elif cfg.inference.infer_type == "onnx":
        infer_machine = getattr(
            importlib.import_module("inference"), "BatchInferenceOnnx"
        )
    infer_machine = infer_machine(cfg, eval_cases)
    infer_machine.infer()

    # Prepare lat/lon
    data_gnrt: DataGenerator = infer_machine.data_manager.data_gnrt
    dc_lat, dc_lon = DataCompose.from_config({"Lat": ["NoRule"], "Lon": ["NoRule"]})
    lat = data_gnrt.yield_data(datetime(2022, 10, 1, 0), dc_lat)
    lon = data_gnrt.yield_data(datetime(2022, 10, 1, 0), dc_lon)

    # Plot radar
    (data_compose,) = DataCompose.from_config({"Radar": ["Surface"]})
    radar_painter = VizRadar()
    for eval_case in tqdm(eval_cases, desc="Plot radar figures"):
        all_init_times = infer_machine.showcase_init_time_list(eval_case)
        gt, pred = infer_machine.get_figure_materials(eval_case, data_compose)
        fig, ax = radar_painter.plot_mxn(lon, lat, gt, pred, grid_on=True)
        fig.tight_layout()
        fig.savefig(
            f"./gallery/{data_compose}_{eval_case.strftime('%Y%m%d_%H%M')}_{cfg.inference.output_itv.hours}.png",
            transparent=False,
        )
        plt.close()

    # plot wind 850
    u_compose, v_compose = DataCompose.from_config({"U": ["Hpa850"], "V": ["Hpa850"]})
    wind_painter = VizWind(u_compose.level.value)
    for eval_case in tqdm(eval_cases, desc="Plot wind figures"):
        gt_u, pred_u = infer_machine.get_figure_materials(eval_case, u_compose)
        gt_v, pred_v = infer_machine.get_figure_materials(eval_case, v_compose)
        fig, ax = wind_painter.plot_mxn(lon, lat, gt_u, gt_v, pred_u, pred_v)
        fig.savefig(
            f"./gallery/{u_compose}_{eval_case.strftime('%Y%m%d_%H%M')}_{cfg.inference.output_itv.hours}.png",
            transparent=False,
        )
        plt.close()


if __name__ == "__main__":
    main()
