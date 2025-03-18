import importlib
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm, trange

from inference import InferenceBase
from src.const import DATA_PATH, FIGURE_PATH
from src.utils import DataCompose, DataGenerator, read_cwa_ncfile
from visual import *


@hydra.main(version_base=None, config_path="config", config_name="predict")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, True)
    eval_cases = [
        datetime(2020, 5, 21, 17),  # Meiyu
        datetime(2022, 9, 11),  # TC MUIFA
        datetime(2024, 10, 31, 2),  # TC Kong-rey
    ]
    eval_cases.sort()

    # Inference
    if cfg.inference.infer_type == "ckpt":
        infer_machine: InferenceBase = getattr(
            importlib.import_module("inference"), "BatchInferenceCkpt"
        )
        save_name = cfg.inference.best_ckpt.split("/")[-1].split("-")[0]
    elif cfg.inference.infer_type == "onnx":
        infer_machine: InferenceBase = getattr(
            importlib.import_module("inference"), "BatchInferenceOnnx"
        )
        save_name = cfg.inference.onnx_path.split("/")[-1].split(".")[0]
    infer_machine = infer_machine(cfg, eval_cases)
    infer_machine.infer(bdy_swap_method=cfg.inference.bdy_swap_method)

    # Prepare lat/lon
    data_gnrt: DataGenerator = infer_machine.data_manager.data_gnrt
    dc_lat, dc_lon = DataCompose.from_config({"Lat": ["NoRule"], "Lon": ["NoRule"]})
    start_t = datetime.strptime(cfg.data.start_time, cfg.data.format)
    lat = data_gnrt.yield_data(start_t, dc_lat)
    lon = data_gnrt.yield_data(start_t, dc_lon)

    # Prepare painter
    u_compose, v_compose = DataCompose.from_config({"U": ["Hpa850"], "V": ["Hpa850"]})
    painter_gt = VizWind(u_compose.level.name)
    painter_pd = VizWind()
    itv = infer_machine.output_itv // infer_machine.data_itv

    """""" """""" """""" """
        Plot wind 850 1xn
    """ """""" """""" """"""
    for eval_case in tqdm(eval_cases, desc="Plot wind figures"):
        gt_u, pd_u = infer_machine.get_figure_materials(eval_case, u_compose)
        gt_v, pd_v = infer_machine.get_figure_materials(eval_case, v_compose)

        t_list, i_list = [], []
        for i in range(infer_machine.showcase_length):
            curr_time = eval_case + infer_machine.output_itv * i
            t_list.append(curr_time.strftime("%Y-%m-%d %HZ"))
            i_list.append(
                f"Init: {eval_case.strftime('%Y-%m-%d %HZ')} Fcst: +{i*itv:02d}H"
            )

        # ground truth
        fig, _ = painter_gt.plot_1xn(lon, lat, gt_u, gt_v, titles=t_list)
        remark = f"itv-{itv}_len-{infer_machine.showcase_length}"
        save_figure(fig, u_compose, save_name, eval_case, None, "gt", "1xn", remark)

        # prediction
        fig, _ = painter_pd.plot_1xn(lon, lat, pd_u, pd_v, titles=i_list)
        dtype = "pd_bdy" if cfg.inference.bdy_swap_method else "pd"
        save_figure(fig, u_compose, save_name, eval_case, None, dtype, "1xn", remark)

        # operational RWRF
        rwrf_dir = f"{DATA_PATH}RWRF_{eval_case.strftime('%Y-%m')}/{eval_case.strftime('%Y-%m-%d_%H')}"
        u_tmp, v_tmp = [], []
        for i in trange(infer_machine.showcase_length, desc="Get RWRF data"):
            curr_time = eval_case + infer_machine.output_itv * i
            filename = Path(
                f"{rwrf_dir}/wrfout_d01_{curr_time.strftime('%Y-%m-%d_%H')}_interp"
            )
            u = read_cwa_ncfile(filename, u_compose)[1:449:2, 1:449:2]  # (224, 224)
            v = read_cwa_ncfile(filename, v_compose)[1:449:2, 1:449:2]  # (224, 224)

            u_tmp.append(u)
            v_tmp.append(v)

        wrf_u, wrf_v = np.stack(u_tmp), np.stack(v_tmp)  # (N, 336, 336)
        fig, _ = painter_pd.plot_1xn(lon, lat, wrf_u, wrf_v, titles=i_list)
        save_figure(fig, u_compose, save_name, eval_case, None, "rwrf", "1xn", remark)

    """""" """""" """""" """
        Plot wind 850 1x1
    """ """""" """""" """"""
    for eval_case in tqdm(eval_cases, desc="Plot wind figures"):
        gt_u, pd_u = infer_machine.get_figure_materials(eval_case, u_compose)
        gt_v, pd_v = infer_machine.get_figure_materials(eval_case, v_compose)

        for i in range(infer_machine.showcase_length):
            curr_time = eval_case + infer_machine.output_itv * i

            # ground truth
            title = f"{curr_time.strftime('%Y-%m-%d %HZ')}"
            fig, _ = painter_gt.plot_1x1(lon, lat, gt_u[i], gt_v[i], title)
            save_figure(fig, u_compose, save_name, eval_case, curr_time, "gt", "1x1")

            # prediction
            title = f"Init: {eval_case.strftime('%Y-%m-%d %HZ')} Fcst: +{i*itv:02d}H"
            fig, _ = painter_pd.plot_1x1(lon, lat, pd_u[i], pd_v[i], title)
            dtype = "pd_bdy" if cfg.inference.bdy_swap_method else "pd"
            save_figure(fig, u_compose, save_name, eval_case, curr_time, dtype, "1x1")


def save_figure(
    fig: Figure,
    dc: DataCompose,
    model_name: str,
    init_time: datetime,
    curr_time: datetime | None,
    data_type: str,
    fig_type: str,
    remark: str = "",
):
    """
    curr_time: only be used in "1x1" figures, else None
    data_type: "gt" | "pd" | "pd_bdy"
    fig_type: "1x1" | "1xn"
    remark: "itv-{itv}_len-{len}"
    """
    filename = (
        f"{dc.var_name.name}_{dc.level.name}_{curr_time.strftime('%Y%m%d_%H%M')}.png"
        if curr_time
        else f"{dc.var_name.name}_{dc.level.name}_{remark}.png"
    )

    filepath = (
        Path(FIGURE_PATH)
        / model_name
        / f"init_{init_time.strftime('%Y%m%d_%H%M')}"
        / data_type
        / fig_type
        / filename
    )
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(filepath, transparent=False)
    plt.close()


if __name__ == "__main__":
    main()
