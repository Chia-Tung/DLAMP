from datetime import datetime, timedelta

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from inference import prediction_postprocess
from src.managers import DataManager
from src.models import get_builder
from src.utils import DataCompose
from visual import *


@hydra.main(version_base=None, config_path="config", config_name="predict")
def main(cfg: DictConfig) -> None:
    # prevent access to non-existing keys
    OmegaConf.set_struct(cfg, True)

    # prepare data
    data_list = DataCompose.from_config(cfg.data.train_data)
    data_manager = DataManager(data_list, **cfg.data, **cfg.lightning)

    # model
    model_builder = get_builder(cfg.model.model_name)(
        "predict", data_list, **cfg.model, **cfg.lightning
    )
    model = model_builder.build_model()
    mapping = model.get_product_mapping()

    # trainer.predict()
    trainer = model_builder.build_trainer(False)
    best_ckpt = "./checkpoints/Pangu_240514_002602-epoch=045-val_loss_epoch=2.9630.ckpt"
    predictions = trainer.predict(model, data_manager, ckpt_path=best_ckpt)
    predictions = prediction_postprocess(predictions, mapping)

    # plot figures
    data_gnrt = data_manager.data_gnrt
    data_lat, data_lon = data_gnrt.yield_data(
        datetime(2022, 10, 1, 0), {"Lat": ["NoRule"], "Lon": ["NoRule"]}
    )
    time_list = data_manager.dtm.ordered_predict_time

    # prepare input output pair
    # (predictions), (product_type), (find time index: figure_columns, target time, time_list), (level and variable)
    cases = [datetime(2021, 8, 7), datetime(2022, 9, 12)]
    for case in cases:
        for offset in trange(cfg.plot.figure_columns):
            time_offset = timedelta(**cfg.data.time_interval) * offset
            target_time = case + time_offset
            time_index = time_list.index(target_time)

            viz = VizRadar()
            for product_type in ["input_sruface", "output_surface"]:
                data_radar = predictions[product_type][time_index].squeeze()
                fig, ax = viz.plot(
                    data_lon,
                    data_lat,
                    data_radar,
                    title=target_time.strftime("%Y%m%d_%H%MUTC"),
                    grid_on=True,
                )
                fig.savefig(
                    f"./gallery/radar_{product_type}_{target_time.strftime('%Y%m%d_%H%M')}.png",
                    transparent=False,
                )
                plt.close()

            pressure_level = "Hpa850"
            wind = VizWind(pressure_level)
            for product_type in ["input_upper", "output_upper"]:
                level_idx = model_builder.pressure_levels.index(pressure_level)
                var_u_idx = model_builder.upper_vars.index("U")
                var_v_idx = model_builder.upper_vars.index("V")
                u_wind = predictions[product_type][
                    time_index, level_idx, :, :, var_u_idx
                ]
                v_wind = predictions[product_type][
                    time_index, level_idx, :, :, var_v_idx
                ]
                fig, ax = wind.plot(
                    data_lon,
                    data_lat,
                    u_wind,
                    v_wind,
                    title=target_time.strftime("%Y%m%d_%H%MUTC"),
                )
                fig.savefig(
                    f"./gallery/wind_{pressure_level}_{product_type}_{target_time.strftime('%Y%m%d_%H%M')}.png",
                    transparent=False,
                )
                plt.close()


if __name__ == "__main__":
    main()
