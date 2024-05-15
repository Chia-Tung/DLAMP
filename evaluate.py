from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.managers import DataManager
from src.models import get_builder
from src.utils import DataCompose


@hydra.main(version_base=None, config_path="config", config_name="evaluate")
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

    # predict
    trainer = model_builder.build_trainer(False)
    best_ckpt = "./checkpoints/Pangu_240514_002602-epoch=045-val_loss_epoch=2.9630.ckpt"

    # shape: [epochs][6][B, lv, h, w, c]
    predictions = trainer.predict(model, data_manager, ckpt_path=best_ckpt)
    output = defaultdict(list)
    for epoch_id in range(len(predictions)):
        for output_type in range(6):
            tmp = predictions[epoch_id][output_type]
            output[output_type].append(tmp)

    for k, v in output.items():
        output[k] = torch.cat(v, dim=0)  # {0: (B, lv, h, w, c)...}

    """ TO BE CONTINUED
    from visual import *
    from datetime import datetime, timedelta
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import trange

    case_A = datetime(2022, 9, 12)
    case_B = datetime(2021, 8, 7)

    data_gnrt = data_manager.data_gnrt
    data_lat, data_lon = data_gnrt.yield_data(
        datetime(2022, 10, 1, 0), {"Lat": ["NoRule"], "Lon": ["NoRule"]}
    )

    mapping = {0: "input_upper", 1: "input_sruface", 2: "target_upper", 3: "target_surface", 4: "output_upper", 5: "output_surface"}

    for m in trange(4):
        一次畫多圖
        offset = timedelta(hours=1) * m
        target_time = case_B + offset

        time_list = data_manager.dtm.ordered_predict_time
        time_list = np.array(time_list)
        index = np.where(time_list == target_time)[0].item()

        viz = VizRadar()
        for i in [1, 5]:
            data_radar = output[i][index].squeeze()
            fig, ax = viz.plot(data_lon, data_lat, data_radar, title=target_time.strftime('%Y%m%d_%H%MUTC'), grid_on=True)
            fig.tight_layout()
            fig.savefig(f"./gallery/radar_{mapping[i]}_{target_time.strftime('%Y%m%d_%H%M')}.png", transparent=True)
            plt.close()

        wind = VizWind(pressure_level=850)
        高度mapping在盤古builder裡面
        for i in [0, 4]:
            u_wind = output[i][index, 3, :, :, 2]
            v_wind = output[i][index, 3, :, :, 3]
            fig, ax = wind.plot(data_lon, data_lat, u_wind, v_wind, title=target_time.strftime('%Y%m%d_%H%MUTC'))
            fig.tight_layout()
            fig.savefig(f"./gallery/wind_{mapping[i]}_{target_time.strftime('%Y%m%d_%H%M')}.png", transparent=True)
            plt.close()
    """


if __name__ == "__main__":
    main()
