from pathlib import Path

import hydra
import torch

torch.set_float32_matmul_precision("medium")
from omegaconf import DictConfig, OmegaConf

from src.managers import DataManager
from src.models import get_builder
from src.utils import DataCompose


@hydra.main(version_base=None, config_path="config", config_name="train_pangu")
def main(cfg: DictConfig) -> None:
    # prevent access to non-existing key
    OmegaConf.set_struct(cfg, True)
    hydra_oup_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # prepare data
    data_list = DataCompose.from_config(cfg.data.train_data)
    data_manager = DataManager(data_list, **cfg.data, **cfg.lightning)
    data_manager.setup("test")
    data_manager.setup("fit")

    # model
    model_builder = get_builder(cfg.model.model_name)(
        hydra_oup_dir,
        data_list,
        image_shape=data_manager.image_shape,
        add_time_features=cfg.data.add_time_features,
        **cfg.model,
        **cfg.lightning,
    )
    model = model_builder.build_model(data_manager.test_dataloader())

    # inference
    data_loader = data_manager.train_dataloader()
    inp_data, _ = next(iter(data_loader))
    inp_data["upper_air"] = inp_data["upper_air"].cuda()
    inp_data["surface"] = inp_data["surface"].cuda()
    print(inp_data["upper_air"].shape, inp_data["surface"].shape)

    model = model_builder._backbone_model()
    model = model.cuda()
    with torch.no_grad():
        out = model(inp_data["upper_air"], inp_data["surface"])
    print(out[0].shape, out[1].shape)


if __name__ == "__main__":
    main()
