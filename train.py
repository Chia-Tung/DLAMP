import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.managers import DataManager
from src.utils import DataCompose, DataType, Level

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    log.info(f"Working directory : {Path.cwd()}")
    log.info(
        f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    # prevent access to non-existing keys
    OmegaConf.set_struct(cfg, True)

    # generate data
    data_list = []
    for var, value in cfg.data.train_data.items():
        for lv in value:
            data_list.append(DataCompose(DataType[var], Level[lv]))
    dm = DataManager(data_list, **cfg.data)
    dm.setup("fit")


if __name__ == "__main__":
    main()
