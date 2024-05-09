import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.managers import DataManager
from src.models import get_builder
from src.utils import DataCompose

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    log.info(f"Working directory: {Path.cwd()}")
    log.info(
        f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    # prevent access to non-existing keys
    OmegaConf.set_struct(cfg, True)

    # generate data
    data_list = DataCompose.from_config(cfg.data.train_data)
    data_manager = DataManager(data_list, **cfg.data, **cfg.hparams)
    data_manager.setup("fit")

    # model
    model_builder = get_builder(cfg.model.model_name)(
        data_list, **cfg.model, **cfg.hparams
    )
    model = model_builder.build()

    # trainer


if __name__ == "__main__":
    main()
