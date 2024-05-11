import logging
from pathlib import Path

import hydra
from lightning.pytorch import seed_everything
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
    seed_everything(1000)

    # prevent access to non-existing keys
    OmegaConf.set_struct(cfg, True)

    # prepare data
    data_list = DataCompose.from_config(cfg.data.train_data)
    data_manager = DataManager(data_list, **cfg.data, **cfg.lightning)

    # model
    model_builder = get_builder(cfg.model.model_name)(
        data_list, **cfg.model, **cfg.lightning
    )
    model = model_builder.build_model()

    # trainer
    wandb_logger = model_builder.wandb_logger()
    wandb_logger.watch(model, log="all")
    trainer = model_builder.build_trainer(wandb_logger)

    # start training
    trainer.fit(model, data_manager)


if __name__ == "__main__":
    main()
