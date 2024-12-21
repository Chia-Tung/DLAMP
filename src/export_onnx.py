import hydra
from omegaconf import DictConfig, OmegaConf

from src.managers import DataManager
from src.models import PanguLightningModule, get_builder
from src.utils import DataCompose


@hydra.main(version_base=None, config_path="../config", config_name="predict")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, True)

    # prepare data
    data_list = DataCompose.from_config(cfg.data.train_data)
    data_manager = DataManager(data_list, **cfg.data, **cfg.lightning)
    data_manager.setup("fit")

    # sample data
    data_loader = data_manager.train_dataloader()
    inp_data, oup_data = next(iter(data_loader))
    inp_data["upper_air"] = inp_data["upper_air"].to("cuda")
    inp_data["surface"] = inp_data["surface"].to("cuda")

    # model builder
    model_builder = get_builder(cfg.model.model_name)(
        "export_onnx",
        data_list,
        image_shape=data_manager.image_shape,
        **cfg.model,
        **cfg.lightning
    )

    # load LightningModule from checkpoint
    pl_module = PanguLightningModule.load_from_checkpoint(
        checkpoint_path=cfg.inference.best_ckpt,
        test_dataloader=None,
        backbone_model=model_builder._backbone_model(),
    )

    # export onnx
    pl_module = pl_module.cuda()
    date = cfg.inference.best_ckpt.split("_")[1]  # e.g. 240831
    pl_module.to_onnx(
        file_path=f"./export/{cfg.model.model_name}_model_{date}.onnx",
        input_sample=(inp_data["upper_air"], inp_data["surface"]),
        export_params=True,
        verbose=False,
        input_names=["input_upper", "input_surface"],
        output_names=["output_upper", "output_surface"],
        dynamic_axes={
            "input_upper": {0: "batch_size"},
            "input_surface": {0: "batch_size"},
            "output_upper": {0: "batch_size"},
            "output_surface": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    main()
