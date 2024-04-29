import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from src.models.architectures.earth_3d_specifics import EarthAttention3D

with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="train")

c = OmegaConf.to_container(cfg.model)
c.pop("model_name")
c["heads"] = c["heads"][0]

a = EarthAttention3D(**c)
b = torch.randn((4 * 3 * 14 * 14, 32, 192))
c = a.forward(b)
print(c.shape)
