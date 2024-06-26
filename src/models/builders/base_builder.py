import abc
import logging

from lightning import LightningModule, Trainer
from omegaconf import OmegaConf


class BaseBuilder(metaclass=abc.ABCMeta):

    def __init__(self, *args, **kwargs):
        self.kwargs = OmegaConf.create(kwargs)
        self.log = logging.getLogger(__name__)
        self.log.info(f"Use Builder: {self.__class__.__name__}")

    def info_log(self, content: str):
        self.log.info(f"[{self.__class__.__name__}] {content}")

    @abc.abstractmethod
    def build_model(self) -> LightningModule:
        return NotImplemented

    @abc.abstractmethod
    def build_trainer(self) -> Trainer:
        return NotImplemented
