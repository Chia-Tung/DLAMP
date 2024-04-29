import abc

from lightning import LightningModule


class BaseBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def build(self) -> LightningModule:
        return NotImplemented
