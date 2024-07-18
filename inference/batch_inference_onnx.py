from datetime import datetime

from omegaconf import DictConfig

from .infer_utils import init_ort_instance
from .inference_base import InferenceBase


class BatchInferenceOnnx(InferenceBase):
    def __init__(self, cfg: DictConfig, eval_cases: list[datetime] | None = None):
        """
        This class can only inference by onnx runtime.
        """
        super().__init__(cfg, eval_cases)

    def _setup(self):
        ort_sess = init_ort_instance(
            gpu_id=self.cfg.gpu_id, onnx_path=self.cfg.onnx_path
        )

    def infer(self):
        pass
