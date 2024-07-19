import logging
from datetime import datetime

import torch
from omegaconf import DictConfig
from tqdm import trange

from src.models.lightning_modules import PanguLightningModule

from .infer_utils import init_ort_instance, prediction_postprocess
from .inference_base import InferenceBase

log = logging.getLogger(__name__)


class BatchInferenceOnnx(InferenceBase):
    def __init__(self, cfg: DictConfig, eval_cases: list[datetime] | None = None):
        """
        This class can only inference by onnx runtime.
        """
        super().__init__(cfg, eval_cases)

    def _setup(self):
        self.ort_sess = init_ort_instance(
            gpu_id=self.cfg.inference.gpu_id, onnx_path=self.cfg.inference.onnx_path
        )

        log.info(f"onnx runtime session is ready on GPU {self.cfg.inference.gpu_id}")

    def infer(self):
        """
        Perform batch inference using ONNX runtime.

        This method iterates through the predict dataloader and stores
        intermediat results. The predictions are then post-processed
        and stored as attributes of the class.

        The method performs the following steps:
        1. Initializes data loader and calculates iteration parameters.
        2. Iterates through batches, performing inference for each time step.
        3. Stores intermediate results at specified intervals.
        4. Post-processes the collected results.
        5. Stores the final predictions as class attributes.

        Note:
            The number of iterations and storage interval are determined by
            `self.output_itv` and `self.showcase_length` respectively.
        """
        data_loader = self.data_manager.predict_dataloader()
        predict_iters = self.output_itv.seconds // 3600
        assert (
            predict_iters % self.showcase_length == 0
        ), "predict_iters must be divisible by showcase_length"
        store_interval = predict_iters // self.showcase_length

        ret = []
        for batch_id, (input, target) in enumerate(data_loader):
            inp_upper = input["upper_air"].numpy()
            inp_surface = input["surface"].numpy()
            for step in trange(predict_iters, desc=f"Infer batch {batch_id}"):
                ort_inputs = {
                    self.ort_sess.get_inputs()[0].name: inp_upper,
                    self.ort_sess.get_inputs()[1].name: inp_surface,
                }
                inp_upper, inp_surface = self.ort_sess.run(None, ort_inputs)
                if (step + 1) % store_interval == 0:
                    ret.append(
                        (
                            input["upper_air"],
                            input["surface"],
                            target["upper_air"],
                            target["surface"],
                            torch.from_numpy(inp_upper),
                            torch.from_numpy(inp_surface),
                        )
                    )
        mapping = PanguLightningModule.get_product_mapping()
        predictions = prediction_postprocess(ret, mapping)
        for product_type, tensor in predictions.items():
            setattr(self, product_type, tensor)

        log.info(f"Batch inference finished at {datetime.now()}")
