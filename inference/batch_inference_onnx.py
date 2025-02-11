import logging
from datetime import datetime, timedelta

import numpy as np
from omegaconf import DictConfig
from tqdm import trange

from src.models.lightning_modules import PanguLightningModule
from src.utils import TimeUtil

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

    def infer(self, is_bdy_swap: bool = False):
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
        interval = self.output_itv // self.data_itv
        predict_iters = (self.showcase_length - 1) * interval

        ret = []
        for batch_id, (input, target) in enumerate(data_loader):
            inp_upper = input["upper_air"].cpu().numpy()
            inp_surface = input["surface"].cpu().numpy()

            # auto-regression
            tmp_upper, tmp_sfc = [], []
            for step in trange(predict_iters, desc=f"Infer batch {batch_id}"):
                ort_inputs = {
                    self.ort_sess.get_inputs()[0].name: inp_upper,
                    self.ort_sess.get_inputs()[1].name: inp_surface,
                }
                inp_upper, inp_surface = self.ort_sess.run(None, ort_inputs)

                if (step + 1) % interval == 0:
                    tmp_upper.append(inp_upper.copy())
                    tmp_sfc.append(inp_surface.copy())

                curr_time = self.init_time[batch_id] + timedelta(hours=step + 1)
                if is_bdy_swap:
                    inp_upper = self._boundary_swapping(inp_upper, curr_time, 0.1)
                    inp_surface = self._boundary_swapping(inp_surface, curr_time, 0.1)

                if self.cfg.data.add_time_features:
                    time_features = TimeUtil.create_time_features(
                        curr_time, inp_surface.shape[2:4]
                    )  # (H, W, 4)
                    time_features = np.expand_dims(time_features, axis=(0, 1))
                    inp_surface = np.concatenate((inp_surface, time_features), axis=-1)

            # post-process 1, shape = (1, lv, H, W, c) or (Seq, lv, H, W, c)
            tmp_upper = np.concatenate(tmp_upper, axis=0)
            tmp_sfc = np.concatenate(tmp_sfc, axis=0)
            ret.append(
                (
                    input["upper_air"].cpu().numpy(),
                    input["surface"].cpu().numpy(),
                    target["upper_air"].cpu().numpy(),
                    target["surface"].cpu().numpy(),
                    tmp_upper,
                    tmp_sfc,
                )
            )

        # post-process 2, shape = (B, lv, H, W , c) or (B, Seq, lv, H, W, c)
        mapping = PanguLightningModule.get_product_mapping()
        predictions = prediction_postprocess(ret, mapping)
        for product_type, tensor in predictions.items():
            setattr(self, product_type, tensor)

        log.info(f"Batch inference finished at {datetime.now()}")
