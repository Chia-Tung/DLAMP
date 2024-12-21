import logging
from datetime import datetime, timedelta

import torch
from omegaconf import DictConfig
from tqdm import trange

from src.models.lightning_modules import PanguLightningModule
from src.utils import DataCompose

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
        interval = self.output_itv.seconds // 3600
        predict_iters = (self.showcase_length - 1) * interval

        ret = []
        for batch_id, (input, target) in enumerate(data_loader):
            inp_upper = input["upper_air"].numpy()
            inp_surface = input["surface"].numpy()

            # auto-regression
            tmp_upper, tmp_sfc = [], []
            for step in trange(predict_iters, desc=f"Infer batch {batch_id}"):
                ort_inputs = {
                    self.ort_sess.get_inputs()[0].name: inp_upper,
                    self.ort_sess.get_inputs()[1].name: inp_surface,
                }
                inp_upper, inp_surface = self.ort_sess.run(None, ort_inputs)
                if (step + 1) % interval == 0:
                    tmp_upper.append(torch.from_numpy(inp_upper))
                    tmp_sfc.append(torch.from_numpy(inp_surface))
                if is_bdy_swap:
                    curr_time = self.init_time[batch_id] + timedelta(hours=step + 1)
                    inp_upper = self.boundary_swapping(inp_upper, curr_time, 0.1)
                    inp_surface = self.boundary_swapping(inp_surface, curr_time, 0.1)

            # post-process 1, shape = (1, lv, H, W, c) or (Seq, lv, H, W, c)
            tmp_upper = torch.cat(tmp_upper, dim=0)
            tmp_sfc = torch.cat(tmp_sfc, dim=0)
            ret.append(
                (
                    input["upper_air"],
                    input["surface"],
                    target["upper_air"],
                    target["surface"],
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

    def get_figure_materials(self, case_dt: datetime, data_compose: DataCompose):
        pass

    def get_infer_outputs_from_dt(
        self, dt: datetime, phase: str, data_compose: DataCompose
    ) -> torch.Tensor:
        time_idx = self.init_time.index(dt)
        if data_compose.level.is_surface():
            var_idx = self.surface_vars.index(data_compose.var_name)
            return getattr(self, f"{phase}_surface")[
                time_idx : time_idx + self.showcase_length, 0, :, :, var_idx
            ]
        else:
            level_idx = self.pressure_lv.index(data_compose.level)
            var_idx = self.upper_vars.index(data_compose.var_name)
            return getattr(self, f"{phase}_upper")[
                time_idx : time_idx + self.showcase_length, level_idx, :, :, var_idx
            ]
