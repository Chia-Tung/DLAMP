import logging
from datetime import datetime, timedelta

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import trange

from src.models.lightning_modules import PanguLightningModule
from src.models.model_utils import get_builder
from src.utils import TimeUtil

from .infer_utils import prediction_postprocess
from .inference_base import InferenceBase

log = logging.getLogger(__name__)


class BatchInferenceCkpt(InferenceBase):
    def __init__(self, cfg: DictConfig, eval_cases: list[datetime] | None = None):
        """
        This class can only inference by checkpoint.
        """
        super().__init__(cfg, eval_cases)

    def _setup(self):
        model_builder = get_builder(self.cfg.model.model_name)(
            "predict",
            self.data_list,
            image_shape=self.data_manager.image_shape,
            add_time_features=self.cfg.data.add_time_features,
            **self.cfg.model,
            **self.cfg.lightning,
        )

        self.pl_module = PanguLightningModule.load_from_checkpoint(
            checkpoint_path=self.cfg.inference.best_ckpt,
            test_dataloader=None,
            backbone_model=model_builder._backbone_model(),
        )

        self.pl_module = self.pl_module.cuda()
        self.pl_module.eval()

    def infer(self, bdy_swap_method: dict | None = None):
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
            inp_upper = input["upper_air"].cuda()
            inp_surface = input["surface"].cuda()

            # auto-regression
            tmp_upper, tmp_sfc = [], []
            for step in trange(predict_iters, desc=f"Infer batch {batch_id}"):
                with torch.inference_mode():
                    inp_upper, inp_surface = self.pl_module(inp_upper, inp_surface)
                inp_upper = inp_upper.detach().cpu().numpy()
                inp_surface = inp_surface.detach().cpu().numpy()

                if (step + 1) % interval == 0:
                    tmp_upper.append(inp_upper.copy())
                    tmp_sfc.append(inp_surface.copy())

                curr_time = self.init_time[batch_id] + timedelta(hours=step + 1)
                if self.cfg.data.add_time_features:
                    time_features = TimeUtil.create_time_features(
                        curr_time, inp_surface.shape[2:4]
                    )  # (H, W, 4)
                    time_features = np.expand_dims(time_features, axis=(0, 1))
                    inp_surface = np.concatenate((inp_surface, time_features), axis=-1)

                if bdy_swap_method:
                    inp_upper = self._boundary_swapping(
                        inp_upper,
                        curr_time,
                        bdy_swap_method["name"],
                        bdy_swap_method["n_of_grid"],
                    )
                    inp_surface = self._boundary_swapping(
                        inp_surface,
                        curr_time,
                        bdy_swap_method["name"],
                        bdy_swap_method["n_of_grid"],
                    )

                inp_upper = torch.from_numpy(inp_upper).cuda()
                inp_surface = torch.from_numpy(inp_surface).cuda()

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
